// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package cli

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/openwaldo/waldo/internal/config"
	"github.com/openwaldo/waldo/internal/lookaside"
	"github.com/openwaldo/waldo/internal/model"
	"github.com/openwaldo/waldo/internal/model/applaunch"
	"github.com/openwaldo/waldo/internal/training"
	"golang.org/x/term"
)

func runModelTrainHostfile(commandContext Context, args []string, path string, stdout, stderr io.Writer) error {
	if err := model.ValidateName(args[0]); err != nil {
		return err
	}
	for _, flag := range []string{"nodes", "rendezvous", "rendezvous-id"} {
		if commandContext.Command != nil && commandContext.Command.Flags().Changed(flag) {
			return fmt.Errorf("--hostfile cannot be combined with --%s; WALDO derives the complete topology from the hostfile", flag)
		}
	}
	port := intOption(commandContext, "rendezvous-port")
	if port < 1 || port > 65535 {
		return fmt.Errorf("--rendezvous-port must be in 1..65535")
	}
	configuration, err := config.Load()
	if err != nil {
		return err
	}
	backend := config.EffectiveModelBackend(configuration)
	var hostfile trainingHostfile
	if path != "" {
		hostfile, err = loadTrainingHostfile(path)
	} else {
		hostfile, err = loadFuzzballTrainingHostlist()
	}
	if err != nil {
		return err
	}
	externalJSON := ""
	composePath, err := trainingComposeInput(args[1:])
	if err != nil {
		return err
	}
	if composePath != "" {
		document, _, err := model.LoadComposeDocument(composePath)
		if err != nil {
			return err
		}
		if document.External != nil {
			raw, err := json.Marshal(document.External)
			if err != nil {
				return err
			}
			externalJSON = string(raw)
		}
	}
	if externalJSON == "" {
		if backend != training.BackendAuto && backend != training.BackendTorchTitan {
			return fmt.Errorf("--hostfile requires TorchTitan, but model.backend=%s; set model.backend=auto or torchtitan", backend)
		}

		if complete, err := runCompletedComposeNoop(commandContext, args, stdout, stderr); err != nil {
			return err
		} else if complete {
			fmt.Fprintln(stderr, "multi-host            complete; no training stages required")
			return nil
		}
	}
	if hostfile.Wrapper == "" {
		hostfile.Wrapper, err = loadFuzzballSSHWrapper()
		if err != nil {
			return err
		}
	}
	rendezvousHost := hostfile.RendezvousHost
	if rendezvousHost == "" {
		rendezvousHost = hostfile.Hosts[0]
	}
	if separator := strings.LastIndex(rendezvousHost, "@"); separator >= 0 {
		rendezvousHost = rendezvousHost[separator+1:]
	}
	cluster := training.Cluster{
		Nodes: len(hostfile.Hosts), NodeRank: 0,
		Rendezvous:   net.JoinHostPort(rendezvousHost, fmt.Sprintf("%d", port)),
		RendezvousID: fmt.Sprintf("hostfile-%d-%d", time.Now().UTC().Unix(), os.Getpid()),
		Interface:    configuration.Model.NCCLInterface, HCA: configuration.Model.NCCLHCA,
	}
	if hostfile.Source == "fuzzball" {
		fmt.Fprintf(stderr, "multi-host topology  discovered %d Fuzzball nodes; remote launch uses %s\n", len(hostfile.Hosts), hostfile.Wrapper)
	}
	cacheRoot, err := config.EffectiveCacheRoot(configuration)
	if err != nil {
		return fmt.Errorf("resolve multi-host node-local cache root: %w", err)
	}
	scratchRoot, err := config.EffectiveScratchRoot(configuration)
	if err != nil {
		return fmt.Errorf("resolve multi-host node-local scratch root: %w", err)
	}
	cache, err := lookaside.NewCache(cacheRoot, nil,
		lookaside.WithMirrors(configuration.Lookaside.Mirrors),
		lookaside.WithPersistentStorage(scratchRoot, config.EffectiveCacheMaxBytes(configuration)),
		lookaside.WithCompletedRetention(configuration.Lookaside.RetainCompleted),
	)
	if err != nil {
		return fmt.Errorf("resolve multi-host node-local cache: %w", err)
	}
	if err := cache.EnsureScratch(); err != nil {
		return fmt.Errorf("prepare multi-host node-local cache: %w", err)
	}
	session, err := startHostfileSession(commandContext.Execution, hostfile, cluster, cache, stderr, externalJSON)
	if err != nil {
		return err
	}
	cluster = session.cluster
	commandContext.Execution = session.ctx
	handoff := &model.MultiNodeHandoff{
		RendezvousID: cluster.RendezvousID, Nodes: cluster.Nodes,
		StageOrdinal: 1, StageCount: 1, PrepareResume: session.prepareResume, Publish: session.publish,
		Cleanup: func() {},
	}
	var trainErr error
	if externalJSON != "" {
		var c model.ExternalCompose
		if err = json.Unmarshal([]byte(externalJSON), &c); err != nil {
			return session.finish(err)
		}
		trainErr = runExternalComposeTrainingCluster(commandContext, args[0], c, cluster, session, stdout, stderr)
	} else {
		trainErr = runModelTrainWithCluster(commandContext, args, cluster, handoff, stdout, stderr)
	}
	finishErr := session.finish(trainErr)
	if finishErr == nil && !session.workersStarted {
		fmt.Fprintln(stderr, "multi-host            complete; no training stages required")
	}
	return finishErr
}

type trainingHostfile struct {
	Path           string
	Hosts          []string
	Source         string
	Wrapper        string
	RendezvousHost string
}

var inspectHostfileTorchTitan = training.InspectTorchTitanHost
var currentTrainingHostname = os.Hostname
var listenHostfileRendezvous = func(address string) (io.Closer, error) {
	return listenTrainingRendezvous(address)
}
var dialTrainingRendezvous = func(ctx context.Context, address string) error {
	dialer := net.Dialer{Timeout: 5 * time.Second}
	connection, err := dialer.DialContext(ctx, "tcp", address)
	if err != nil {
		return err
	}
	return connection.Close()
}

func loadTrainingHostfile(path string) (trainingHostfile, error) {
	file, err := os.Open(path)
	if err != nil {
		return trainingHostfile{}, fmt.Errorf("open hostfile: %w", err)
	}
	defer file.Close()
	result := trainingHostfile{Path: path, Source: "hostfile"}
	seen := map[string]bool{}
	scanner := bufio.NewScanner(file)
	for line := 1; scanner.Scan(); line++ {
		value := strings.TrimSpace(strings.SplitN(scanner.Text(), "#", 2)[0])
		if value == "" {
			continue
		}
		fields := strings.Fields(value)
		if len(fields) != 1 {
			return trainingHostfile{}, fmt.Errorf("hostfile %s line %d must contain exactly one host; GPU slots are discovered automatically", path, line)
		}
		host := fields[0]
		if strings.HasPrefix(host, "-") || strings.ContainsAny(host, `/\\`) {
			return trainingHostfile{}, fmt.Errorf("hostfile %s line %d has invalid host %q", path, line, host)
		}
		if seen[host] {
			return trainingHostfile{}, fmt.Errorf("hostfile %s repeats host %q", path, host)
		}
		seen[host] = true
		result.Hosts = append(result.Hosts, host)
	}
	if err := scanner.Err(); err != nil {
		return trainingHostfile{}, fmt.Errorf("read hostfile: %w", err)
	}
	if len(result.Hosts) < 2 {
		return trainingHostfile{}, fmt.Errorf("hostfile %s must list at least two hosts, with the local rank-0 host first", path)
	}
	return result, nil
}

func loadFuzzballTrainingHostlist() (trainingHostfile, error) {
	raw := strings.TrimSpace(os.Getenv("MULTINODE_HOSTLIST_NOSLOTS"))
	if raw == "" {
		return trainingHostfile{}, fmt.Errorf("MULTINODE_HOSTLIST_NOSLOTS is empty")
	}
	result := trainingHostfile{Path: "MULTINODE_HOSTLIST_NOSLOTS", Source: "fuzzball"}
	seen := map[string]bool{}
	for index, item := range strings.Split(raw, ",") {
		host := strings.TrimSpace(item)
		if host == "" {
			return trainingHostfile{}, fmt.Errorf("MULTINODE_HOSTLIST_NOSLOTS entry %d is empty", index+1)
		}
		if strings.Contains(host, ":") {
			return trainingHostfile{}, fmt.Errorf("MULTINODE_HOSTLIST_NOSLOTS entry %d %q contains a slot count or port; use host names without slots", index+1, host)
		}
		if strings.HasPrefix(host, "-") || strings.ContainsAny(host, `/\\`) {
			return trainingHostfile{}, fmt.Errorf("MULTINODE_HOSTLIST_NOSLOTS entry %d has invalid host %q", index+1, host)
		}
		if seen[host] {
			return trainingHostfile{}, fmt.Errorf("MULTINODE_HOSTLIST_NOSLOTS repeats host %q", host)
		}
		seen[host] = true
		result.Hosts = append(result.Hosts, host)
	}
	if len(result.Hosts) < 2 {
		return trainingHostfile{}, fmt.Errorf("MULTINODE_HOSTLIST_NOSLOTS must list at least two hosts")
	}
	localHost, err := currentTrainingHostname()
	if err != nil {
		return trainingHostfile{}, fmt.Errorf("determine local Fuzzball host: %w", err)
	}
	localIndex := -1
	for index, host := range result.Hosts {
		if host == localHost {
			localIndex = index
			break
		}
	}
	if localIndex < 0 {
		return trainingHostfile{}, fmt.Errorf("MULTINODE_HOSTLIST_NOSLOTS does not contain local rank-0 host %q", localHost)
	}
	if localIndex > 0 {
		ordered := []string{localHost}
		ordered = append(ordered, result.Hosts[:localIndex]...)
		ordered = append(ordered, result.Hosts[localIndex+1:]...)
		result.Hosts = ordered
	}
	result.RendezvousHost = strings.TrimSpace(os.Getenv("MULTINODE_NODE_IP"))
	if result.RendezvousHost != "" && net.ParseIP(result.RendezvousHost) == nil {
		return trainingHostfile{}, fmt.Errorf("MULTINODE_NODE_IP must be an IP address, got %q", result.RendezvousHost)
	}
	result.Wrapper, err = loadFuzzballSSHWrapper()
	if err != nil {
		return trainingHostfile{}, err
	}
	if result.Wrapper == "" {
		return trainingHostfile{}, fmt.Errorf("MULTINODE_SSH_WRAPPER is required when MULTINODE_HOSTLIST_NOSLOTS selects Fuzzball multi-node training")
	}
	return result, nil
}

func loadFuzzballSSHWrapper() (string, error) {
	sshWrapper := strings.TrimSpace(os.Getenv("MULTINODE_SSH_WRAPPER"))
	rshWrapper := strings.TrimSpace(os.Getenv("MULTINODE_RSH_WRAPPER"))
	if sshWrapper != "" && rshWrapper != "" && sshWrapper != rshWrapper {
		return "", fmt.Errorf("MULTINODE_SSH_WRAPPER and MULTINODE_RSH_WRAPPER name different launchers")
	}
	wrapper := sshWrapper
	if wrapper == "" {
		wrapper = rshWrapper
	}
	if wrapper == "" {
		return "", nil
	}
	if !filepath.IsAbs(wrapper) {
		return "", fmt.Errorf("Fuzzball SSH wrapper must be an absolute path, got %q", wrapper)
	}
	info, err := os.Stat(wrapper)
	if err != nil {
		return "", fmt.Errorf("inspect Fuzzball SSH wrapper %s: %w", wrapper, err)
	}
	if !info.Mode().IsRegular() || info.Mode().Perm()&0o111 == 0 {
		return "", fmt.Errorf("Fuzzball SSH wrapper %s is not an executable file", wrapper)
	}
	return wrapper, nil
}

type hostfileWorker struct {
	host    string
	rank    int
	stdin   io.WriteCloser
	command *exec.Cmd
	ready   chan hostfileStageReady
	done    chan struct{}
	err     error
}

const hostfileStageReadyKind = "waldo-hostfile-stage-ready"

type hostfileStageReady struct {
	Kind         string `json:"kind"`
	Schema       int    `json:"schema"`
	RunID        string `json:"run_id"`
	Stage        string `json:"stage"`
	StageOrdinal int    `json:"stage_ordinal"`
	StageCount   int    `json:"stage_count"`
}

type hostfileSession struct {
	externalLaunches     map[string][]applaunch.Evidence
	externalCompose      string
	ctx                  context.Context
	cancel               context.CancelFunc
	hostfile             trainingHostfile
	cluster              training.Cluster
	binary               string
	binarySHA256         string
	remoteBinary         string
	remoteRoot           string
	resumeRoot           string
	resumeStaged         bool
	pythonDir            string
	cacheRoot            string
	cacheScratch         string
	cacheMaxBytes        int64
	cacheRetainCompleted bool
	cacheMirrors         []string
	workers              []*hostfileWorker
	workersStarted       bool
	output               io.Writer
	outputTerminal       bool
	remoteBarActive      bool
	outputMu             sync.Mutex
	publishMu            sync.Mutex
}

const hostfileWorkerExitGrace = 10 * time.Second

var hostfileStageReadyTimeout = 24 * time.Hour

func startHostfileSession(ctx context.Context, hostfile trainingHostfile, cluster training.Cluster, cache *lookaside.Cache, output io.Writer, external ...string) (*hostfileSession, error) {
	if cache == nil {
		return nil, fmt.Errorf("multi-host launch requires a node-local cache configuration")
	}
	binary, err := os.Executable()
	if err != nil {
		return nil, fmt.Errorf("locate current WALDO executable: %w", err)
	}
	binary, err = filepath.EvalSymlinks(binary)
	if err != nil {
		return nil, fmt.Errorf("resolve current WALDO executable: %w", err)
	}
	digest, err := fileSHA256(binary)
	if err != nil {
		return nil, err
	}
	sessionContext, cancel := context.WithCancel(ctx)
	remoteRoot := filepath.Join(cache.Scratch(), "multinode", digest, cluster.RendezvousID)
	session := &hostfileSession{
		ctx: sessionContext, cancel: cancel, hostfile: hostfile, cluster: cluster,
		binary: binary, binarySHA256: digest, remoteRoot: remoteRoot,
		remoteBinary: remoteRoot + "/waldo",
		resumeRoot:   filepath.Join(remoteRoot, "resume"),
		cacheRoot:    cache.Root(), cacheScratch: cache.Scratch(), cacheMaxBytes: cache.MaxBytes(),
		cacheRetainCompleted: cache.RetainCompleted(), cacheMirrors: cache.Mirrors(),
		output: output, outputTerminal: terminalWriter(output),
	}
	var local training.TorchTitanHost
	if len(external) > 0 && external[0] != "" {
		session.externalCompose = external[0]
		session.externalLaunches = map[string][]applaunch.Evidence{}
		local, err = externalHostCapabilities(sessionContext, external[0])
	} else {
		local, err = inspectHostfileTorchTitan(sessionContext)
	}
	if err != nil {
		cancel()
		return nil, fmt.Errorf("rank 0 TorchTitan preflight: %w", err)
	}
	if err := training.ValidateTorchTitanHostConfiguration(local, cluster); err != nil {
		cancel()
		return nil, fmt.Errorf("rank 0 TorchTitan preflight: %w", err)
	}
	cluster.WorldSize = cluster.Nodes * len(local.Accelerators)
	session.cluster = cluster
	rendezvousListener, err := listenHostfileRendezvous(cluster.Rendezvous)
	if err != nil {
		cancel()
		return nil, fmt.Errorf("rank 0 rendezvous preflight: %w", err)
	}
	defer rendezvousListener.Close()
	session.pythonDir = filepath.Dir(local.Python)
	fmt.Fprintf(output, "multi-host preflight  rank 0 ready: %s\n", torchTitanHostSummary(local))
	for rank, host := range hostfile.Hosts[1:] {
		nodeRank := rank + 1
		fmt.Fprintf(output, "multi-host preflight  staging WALDO on %s\n", host)
		if err := session.stageBinary(host); err != nil {
			session.abort()
			return nil, err
		}
		remote, err := session.probeHost(host, nodeRank)
		if err != nil {
			session.abort()
			return nil, err
		}
		if err := compareTorchTitanHosts(local, remote); err != nil {
			session.abort()
			if session.externalCompose != "" {
				return nil, fmt.Errorf("host %s external trainer preflight: %w; every node must have matching runtime and accelerator topology", host, err)
			}
			return nil, torchTitanHostMismatchError(host, err)
		}
		if err := training.ValidateTorchTitanHostConfiguration(remote, cluster); err != nil {
			session.abort()
			return nil, fmt.Errorf("host %s TorchTitan preflight: %w", host, err)
		}
		fmt.Fprintf(output, "multi-host preflight  %s ready: %s\n", host, torchTitanHostSummary(remote))
	}
	if err := rendezvousListener.Close(); err != nil {
		session.abort()
		return nil, fmt.Errorf("close rank 0 rendezvous preflight listener: %w", err)
	}
	return session, nil
}

func listenTrainingRendezvous(address string) (net.Listener, error) {
	_, port, err := net.SplitHostPort(address)
	if err != nil {
		return nil, err
	}
	listener, err := net.Listen("tcp", net.JoinHostPort("", port))
	if err != nil {
		return nil, fmt.Errorf("port %s is unavailable: %w", port, err)
	}
	go func() {
		for {
			connection, err := listener.Accept()
			if err != nil {
				return
			}
			_ = connection.Close()
		}
	}()
	return listener, nil
}

func checkTrainingRendezvous(ctx context.Context, address string) error {
	if err := dialTrainingRendezvous(ctx, address); err != nil {
		return fmt.Errorf("cannot reach rank 0 at %s: %w; verify name resolution, routing, and firewall rules", address, err)
	}
	return nil
}

func torchTitanHostMismatchError(host string, mismatch error) error {
	return fmt.Errorf("host %s is incompatible with rank 0: %w\n\ninstall the tested Python runtime on host %s, then retry:\n%s", host, mismatch, host, training.TorchTitanPythonInstallScript())
}

func fileSHA256(path string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer file.Close()
	digest := sha256.New()
	if _, err := io.Copy(digest, file); err != nil {
		return "", err
	}
	return hex.EncodeToString(digest.Sum(nil)), nil
}

func (session *hostfileSession) remoteCommand(arguments ...string) *exec.Cmd {
	return session.remoteCommandContext(session.ctx, arguments...)
}

func (session *hostfileSession) remoteCommandContext(ctx context.Context, arguments ...string) *exec.Cmd {
	launcher := "ssh"
	base := []string{"-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "--"}
	if session.hostfile.Wrapper != "" {
		launcher = session.hostfile.Wrapper
		base = nil
	}
	command := exec.CommandContext(ctx, launcher, append(base, arguments...)...)
	command.Cancel = func() error {
		if command.Process == nil {
			return os.ErrProcessDone
		}
		err := command.Process.Signal(syscall.SIGTERM)
		if errors.Is(err, os.ErrProcessDone) || errors.Is(err, syscall.ESRCH) {
			return os.ErrProcessDone
		}
		return err
	}
	command.WaitDelay = hostfileWorkerExitGrace
	return command
}

// remoteStagingFunctions defines POSIX shell functions for staging files on
// secondary hosts. waldo_private_dir creates an absolute directory and
// verifies that it, every ancestor, and every ancestor of its resolved path
// are owned by the remote user or root and cannot be modified by another
// account. Directories owned by the remote user must not be writable by their
// group or other users. Root-owned directories writable by others, such as
// /tmp, must be sticky. On Linux the group permission bits report the POSIX ACL
// mask, which bounds every named ACL entry; other systems' extended ACLs cannot
// be evaluated from the mode and are refused. Only portable tools are used so
// the same script runs on Linux hosts and in macOS unit tests.
const remoteStagingFunctions = `waldo_uid=$(id -u)
waldo_system=$(uname -s)
waldo_check_entry() {
	waldo_listing=$(LC_ALL=C ls -ldn -- "$1") || return 1
	set -- "$1" $waldo_listing
	case $4 in
	"$waldo_uid") waldo_owner=user ;;
	0) waldo_owner=root ;;
	*) echo "$1 is owned by uid $4, not uid $waldo_uid or root" >&2; return 1 ;;
	esac
	case $2 in l*) return 0 ;; esac
	case $2 in
	??????????+*)
		if [ "$waldo_system" != Linux ]; then
			echo "$1 has an access control list that cannot be verified" >&2
			return 1
		fi
		;;
	esac
	case $2 in ?????w*|????????w*) ;; *) return 0 ;; esac
	if [ "$waldo_owner" = root ]; then
		case $2 in ?????????[tT]*) return 0 ;; esac
		echo "$1 is writable by other users and is not sticky" >&2
		return 1
	fi
	echo "$1 is writable by its group or other users; remove that access with chmod go-w" >&2
	return 1
}
waldo_check_ancestors() {
	waldo_path=$1
	while :; do
		waldo_check_entry "$waldo_path" || return 1
		[ "$waldo_path" = / ] && return 0
		waldo_path=$(dirname -- "$waldo_path")
	done
}
waldo_private_dir() {
	case $1 in /*) ;; *) echo "staging directory $1 is not absolute" >&2; return 1 ;; esac
	mkdir -p -- "$1" || return 1
	waldo_physical=$(cd -P -- "$1" && pwd -P) || return 1
	waldo_check_ancestors "$1" || return 1
	waldo_check_ancestors "$waldo_physical"
}
waldo_sha256() {
	if command -v sha256sum >/dev/null 2>&1; then
		waldo_sum=$(sha256sum -- "$1")
	else
		waldo_sum=$(shasum -a 256 -- "$1")
	fi
	printf '%s\n' "${waldo_sum%% *}"
}
`

// remoteStageScript returns a script that receives a file on standard input
// and publishes it at target only after its SHA-256 matches digest. The upload
// goes to a unique temporary file inside the verified private directory and
// is removed on any failure.
func remoteStageScript(target, digest, mode, label string) string {
	directory := shellQuote(filepath.Dir(target))
	sum := shellQuote(digest)
	return "set -euf\numask 077\n" + remoteStagingFunctions +
		"waldo_private_dir " + directory + "\n" +
		"tmp=$(mktemp " + directory + "/.waldo-transfer.XXXXXX)\n" +
		"trap 'rm -f -- \"$tmp\"' EXIT\n" +
		"cat > \"$tmp\"\n" +
		"chmod " + mode + " \"$tmp\"\n" +
		"sum=$(waldo_sha256 \"$tmp\")\n" +
		"if [ \"$sum\" != " + sum + " ]; then echo \"staged " + label + " SHA-256 $sum does not match \"" + sum + " >&2; exit 1; fi\n" +
		"mv -f -- \"$tmp\" " + shellQuote(target) + "\n"
}

// remoteVerifiedFileScript succeeds only when target already exists inside a
// verified private directory with the expected SHA-256.
func remoteVerifiedFileScript(target, digest string) string {
	return "set -euf\numask 077\n" + remoteStagingFunctions +
		"waldo_private_dir " + shellQuote(filepath.Dir(target)) + "\n" +
		"test -f " + shellQuote(target) + "\n" +
		"test \"$(waldo_sha256 " + shellQuote(target) + ")\" = " + shellQuote(digest) + "\n"
}

func (session *hostfileSession) stageBinary(host string) error {
	file, err := os.Open(session.binary)
	if err != nil {
		return err
	}
	defer file.Close()
	command := session.remoteCommand(host, remoteStageScript(session.remoteBinary, session.binarySHA256, "700", "WALDO"))
	command.Stdin = file
	var output strings.Builder
	command.Stdout, command.Stderr = &output, &output
	if err := command.Run(); err != nil {
		return fmt.Errorf("stage WALDO on %s: %w%s", host, err, commandOutput(output.String()))
	}
	return nil
}

func (session *hostfileSession) probeHost(host string, rank int) (training.TorchTitanHost, error) {
	arguments := session.workerArguments(rank, true)
	command := session.remoteCommand(host, session.remoteInvocation(arguments))
	var stdout, stderr strings.Builder
	command.Stdout, command.Stderr = &stdout, &stderr
	if err := command.Run(); err != nil {
		return training.TorchTitanHost{}, fmt.Errorf("host %s is not ready for TorchTitan%s\ncorrect the reported condition on host %s, then retry the training command: %w", host, commandOutput(stderr.String()), host, err)
	}
	var result externalHostPreflight
	if err := json.Unmarshal([]byte(stdout.String()), &result); err != nil {
		return training.TorchTitanHost{}, fmt.Errorf("decode TorchTitan preflight from %s: %w%s", host, err, commandOutput(stdout.String()))
	}
	if session.externalCompose != "" {
		var compose model.ExternalCompose
		if err := json.Unmarshal([]byte(session.externalCompose), &compose); err != nil {
			return training.TorchTitanHost{}, err
		}
		if len(result.Launches) != len(compose.Stages) {
			return training.TorchTitanHost{}, fmt.Errorf("host %s omitted external launch evidence", host)
		}
		for _, stage := range compose.Stages {
			evidence, ok := result.Launches[stage.Name]
			if !ok || evidence.Profile != "distributed" || evidence.Launcher != "torchrun" || evidence.World.NodeRank != rank || evidence.World.Nodes != session.cluster.Nodes {
				return training.TorchTitanHost{}, fmt.Errorf("host %s returned inconsistent external launch evidence for %s", host, stage.Name)
			}
			session.externalLaunches[stage.Name] = append(session.externalLaunches[stage.Name], evidence)
		}
	}
	return result.TorchTitanHost, nil
}

func torchTitanHostSummary(host training.TorchTitanHost) string {
	interconnect := host.LocalInterconnect
	switch interconnect {
	case "nvlink":
		interconnect = "NVLink between local GPUs"
	case "gpu-peer-to-peer":
		interconnect = "direct peer-to-peer links between local GPUs"
	case "pcie":
		interconnect = "PCIe between local GPUs"
	case "single-gpu":
		interconnect = "one local GPU"
	case "":
		interconnect = "unknown local interconnect"
	}
	if host.TorchTitanVersion == "external" {
		return fmt.Sprintf("%d GPUs (%s), Python %s, PyTorch %s (external trainer)", len(host.Accelerators), interconnect, host.PythonVersion, host.TorchVersion)
	}
	return fmt.Sprintf("%d GPUs (%s), Python %s, PyTorch %s, TorchTitan %s", len(host.Accelerators), interconnect, host.PythonVersion, host.TorchVersion, host.TorchTitanVersion)
}

func compareTorchTitanHosts(primary, secondary training.TorchTitanHost) error {
	if primary.PythonVersion != secondary.PythonVersion || primary.TorchVersion != secondary.TorchVersion || primary.TorchTitanVersion != secondary.TorchTitanVersion {
		return fmt.Errorf("runtime differs (Python %s/%s, PyTorch %s/%s, TorchTitan %s/%s)", primary.PythonVersion, secondary.PythonVersion, primary.TorchVersion, secondary.TorchVersion, primary.TorchTitanVersion, secondary.TorchTitanVersion)
	}
	if !reflect.DeepEqual(primary.Accelerators, secondary.Accelerators) {
		return fmt.Errorf("visible accelerator topology differs: rank 0 has %v, secondary has %v", primary.Accelerators, secondary.Accelerators)
	}
	if primary.LocalInterconnect != secondary.LocalInterconnect {
		return fmt.Errorf("local GPU interconnect differs: rank 0 has %s, secondary has %s", primary.LocalInterconnect, secondary.LocalInterconnect)
	}
	return nil
}

func (session *hostfileSession) workerArguments(rank int, check bool) []string {
	arguments := []string{session.remoteBinary, "--json", "model", "train-worker",
		"--nodes", fmt.Sprintf("%d", session.cluster.Nodes),
		"--node-rank", fmt.Sprintf("%d", rank),
		"--rendezvous", session.cluster.Rendezvous,
		"--rendezvous-id", session.cluster.RendezvousID,
		"--nccl-interface", session.cluster.Interface,
		"--nccl-hca", session.cluster.HCA,
	}
	if session.externalCompose != "" {
		python := os.Getenv("WALDO_APPLICATION_PYTHON")
		if python == "" {
			python = "python3"
		}
		arguments = append(arguments, "--external-compose-json", session.externalCompose, "--application-python", python)
	}
	scratch := fmt.Sprintf("%s/runs/%s/node-%d", session.remoteRoot, session.cluster.RendezvousID, rank)
	if check {
		if session.externalCompose != "" {
			arguments = append(arguments, "--scratch", scratch)
		}
		return append(arguments, "--check")
	}
	arguments = append(arguments,
		"--plan-stdin", "--scratch", scratch,
		"--cache-root", session.cacheRoot,
		"--cache-scratch", session.cacheScratch,
		"--cache-max-bytes", fmt.Sprintf("%d", session.cacheMaxBytes),
	)
	if session.cacheRetainCompleted {
		arguments = append(arguments, "--cache-retain-completed")
	}
	for _, mirror := range session.cacheMirrors {
		arguments = append(arguments, "--cache-mirror", mirror)
	}
	return arguments
}

func (session *hostfileSession) startWorker(host string, rank int) (*hostfileWorker, error) {
	command := session.remoteCommand(host, session.remoteWorkerInvocation(rank, session.workerArguments(rank, false)))
	command.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	stdin, err := command.StdinPipe()
	if err != nil {
		return nil, err
	}
	stdout, err := command.StdoutPipe()
	if err != nil {
		return nil, err
	}
	stderr, err := command.StderrPipe()
	if err != nil {
		return nil, err
	}
	if err := command.Start(); err != nil {
		return nil, fmt.Errorf("start training worker on %s: %w", host, err)
	}
	worker := &hostfileWorker{
		host: host, rank: rank, stdin: stdin, command: command,
		ready: make(chan hostfileStageReady, 1), done: make(chan struct{}),
	}
	var readers sync.WaitGroup
	readers.Add(2)
	go func() { defer readers.Done(); session.copyWorkerStdout(worker, stdout); close(worker.ready) }()
	go func() { defer readers.Done(); session.copyWorkerOutput(host, stderr) }()
	go func() {
		// StdoutPipe and StderrPipe must be drained before Wait closes them.
		readers.Wait()
		worker.err = command.Wait()
		close(worker.done)
		if worker.err != nil && session.ctx.Err() == nil {
			session.cancel()
		}
	}()
	return worker, nil
}

func (session *hostfileSession) secondaryHosts() []string {
	if len(session.hostfile.Hosts) < 2 {
		return nil
	}
	return session.hostfile.Hosts[1:]
}

func (session *hostfileSession) startWorkers() error {
	if session.workersStarted {
		return nil
	}
	for rank, host := range session.secondaryHosts() {
		worker, err := session.startWorker(host, rank+1)
		if err != nil {
			return err
		}
		session.workers = append(session.workers, worker)
	}
	session.workersStarted = true
	return nil
}

func (session *hostfileSession) remoteInvocation(arguments []string) string {
	path := strings.Join([]string{session.pythonDir, "/usr/local/bin", "/usr/bin", "/bin"}, ":")
	return "PATH=" + shellQuote(path) + " " + joinRemoteArguments(arguments)
}

func (session *hostfileSession) workerPIDPath(rank int) string {
	return filepath.Join(session.remoteRoot, fmt.Sprintf("worker-%d.pid", rank))
}

func (session *hostfileSession) remoteWorkerInvocation(rank int, arguments []string) string {
	path := strings.Join([]string{session.pythonDir, "/usr/local/bin", "/usr/bin", "/bin"}, ":")
	pidPath := shellQuote(session.workerPIDPath(rank))
	// POSIX shells may connect an asynchronous command's standard input to
	// /dev/null when job control is disabled. Preserve the SSH stream on a
	// dedicated descriptor before starting the worker in the background.
	return fmt.Sprintf("umask 077; exec 3<&0; env PATH=%s %s <&3 & child=$!; exec 3<&-; printf '%%s\\n' \"$child\" > %s; trap 'kill -TERM \"$child\" 2>/dev/null || true; wait \"$child\"; exit 143' HUP INT TERM; wait \"$child\"; status=$?; rm -f -- %s; exit \"$status\"",
		shellQuote(path), joinRemoteArguments(arguments), pidPath, pidPath)
}

func (session *hostfileSession) copyWorkerOutput(host string, source io.Reader) {
	scanner := bufio.NewScanner(source)
	for scanner.Scan() {
		session.writeWorkerLine(host, scanner.Text())
	}
}

func (session *hostfileSession) copyWorkerStdout(worker *hostfileWorker, source io.Reader) {
	ctx := session.ctx
	if ctx == nil {
		ctx = context.Background()
	}
	scanner := bufio.NewScanner(source)
	for scanner.Scan() {
		line := scanner.Text()
		var ready hostfileStageReady
		if json.Unmarshal([]byte(line), &ready) == nil && ready.Kind == hostfileStageReadyKind {
			select {
			case worker.ready <- ready:
			case <-ctx.Done():
				return
			}
			continue
		}
		session.writeWorkerLine(worker.host, line)
	}
}

func terminalWriter(output io.Writer) bool {
	type fileDescriptor interface{ Fd() uintptr }
	writer, ok := output.(fileDescriptor)
	return ok && term.IsTerminal(int(writer.Fd()))
}

func (session *hostfileSession) writeWorkerLine(host, line string) {
	session.outputMu.Lock()
	defer session.outputMu.Unlock()
	if session.outputTerminal && strings.HasPrefix(line, "  materialize [") {
		fmt.Fprintf(session.output, "\r\x1b[K[%s] %s", host, line)
		session.remoteBarActive = !strings.Contains(line, "] 100%")
		if !session.remoteBarActive {
			fmt.Fprintln(session.output)
		}
		return
	}
	if session.remoteBarActive {
		fmt.Fprintln(session.output)
		session.remoteBarActive = false
	}
	fmt.Fprintf(session.output, "[%s] %s\n", host, line)
}

func (session *hostfileSession) publish(plan model.MultiNodePlan) error {
	session.publishMu.Lock()
	defer session.publishMu.Unlock()
	if err := session.startWorkers(); err != nil {
		return err
	}
	if err := session.prepareExternalFiles(&plan); err != nil {
		return err
	}
	if err := session.prepareInitialization(&plan); err != nil {
		return err
	}
	rendezvousListener, err := listenHostfileRendezvous(session.cluster.Rendezvous)
	if err != nil {
		return fmt.Errorf("prepare rank 0 rendezvous for stage %d/%d: %w", plan.StageOrdinal, plan.StageCount, err)
	}
	for _, worker := range session.workers {
		if err := json.NewEncoder(worker.stdin).Encode(plan); err != nil {
			_ = rendezvousListener.Close()
			return fmt.Errorf("send stage %d/%d to %s: %w", plan.StageOrdinal, plan.StageCount, worker.host, err)
		}
	}
	for _, worker := range session.workers {
		timer := time.NewTimer(hostfileStageReadyTimeout)
		select {
		case ready, ok := <-worker.ready:
			if !timer.Stop() {
				<-timer.C
			}
			if !ok {
				_ = rendezvousListener.Close()
				return fmt.Errorf("host %s closed output before acknowledging stage %d/%d", worker.host, plan.StageOrdinal, plan.StageCount)
			}
			if ready.Schema != 1 || ready.RunID != plan.RunID || ready.Stage != plan.Stage || ready.StageOrdinal != plan.StageOrdinal || ready.StageCount != plan.StageCount {
				_ = rendezvousListener.Close()
				return fmt.Errorf("host %s acknowledged unexpected stage plan: schema %d, run %q, stage %q (%d/%d); expected schema 1, run %q, stage %q (%d/%d)", worker.host, ready.Schema, ready.RunID, ready.Stage, ready.StageOrdinal, ready.StageCount, plan.RunID, plan.Stage, plan.StageOrdinal, plan.StageCount)
			}
		case <-session.ctx.Done():
			if !timer.Stop() {
				<-timer.C
			}
			_ = rendezvousListener.Close()
			return fmt.Errorf("host %s did not acknowledge stage %d/%d: %w", worker.host, plan.StageOrdinal, plan.StageCount, session.ctx.Err())
		case <-timer.C:
			_ = rendezvousListener.Close()
			return fmt.Errorf("host %s did not acknowledge stage %d/%d within %s; verify that the host is reachable and its launcher is still running", worker.host, plan.StageOrdinal, plan.StageCount, hostfileStageReadyTimeout)
		}
	}
	if err := rendezvousListener.Close(); err != nil {
		return fmt.Errorf("release rank 0 rendezvous preflight for stage %d/%d: %w", plan.StageOrdinal, plan.StageCount, err)
	}
	return nil
}

func (session *hostfileSession) prepareInitialization(plan *model.MultiNodePlan) error {
	if plan.Initialization == nil {
		return nil
	}
	if plan.Initialization.Path == "" {
		return fmt.Errorf("stage initialization for run %s: source path is missing", plan.RunID)
	}
	artifact := plan.Initialization.Artifact
	target := filepath.Join(session.remoteRoot, "initialization", plan.RunID, filepath.Base(artifact.Path))
	if err := os.MkdirAll(filepath.Dir(target), 0o700); err != nil {
		return fmt.Errorf("create local initialization staging directory: %w", err)
	}
	if err := ensureLocalResumeLink(plan.Initialization.Path, target, artifact); err != nil {
		return fmt.Errorf("stage initialization for run %s: %w", plan.RunID, err)
	}
	for _, worker := range session.workers {
		session.outputMu.Lock()
		fmt.Fprintf(session.output, "multi-host initialize staging %s on %s\n", humanBytes(artifact.Bytes), worker.host)
		session.outputMu.Unlock()
		if err := session.stageResume(worker.host, []training.Artifact{artifact}, []string{plan.Initialization.Path}, []string{target}); err != nil {
			return err
		}
	}
	plan.InitializationPath = target
	return nil
}

func (session *hostfileSession) prepareResume(runID string, resume *training.ResumePoint) error {
	if resume == nil || len(resume.Paths) != len(resume.Checkpoint.Artifacts) || len(resume.Paths) == 0 {
		return fmt.Errorf("checkpoint metadata is incomplete")
	}
	targetDirectory := filepath.Join(session.resumeRoot, runID)
	if err := os.MkdirAll(targetDirectory, 0o700); err != nil {
		return fmt.Errorf("create local checkpoint staging directory: %w", err)
	}
	session.resumeStaged = true
	sources := append([]string(nil), resume.Paths...)
	targets := make([]string, len(sources))
	for index, artifact := range resume.Checkpoint.Artifacts {
		targets[index] = filepath.Join(targetDirectory, filepath.Base(artifact.Path))
		if err := ensureLocalResumeLink(sources[index], targets[index], artifact); err != nil {
			return err
		}
	}
	var total int64
	for _, artifact := range resume.Checkpoint.Artifacts {
		total += artifact.Bytes
	}
	for _, host := range session.secondaryHosts() {
		session.outputMu.Lock()
		fmt.Fprintf(session.output, "multi-host resume     staging checkpoint step %d (%s) on %s\n", resume.Step, humanBytes(total), host)
		session.outputMu.Unlock()
		if err := session.stageResume(host, resume.Checkpoint.Artifacts, sources, targets); err != nil {
			return err
		}
	}
	resume.Paths = targets
	return nil
}

func ensureLocalResumeLink(source, target string, artifact training.Artifact) error {
	if err := model.VerifyArtifactFile(target, artifact); err == nil {
		return nil
	}
	if err := os.Remove(target); err != nil && !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("replace local checkpoint staging path %s: %w", artifact.Path, err)
	}
	if err := os.Symlink(source, target); err != nil {
		return fmt.Errorf("stage local checkpoint artifact %s: %w", artifact.Path, err)
	}
	if err := model.VerifyArtifactFile(target, artifact); err != nil {
		return fmt.Errorf("verify local checkpoint staging link: %w", err)
	}
	return nil
}

func (session *hostfileSession) stageResume(host string, artifacts []training.Artifact, sources, targets []string) error {
	for index, artifact := range artifacts {
		localPath := sources[index]
		remotePath := targets[index]
		if err := session.remoteCommand(host, remoteVerifiedFileScript(remotePath, artifact.SHA256)).Run(); err == nil {
			continue
		}
		file, err := os.Open(localPath)
		if err != nil {
			return fmt.Errorf("stage checkpoint on %s: open %s: %w", host, artifact.Path, err)
		}
		command := session.remoteCommand(host, remoteStageScript(remotePath, artifact.SHA256, "600", "checkpoint"))
		command.Stdin = file
		var output strings.Builder
		command.Stdout, command.Stderr = &output, &output
		runErr := command.Run()
		closeErr := file.Close()
		if runErr != nil {
			return fmt.Errorf("stage checkpoint artifact %s on %s: %w%s", artifact.Path, host, runErr, commandOutput(output.String()))
		}
		if closeErr != nil {
			return fmt.Errorf("stage checkpoint artifact %s on %s: %w", artifact.Path, host, closeErr)
		}
	}
	return nil
}

func (session *hostfileSession) finish(primaryErr error) error {
	if primaryErr != nil {
		session.stopRemoteWorkers()
		session.cancel()
	}
	for _, worker := range session.workers {
		_ = worker.stdin.Close()
	}
	var workerErrors []string
	for _, worker := range session.workers {
		<-worker.done
		if worker.err != nil {
			workerErrors = append(workerErrors, fmt.Sprintf("%s: %v", worker.host, worker.err))
		}
	}
	session.cleanupStaging()
	session.cancel()
	if len(workerErrors) > 0 && (primaryErr == nil || errors.Is(primaryErr, context.Canceled)) {
		return fmt.Errorf("secondary training workers failed: %s", strings.Join(workerErrors, "; "))
	}
	if primaryErr != nil {
		return primaryErr
	}
	return nil
}

func (session *hostfileSession) stopRemoteWorkers() {
	for _, worker := range session.workers {
		pidPath := session.workerPIDPath(worker.rank)
		remote := fmt.Sprintf("if test -r %s; then IFS= read -r pid < %s; case \"$pid\" in ''|*[!0-9]*) exit 1;; esac; kill -TERM \"$pid\" 2>/dev/null || true; fi",
			shellQuote(pidPath), shellQuote(pidPath))
		ctx, cancel := context.WithTimeout(context.Background(), hostfileWorkerExitGrace)
		_ = session.remoteCommandContext(ctx, worker.host, remote).Run()
		cancel()
	}
}

func (session *hostfileSession) cleanupResumeStaging() {
	if !session.resumeStaged {
		return
	}
	if err := os.RemoveAll(session.resumeRoot); err != nil {
		session.outputMu.Lock()
		fmt.Fprintf(session.output, "warning: clean local multi-host checkpoint staging: %v\n", err)
		session.outputMu.Unlock()
	}
	for _, host := range session.secondaryHosts() {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		command := session.remoteCommandContext(ctx, host, "rm -rf -- "+shellQuote(session.resumeRoot))
		err := command.Run()
		cancel()
		if err != nil {
			session.outputMu.Lock()
			fmt.Fprintf(session.output, "warning: clean multi-host checkpoint staging on %s: %v\n", host, err)
			session.outputMu.Unlock()
		}
	}
}

func (session *hostfileSession) cleanupStaging() {
	if err := os.RemoveAll(session.remoteRoot); err != nil {
		session.outputMu.Lock()
		fmt.Fprintf(session.output, "warning: clean local multi-host launch staging: %v\n", err)
		session.outputMu.Unlock()
	}
	for _, host := range session.secondaryHosts() {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		command := session.remoteCommandContext(ctx, host, "rm -rf -- "+shellQuote(session.remoteRoot))
		err := command.Run()
		cancel()
		if err != nil {
			session.outputMu.Lock()
			fmt.Fprintf(session.output, "warning: clean multi-host launch staging on %s: %v\n", host, err)
			session.outputMu.Unlock()
		}
	}
}

func (session *hostfileSession) abort() {
	session.stopRemoteWorkers()
	session.cancel()
	for _, worker := range session.workers {
		_ = worker.stdin.Close()
	}
	for _, worker := range session.workers {
		<-worker.done
	}
	session.cleanupStaging()
}

func joinRemoteArguments(arguments []string) string {
	quoted := make([]string, len(arguments))
	for index, argument := range arguments {
		quoted[index] = shellQuote(argument)
	}
	return strings.Join(quoted, " ")
}

func commandOutput(value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return ""
	}
	return ": " + value
}

func (session *hostfileSession) prepareExternalFiles(plan *model.MultiNodePlan) error {
	if plan.External == nil {
		return nil
	}
	p := plan.External
	if p.Checkpoint != nil {
		var artifacts []training.Artifact
		var sources, targets []string
		targetRoot := filepath.Join(session.remoteRoot, "external-checkpoints", plan.RunID)
		sourceRoot := p.CheckpointDirectory
		err := filepath.WalkDir(filepath.Join(sourceRoot, filepath.Dir(p.Checkpoint.Path)), func(path string, entry os.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if entry.IsDir() {
				return nil
			}
			info, err := entry.Info()
			if err != nil {
				return err
			}
			if !info.Mode().IsRegular() {
				return fmt.Errorf("checkpoint staging requires regular files")
			}
			relative, err := filepath.Rel(sourceRoot, path)
			if err != nil {
				return err
			}
			hash, err := fileSHA256(path)
			if err != nil {
				return err
			}
			artifacts = append(artifacts, training.Artifact{Path: relative, SHA256: hash, Bytes: info.Size()})
			sources = append(sources, path)
			targets = append(targets, filepath.Join(targetRoot, relative))
			return nil
		})
		if err != nil {
			return err
		}
		for _, host := range session.secondaryHosts() {
			if err = session.stageResume(host, artifacts, sources, targets); err != nil {
				return err
			}
		}
		p.CheckpointDirectory = targetRoot
	}
	if p.Begin.Initialization != nil {
		init := *p.Begin.Initialization
		target := filepath.Join(session.remoteRoot, "external-initialization", plan.RunID, filepath.Base(init.Path))
		for _, host := range session.secondaryHosts() {
			if err := session.stageResume(host, []training.Artifact{init.Artifact}, []string{init.Path}, []string{target}); err != nil {
				return err
			}
		}
		init.Path = target
		p.Begin.Initialization = &init
	}
	return nil
}
