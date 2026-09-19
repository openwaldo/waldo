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
	"github.com/openwaldo/waldo/internal/training"
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
	if backend != training.BackendAuto && backend != training.BackendTorchTitan {
		return fmt.Errorf("--hostfile requires TorchTitan, but model.backend=%s; set model.backend=auto or torchtitan", backend)
	}
	var hostfile trainingHostfile
	if path != "" {
		hostfile, err = loadTrainingHostfile(path)
	} else {
		hostfile, err = loadFuzzballTrainingHostlist()
	}
	if err != nil {
		return err
	}
	if complete, err := runCompletedComposeNoop(commandContext, args, stdout, stderr); err != nil {
		return err
	} else if complete {
		fmt.Fprintln(stderr, "multi-host            complete; no training stages required")
		return nil
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
	)
	if err != nil {
		return fmt.Errorf("resolve multi-host node-local cache: %w", err)
	}
	if err := cache.EnsureScratch(); err != nil {
		return fmt.Errorf("prepare multi-host node-local cache: %w", err)
	}
	session, err := startHostfileSession(commandContext.Execution, hostfile, cluster, cache, stderr)
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
	trainErr := runModelTrainWithCluster(commandContext, args, cluster, handoff, stdout, stderr)
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
	ctx            context.Context
	cancel         context.CancelFunc
	hostfile       trainingHostfile
	cluster        training.Cluster
	binary         string
	binarySHA256   string
	remoteBinary   string
	remoteRoot     string
	resumeRoot     string
	resumeStaged   bool
	pythonDir      string
	cacheRoot      string
	cacheScratch   string
	cacheMaxBytes  int64
	cacheMirrors   []string
	workers        []*hostfileWorker
	workersStarted bool
	output         io.Writer
	outputMu       sync.Mutex
	publishMu      sync.Mutex
}

const hostfileWorkerExitGrace = 10 * time.Second

var hostfileStageReadyTimeout = 24 * time.Hour

func startHostfileSession(ctx context.Context, hostfile trainingHostfile, cluster training.Cluster, cache *lookaside.Cache, output io.Writer) (*hostfileSession, error) {
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
		cacheRoot:    cache.Root(), cacheScratch: cache.Scratch(), cacheMaxBytes: cache.MaxBytes(), cacheMirrors: cache.Mirrors(),
		output: output,
	}
	local, err := inspectHostfileTorchTitan(sessionContext)
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

func (session *hostfileSession) stageBinary(host string) error {
	file, err := os.Open(session.binary)
	if err != nil {
		return err
	}
	defer file.Close()
	temporary := session.remoteBinary + ".tmp"
	remote := fmt.Sprintf("umask 077; mkdir -p %s; cat > %s; chmod 700 %s; test \"$(sha256sum %s | cut -d' ' -f1)\" = %s; mv -f %s %s",
		shellQuote(session.remoteRoot), shellQuote(temporary), shellQuote(temporary), shellQuote(temporary), shellQuote(session.binarySHA256), shellQuote(temporary), shellQuote(session.remoteBinary))
	command := session.remoteCommand(host, remote)
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
	var capabilities training.TorchTitanHost
	if err := json.Unmarshal([]byte(stdout.String()), &capabilities); err != nil {
		return training.TorchTitanHost{}, fmt.Errorf("decode TorchTitan preflight from %s: %w%s", host, err, commandOutput(stdout.String()))
	}
	return capabilities, nil
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
	if check {
		return append(arguments, "--check")
	}
	scratch := fmt.Sprintf("%s/runs/%s/node-%d", session.remoteRoot, session.cluster.RendezvousID, rank)
	arguments = append(arguments,
		"--plan-stdin", "--scratch", scratch,
		"--cache-root", session.cacheRoot,
		"--cache-scratch", session.cacheScratch,
		"--cache-max-bytes", fmt.Sprintf("%d", session.cacheMaxBytes),
	)
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
	go session.copyWorkerStdout(worker, stdout)
	go session.copyWorkerOutput(host, stderr)
	go func() {
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
	return fmt.Sprintf("umask 077; env PATH=%s %s <&0 & child=$!; printf '%%s\\n' \"$child\" > %s; trap 'kill -TERM \"$child\" 2>/dev/null || true; wait \"$child\"; exit 143' HUP INT TERM; wait \"$child\"; status=$?; rm -f -- %s; exit \"$status\"",
		shellQuote(path), joinRemoteArguments(arguments), pidPath, pidPath)
}

func (session *hostfileSession) copyWorkerOutput(host string, source io.Reader) {
	scanner := bufio.NewScanner(source)
	for scanner.Scan() {
		session.outputMu.Lock()
		fmt.Fprintf(session.output, "[%s] %s\n", host, scanner.Text())
		session.outputMu.Unlock()
	}
}

func (session *hostfileSession) copyWorkerStdout(worker *hostfileWorker, source io.Reader) {
	scanner := bufio.NewScanner(source)
	for scanner.Scan() {
		line := scanner.Text()
		var ready hostfileStageReady
		if json.Unmarshal([]byte(line), &ready) == nil && ready.Kind == hostfileStageReadyKind {
			worker.ready <- ready
			continue
		}
		session.outputMu.Lock()
		fmt.Fprintf(session.output, "[%s] %s\n", worker.host, line)
		session.outputMu.Unlock()
	}
}

func (session *hostfileSession) publish(plan model.MultiNodePlan) error {
	session.publishMu.Lock()
	defer session.publishMu.Unlock()
	if err := session.startWorkers(); err != nil {
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
		case ready := <-worker.ready:
			if !timer.Stop() {
				<-timer.C
			}
			if ready.Schema != 1 || ready.RunID != plan.RunID || ready.Stage != plan.Stage || ready.StageOrdinal != plan.StageOrdinal || ready.StageCount != plan.StageCount {
				_ = rendezvousListener.Close()
				return fmt.Errorf("host %s acknowledged unexpected stage plan: schema %d, run %q, stage %q (%d/%d); expected schema 1, run %q, stage %q (%d/%d)", worker.host, ready.Schema, ready.RunID, ready.Stage, ready.StageOrdinal, ready.StageCount, plan.RunID, plan.Stage, plan.StageOrdinal, plan.StageCount)
			}
		case <-worker.done:
			if !timer.Stop() {
				<-timer.C
			}
			_ = rendezvousListener.Close()
			return fmt.Errorf("host %s exited before acknowledging stage %d/%d: %v", worker.host, plan.StageOrdinal, plan.StageCount, worker.err)
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
		check := fmt.Sprintf("test -f %s; test \"$(sha256sum %s | cut -d' ' -f1)\" = %s", shellQuote(remotePath), shellQuote(remotePath), shellQuote(artifact.SHA256))
		if err := session.remoteCommand(host, check).Run(); err == nil {
			continue
		}
		file, err := os.Open(localPath)
		if err != nil {
			return fmt.Errorf("stage checkpoint on %s: open %s: %w", host, artifact.Path, err)
		}
		temporary := remotePath + ".waldo-transfer"
		remote := fmt.Sprintf("umask 077; mkdir -p %s; cat > %s; test \"$(sha256sum %s | cut -d' ' -f1)\" = %s; mv -f %s %s",
			shellQuote(filepath.Dir(remotePath)), shellQuote(temporary), shellQuote(temporary), shellQuote(artifact.SHA256), shellQuote(temporary), shellQuote(remotePath))
		command := session.remoteCommand(host, remote)
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
