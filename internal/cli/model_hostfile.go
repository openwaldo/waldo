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
	hostfile, err := loadTrainingHostfile(path)
	if err != nil {
		return err
	}
	rendezvousHost := hostfile.Hosts[0]
	if separator := strings.LastIndex(rendezvousHost, "@"); separator >= 0 {
		rendezvousHost = rendezvousHost[separator+1:]
	}
	cluster := training.Cluster{
		Nodes: len(hostfile.Hosts), NodeRank: 0,
		Rendezvous:   net.JoinHostPort(rendezvousHost, fmt.Sprintf("%d", port)),
		RendezvousID: fmt.Sprintf("hostfile-%d-%d", time.Now().UTC().Unix(), os.Getpid()),
		Interface:    configuration.Model.NCCLInterface, HCA: configuration.Model.NCCLHCA,
	}
	session, err := startHostfileSession(commandContext.Execution, hostfile, cluster, stderr)
	if err != nil {
		return err
	}
	commandContext.Execution = session.ctx
	handoff := &model.MultiNodeHandoff{
		RendezvousID: cluster.RendezvousID, Nodes: cluster.Nodes,
		StageOrdinal: 1, StageCount: 1, Publish: session.publish,
		Cleanup: func() {},
	}
	trainErr := runModelTrainWithCluster(commandContext, args, cluster, handoff, stdout, stderr)
	return session.finish(trainErr)
}

type trainingHostfile struct {
	Path  string
	Hosts []string
}

var inspectHostfileTorchTitan = training.InspectTorchTitanHost

func loadTrainingHostfile(path string) (trainingHostfile, error) {
	file, err := os.Open(path)
	if err != nil {
		return trainingHostfile{}, fmt.Errorf("open hostfile: %w", err)
	}
	defer file.Close()
	result := trainingHostfile{Path: path}
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

type hostfileWorker struct {
	host    string
	rank    int
	stdin   io.WriteCloser
	command *exec.Cmd
	done    chan error
}

type hostfileSession struct {
	ctx          context.Context
	cancel       context.CancelFunc
	hostfile     trainingHostfile
	cluster      training.Cluster
	binary       string
	binarySHA256 string
	remoteBinary string
	remoteRoot   string
	pythonDir    string
	workers      []*hostfileWorker
	output       io.Writer
	outputMu     sync.Mutex
	publishMu    sync.Mutex
}

func startHostfileSession(ctx context.Context, hostfile trainingHostfile, cluster training.Cluster, output io.Writer) (*hostfileSession, error) {
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
	remoteRoot := "/tmp/waldo-launch/" + digest
	session := &hostfileSession{
		ctx: sessionContext, cancel: cancel, hostfile: hostfile, cluster: cluster,
		binary: binary, binarySHA256: digest, remoteRoot: remoteRoot,
		remoteBinary: remoteRoot + "/waldo", output: output,
	}
	local, err := inspectHostfileTorchTitan(sessionContext)
	if err != nil {
		cancel()
		return nil, fmt.Errorf("rank 0 TorchTitan preflight: %w", err)
	}
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
			return nil, fmt.Errorf("host %s is incompatible with rank 0: %w", host, err)
		}
		fmt.Fprintf(output, "multi-host preflight  %s ready: %s\n", host, torchTitanHostSummary(remote))
	}
	for rank, host := range hostfile.Hosts[1:] {
		worker, err := session.startWorker(host, rank+1)
		if err != nil {
			session.abort()
			return nil, err
		}
		session.workers = append(session.workers, worker)
	}
	return session, nil
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

func (session *hostfileSession) sshCommand(arguments ...string) *exec.Cmd {
	base := []string{"-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "--"}
	return exec.CommandContext(session.ctx, "ssh", append(base, arguments...)...)
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
	command := session.sshCommand(host, remote)
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
	command := session.sshCommand(host, session.remoteInvocation(arguments))
	var stdout, stderr strings.Builder
	command.Stdout, command.Stderr = &stdout, &stderr
	if err := command.Run(); err != nil {
		return training.TorchTitanHost{}, fmt.Errorf("TorchTitan preflight on %s: %w%s", host, err, commandOutput(stderr.String()))
	}
	var capabilities training.TorchTitanHost
	if err := json.Unmarshal([]byte(stdout.String()), &capabilities); err != nil {
		return training.TorchTitanHost{}, fmt.Errorf("decode TorchTitan preflight from %s: %w%s", host, err, commandOutput(stdout.String()))
	}
	return capabilities, nil
}

func torchTitanHostSummary(host training.TorchTitanHost) string {
	return fmt.Sprintf("%d GPUs, Python %s, PyTorch %s, TorchTitan %s", len(host.Accelerators), host.PythonVersion, host.TorchVersion, host.TorchTitanVersion)
}

func compareTorchTitanHosts(primary, secondary training.TorchTitanHost) error {
	if primary.PythonVersion != secondary.PythonVersion || primary.TorchVersion != secondary.TorchVersion || primary.TorchTitanVersion != secondary.TorchTitanVersion {
		return fmt.Errorf("runtime differs (Python %s/%s, PyTorch %s/%s, TorchTitan %s/%s)", primary.PythonVersion, secondary.PythonVersion, primary.TorchVersion, secondary.TorchVersion, primary.TorchTitanVersion, secondary.TorchTitanVersion)
	}
	if !reflect.DeepEqual(primary.Accelerators, secondary.Accelerators) {
		return fmt.Errorf("visible accelerator topology differs: rank 0 has %v, secondary has %v", primary.Accelerators, secondary.Accelerators)
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
	return append(arguments, "--plan-stdin", "--scratch", scratch)
}

func (session *hostfileSession) startWorker(host string, rank int) (*hostfileWorker, error) {
	command := session.sshCommand(host, session.remoteInvocation(session.workerArguments(rank, false)))
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
	worker := &hostfileWorker{host: host, rank: rank, stdin: stdin, command: command, done: make(chan error, 1)}
	go session.copyWorkerOutput(host, stdout)
	go session.copyWorkerOutput(host, stderr)
	go func() {
		err := command.Wait()
		worker.done <- err
		if err != nil && session.ctx.Err() == nil {
			session.cancel()
		}
	}()
	return worker, nil
}

func (session *hostfileSession) remoteInvocation(arguments []string) string {
	path := strings.Join([]string{session.pythonDir, "/usr/local/bin", "/usr/bin", "/bin"}, ":")
	return "PATH=" + shellQuote(path) + " " + joinRemoteArguments(arguments)
}

func (session *hostfileSession) copyWorkerOutput(host string, source io.Reader) {
	scanner := bufio.NewScanner(source)
	for scanner.Scan() {
		session.outputMu.Lock()
		fmt.Fprintf(session.output, "[%s] %s\n", host, scanner.Text())
		session.outputMu.Unlock()
	}
}

func (session *hostfileSession) publish(plan model.MultiNodePlan) error {
	session.publishMu.Lock()
	defer session.publishMu.Unlock()
	for _, worker := range session.workers {
		if err := json.NewEncoder(worker.stdin).Encode(plan); err != nil {
			return fmt.Errorf("send stage %d/%d to %s: %w", plan.StageOrdinal, plan.StageCount, worker.host, err)
		}
	}
	return nil
}

func (session *hostfileSession) finish(primaryErr error) error {
	if primaryErr != nil {
		session.cancel()
	}
	for _, worker := range session.workers {
		_ = worker.stdin.Close()
	}
	var workerErrors []string
	for _, worker := range session.workers {
		if err := <-worker.done; err != nil {
			workerErrors = append(workerErrors, fmt.Sprintf("%s: %v", worker.host, err))
		}
	}
	session.cancel()
	if len(workerErrors) > 0 && (primaryErr == nil || errors.Is(primaryErr, context.Canceled)) {
		return fmt.Errorf("secondary training workers failed: %s", strings.Join(workerErrors, "; "))
	}
	if primaryErr != nil {
		return primaryErr
	}
	return nil
}

func (session *hostfileSession) abort() {
	session.cancel()
	for _, worker := range session.workers {
		_ = worker.stdin.Close()
		if worker.command.Process != nil {
			_ = syscall.Kill(-worker.command.Process.Pid, syscall.SIGKILL)
		}
	}
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
