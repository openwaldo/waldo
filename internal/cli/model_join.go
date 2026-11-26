// Copyright (c) 2026 OpenWALDO Project contributors
// Copyright (c) 2026 CtrlIQ, Inc.
// Copyright (c) 2026 Gregory M. Kurtzer
// SPDX-License-Identifier: Apache-2.0

package cli

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/openwaldo/waldo/internal/join"
)

func runModelJoin(context Context, args []string, stdout, stderr io.Writer) error {
	identity := stringOption(context, "identity")
	if err := validateJoinIdentity(identity); err != nil {
		return err
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return fmt.Errorf("find home directory: %w", err)
	}
	email := stringOption(context, "email")
	if email == "" {
		email = identity + "@waldo.invalid"
	}
	executable, err := os.Executable()
	if err != nil {
		return err
	}
	joiner := &join.Joiner{Waldo: executable,
		Home: filepath.Join(home, ".waldo", "join", identity), Identity: identity,
		Key: stringOption(context, "key"), Email: email, Log: stdout, Rounds: intOption(context, "rounds"),
	}
	if err := joiner.Open(context.Execution, args[0], args[1]); err != nil {
		return err
	}
	wait := time.Duration(intOption(context, "wait")) * time.Second
	failures := 0
	for {
		result, err := joiner.Step(context.Execution)
		if err != nil && failures < joinRetries && context.Execution.Err() == nil && !boolOption(context, "once") {
			failures++
			pause := wait << failures
			fmt.Fprintf(stdout, "%v\nretrying in %s (%d of %d)\n", err, pause, failures, joinRetries)
			if !sleepJoin(context, pause) {
				return context.Execution.Err()
			}
			continue
		}
		if err != nil || result == "done" || result == "stopped" || boolOption(context, "once") {
			return err
		}
		failures = 0
		if result == "idle" && !sleepJoin(context, wait) {
			return context.Execution.Err()
		}
	}
}

// joinRetries is how many times in a row a failed step is retried, waiting twice as long each
// time, so a network drop doesn't end an identity's night.
const joinRetries = 6

func sleepJoin(context Context, wait time.Duration) bool {
	select {
	case <-context.Execution.Done():
		return false
	case <-time.After(wait):
		return true
	}
}

func validateJoinIdentity(identity string) error {
	if identity == "" || strings.ContainsAny(identity, `/\ `) {
		return fmt.Errorf("--identity must be a plain name from the plan's identities")
	}
	return nil
}
