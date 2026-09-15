// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0

package index

import (
	"fmt"
	"net/url"
	"strings"
	"time"
)

const (
	RightsApproved        = "approved"
	RightsPrivateResearch = "private-research"
	RightsUnresolved      = "unresolved"
	RightsExcluded        = "excluded"
)

func ValidateRightsReview(review RightsReview) error {
	if _, err := time.Parse("2006-01-02", review.ReviewedAt); err != nil {
		return fmt.Errorf("reviewed_at must be an ISO 8601 date")
	}
	if !oneOf(review.Training, RightsApproved, RightsPrivateResearch, RightsUnresolved, RightsExcluded) {
		return fmt.Errorf("training has unsupported status %q", review.Training)
	}
	if !oneOf(review.CorpusRedistribution, RightsApproved, RightsUnresolved, RightsExcluded) {
		return fmt.Errorf("corpus_redistribution has unsupported status %q", review.CorpusRedistribution)
	}
	if !oneOf(review.ModelWeights, RightsApproved, RightsPrivateResearch, RightsUnresolved, RightsExcluded) {
		return fmt.Errorf("model_weights has unsupported status %q", review.ModelWeights)
	}
	if strings.TrimSpace(review.Reason) == "" {
		return fmt.Errorf("reason is required")
	}
	if len(review.Evidence) == 0 {
		return fmt.Errorf("at least one evidence URL is required")
	}
	for _, value := range review.Evidence {
		parsed, err := url.Parse(value)
		if err != nil || parsed.Scheme != "https" || parsed.Host == "" {
			return fmt.Errorf("evidence URL %q must be absolute HTTPS", value)
		}
	}
	return nil
}

func oneOf(value string, allowed ...string) bool {
	for _, candidate := range allowed {
		if value == candidate {
			return true
		}
	}
	return false
}
