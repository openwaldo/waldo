// Copyright (c) 2026 OpenWALDO Project contributors
// SPDX-License-Identifier: Apache-2.0

package index

import "testing"

func TestValidateRightsReview(t *testing.T) {
	valid := RightsReview{
		ReviewedAt: "2026-09-14", Training: RightsApproved,
		CorpusRedistribution: RightsApproved, ModelWeights: RightsApproved,
		Reason: "reviewed upstream declaration", Evidence: []string{"https://example.test/license"},
	}
	if err := ValidateRightsReview(valid); err != nil {
		t.Fatal(err)
	}
	for name, mutate := range map[string]func(*RightsReview){
		"date":     func(review *RightsReview) { review.ReviewedAt = "September 14" },
		"training": func(review *RightsReview) { review.Training = "public" },
		"corpus":   func(review *RightsReview) { review.CorpusRedistribution = "maybe" },
		"weights":  func(review *RightsReview) { review.ModelWeights = "maybe" },
		"reason":   func(review *RightsReview) { review.Reason = "" },
		"evidence": func(review *RightsReview) { review.Evidence = []string{"file:///license"} },
	} {
		t.Run(name, func(t *testing.T) {
			copy := valid
			mutate(&copy)
			if err := ValidateRightsReview(copy); err == nil {
				t.Fatal("invalid rights review accepted")
			}
		})
	}
}
