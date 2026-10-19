# Chronological analysis contract

This analysis contract is recorded before any NF-ToN model result. It supplements,
and does not replace, the frozen source, sampling, and role protocol.

- The primary classification endpoint is Macro F1 over the ten source attack
  labels, with zero division handled as zero. Report observed-test-label and
  known-at-fit-only scores separately to expose changes in class coverage.
- Report unseen-at-fit test labels and their row counts. Never move time
  boundaries, transfer future labels into fitting, or remove unknown attacks to
  improve the primary result.
- Rare recall averages the fit-defined rare labels with observed test support;
  explicitly report the denominator. It is undefined if none has future support.
  An absent fit class cannot be called a learned rare class.
- Report the number of occupied classes for every role, full-source eligibility
  and bounded working subset support, and all removed cross-role predictor hashes.
- Use ten equal-width confidence bins for any ECE endpoint, matching the
  corrected calibration analysis. Calibration is secondary and conditional on
  labels represented during model fitting; disclose this conditioning.
- The three model seeds repeat fitting on one chronological allocation. They
  are not three independent periods or three independent corpora. Means and
  sample SD describe fitted-model variation, not uncertainty across deployments.
- Paired row intervals, if reported, are class-stratified conditional bootstrap
  intervals over the bounded fixed future test, not temporal drift uncertainty.
- No zero-shot or concept-drift adaptation claim follows from a static future
  window with no adaptation. Both positive and negative comparator effects
  enter the manuscript and response.
- The family assignment is researcher-supplied. NF-ToN-IoT is a documented
  testbed source and must not be relabeled as newly collected production traffic.

All new computation is on physical D. Raw official ZIP remains unchanged on F.
