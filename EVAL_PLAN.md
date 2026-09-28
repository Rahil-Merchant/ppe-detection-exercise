# EVAL_PLAN

## What I'm measuring

The product decision is an alert, so my unit of evaluation is a violation instance, i.e. a
ground-truth bare head in the test split. I kept person-level tracking out of scope because the
person class is about 98% unlabeled (details in LABEL_POLICY).

I report two numbers. Missed-violation rate is the fraction of GT bare heads that got no matching
alert, with matching at IoU 0.3 — heads are 15-30px here, and the alert only needs to point the
officer at the right spot, so IoU 0.5 would be needlessly strict. False-alarm rate is false
alerts per 100 images, which I extrapolate to alerts/hour at 12 streams sampled at 1 frame per
2s in the deployment note.

Both arms produce a per-box violation score (head confidence for the direct arm, detector
confidence times P(bare) for the cascade). I sweep the threshold and read off operating points.

## Operating point

Since false positives are stated to be costlier than misses, I pick the threshold that minimizes
miss rate subject to FA <= 1 per 100 images (roughly 2.2 false alerts per hour per officer at 12
streams), and I also report FA <= 0.5 and <= 2 for sensitivity.

## Trusted test labels

The raw test labels cannot rank models: there are head/helmet flips, and unlabeled heads turn
into fake false positives, which is exactly the metric the client cares about. My protocol:

1. Every test crop flagged by confident learning (DINOv2 embeddings + cleanlab) gets reviewed
   visually, and confirmed flips are corrected.
2. For missing boxes, I take the union of both arms' unmatched detections above 0.5 confidence,
   review each one, and add the real heads and helmets. Using the union keeps the process
   symmetric, so neither arm's blind spots define the ground truth. I had planned a COCO person
   detector as a third proposer but dropped it, since person boxes don't localize heads; the
   caveat is that a blind spot shared by both arms would still be missed.
3. The trusted set is frozen before I read any comparison numbers.
4. I also score both arms against the raw labels and report both regimes side by side. If the
   ranking flips between them, that's a result in itself.

## Hypothesis

H1: at the FA <= 1/100 operating point, the cascade beats direct detection, because it separates
localization (trainable on merged labels that are immune to flips) from classification
(trainable on cleaned crops).

This is falsified if direct's miss rate is lower than, or within the 95% bootstrap CI of, the
cascade's at that operating point. As a secondary check I break results down by head height
(<16, 16-32, 32-96, >96 px); H1 predicts the cascade's advantage concentrates below 32px.

Controls: both arms train on the same images with the same detector family, resolution and
schedule, the same label cleaning, and are scored on the same trusted test set. Seeds and configs
are logged in `runs/`.
