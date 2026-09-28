# LABEL_POLICY

## What the audit found (v10, 7035 images)

| Issue | Method | Result |
|---|---|---|
| person under-annotation | COCO detector (conf>=0.7) vs GT at IoU>=0.3 | 97-98% of confident person detections have no GT box. Train has 16,132 predicted vs 450 labeled; 96% of images contain at least one unlabeled person |
| head/helmet flips | DINOv2 embeddings + cleanlab; I reviewed all 157 flags on zoomed panels | 41 real flips and 4 bogus boxes. The other 112 flags were wrong — mostly caps and hoods that embed like helmets. head gets flagged 3-5x more than helmet |
| cross-split leakage | md5 + pHash (hamming <= 4) | 0 exact duplicates, 174 near-duplicate pairs, 79 of them cross-split (~1% of test) |
| scale | box-height histogram | 40% of heads and 35% of helmets are 32px or shorter. Images are ~500x333, not the 1080p the brief assumes |
| class balance | counts | helmet 19,747 / head 6,677 / person 615 |

## Policy

I dropped the person class everywhere — with 98% of instances missing, training on it teaches
the model to suppress people. The classes I kept are `head` (a visible human head not wearing a
hard hat: bare, cap, hood, turban, or helmet carried in hand) and `helmet` (a hard hat actually
worn). The violation class is `head`.

For training labels I applied only the confirmed corrections from the review above (41 flips, 4
deletions), identically to both arms. Missing heads in train stay missing: pseudo-labeling
thousands of boxes would import the proposer's bias into both arms, and since both arms see the
same data the comparison stays fair either way.

Test labels get the full treatment because they decide the comparison: flips corrected and
missing boxes added through the reviewed union-of-arms process described in EVAL_PLAN.

## Hard cases

- A tilted or perched helmet counts as `helmet` if it covers the crown, otherwise `head`.
- Hoods, caps, turbans and hair are `head` — they aren't PPE. This was the main flip source.
- A helmet in hand, on the ground or hanging is not a helmet instance; the carrier's head is `head`.
- The back of a head is still labeled, since compliance is visible from behind.
- Occluded heads are labeled if roughly 30%+ is visible and the protection status is decidable.
  If it's genuinely undecidable at any zoom, I leave it unlabeled rather than guess violations
  into the eval set.
- Heads under ~10px are labeled only when decidable — an alert nobody can verify is effectively
  a false alarm.
- In crowds, every decidable head is labeled individually; no group boxes.
