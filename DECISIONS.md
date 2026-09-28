# DECISIONS

**1. RF-DETR-Small for both arms.** The brief requires a vision-transformer backbone, which
narrows the field considerably. RF-DETR fits well: its backbone is a DINOv2 ViT, it ships with a
COCO checkpoint so fine-tuning on 5k images converges in hours, and it's built for real-time
inference, which matters at 12 streams on one GPU. I went with the Small variant at 728px, which
keeps 15-30px heads above the effective stride. I also trained DEIMv2-S (DINOv3 ViT backbone,
completely separate codebase) as a reference so the conclusion isn't an RF-DETR artifact.
I'd have swapped detectors if DEIMv2 clearly won at similar latency — it didn't (8.5% vs 4.4%
miss at a budget of one false alarm per 100 images). With two more weeks: a proper
resolution sweep and distillation for edge hardware.

**2. Dropping the person class.** The audit found 16,132 detected persons vs 450 labeled in
train, so about 98% missing. Training on it anyway teaches the model to suppress people, and
pseudo-labeling imports the proposer's bias into both arms, so I dropped it. I'd revisit if the
client needs per-person deduplicated alerts. With two weeks: re-annotate persons or bolt on an
off-the-shelf tracker for dedup.

**3. Violation unit = bare-head instance.** The officer acts on "there's an unprotected head
here". Image-level scoring hides multi-violation scenes, and person-level needs the class I just
dropped.

**4. Cleaning training labels with confirmed fixes only.** cleanlab flagged 157 of 26,424 crops
and I reviewed all of them: 41 real flips, 4 bogus boxes, 112 correct as labeled. Auto-applying
the flags would have corrupted 112 good labels to fix 41, so only the reviewed fixes went in,
identically for both arms. I'd invest in a full train relabel if error analysis showed misses
clustering on label-noise patterns.

**5. Operating point = false-alarm budget, not max-F1.** FP is stated to be costlier than FN -
officer trust dies with false alarms, and a missed violation usually recurs on the next frame.
So I minimize miss subject to FA <= 1 per 100 images, with 0.5 and 2 reported for sensitivity.
A real cost ratio from the client would move me along the curve. With two weeks: set the budget
from a measured officer-tolerance study plus temporal aggregation.

**6. Matching alerts at IoU 0.3.** A 5px localization error destroys IoU 0.5 on a 20px head but
doesn't hurt the alert's usefulness. 0.3 is loose enough for tiny boxes and tight enough to
forbid "alert somewhere near".

**7. Cascade design.** For stage 1 I considered a person detector (labels unusable), a head-only
detector (can't see helmeted heads) and a merged head+helmet region detector, which is immune to
label flips since flips don't move merged boxes. I picked merged. Stage 2 is a DINOv2-S linear
probe - with 18k crops a full fine-tune overfits, which I later confirmed (96.0% vs 98.3%). The
violation score is detector confidence times P(bare). My worry was stage-1 recall on tiny heads
becoming the bottleneck; the per-scale eval shows it wasn't — stage 2 was.

**8. Test hygiene.** 24 test images near-duplicate train/valid images (out of 79 cross-split
pairs), so I excluded them. Test labels got flips plus reviewed additions, because false
positives in the eval must be real ones when FA is the business metric; I report raw-label
results alongside to show the distortion.
