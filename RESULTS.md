# RESULTS

## Headline

Ship the direct detector. My hypothesis was that the cascade would win, and it lost everywhere.
On the trusted test set (682 images, 755 violations) at the FA <= 1/100 operating point, direct
misses 4.4% of violations [CI 3.0-6.0] against the cascade's 13.9% [11.0-17.1]. The CIs don't
overlap, so H1 from EVAL_PLAN is falsified. Direct also runs at 16.9ms/img in fp16 versus ~60ms
for the cascade.

## Numbers (miss rate at each FA budget, trusted labels)

| FA/100 img | Direct | Cascade | DEIMv2-S ref |
| ---------- | ------ | ------- | ------------ |
| <=0.5      | 5.0%   | 24.4%   | 12.9%        |
| <=1.0      | 4.4%   | 13.9%   | 8.5%         |
| <=2.0      | 3.3%   | 9.8%    | 6.9%         |

Per scale at FA<=2, direct vs cascade miss: medium 3.4% vs 11.8%, small (16-32px) 6.3% vs 22.8%,
tiny 4/6 for both. H1 predicted the cascade would win at small scales; it's actually 3.6x worse
there. Stage-1 recall wasn't the problem (AP50 0.950) — the crop classifier was. DEIMv2-S, from
an entirely separate codebase, also loses to direct at every budget, so the conclusion isn't an
RF-DETR artifact.

## The labels decided the ranking

On the raw labels the ranking flips and the cascade looks better (29.8% vs 41.9% at FA<=1). The
reason is test under-annotation: of the top 13 highest-scoring "false positives", 9 were real
bare heads missing from GT. The better detector finds more of them, so noisy labels punish the
better model harder. After adjudication (111 union-of-arms proposals reviewed: 41 heads, 54
helmets, 16 junk) the picture inverts. Comparing models on this dataset without label repair
selects the wrong model.

## Failure modes at the operating point

The direct arm has 4 false positives in total, all sitting on real heads: three are
helmet-wearers alerted on the face under the brim, one is a motion-blurred pedestrian. Its
misses concentrate in the tiny bucket (4 of 6) and dense-crowd occlusion.

The cascade fails in stage 2. Either a bare head next to helmeted coworkers gets classified as
helmet (a missed violation), or a box on the face under a worn helmet gets classified as bare (a
false alarm). Both are context bleed from classifying a padded crop in isolation — the direct
model sees the whole scene and doesn't make the first error.

## Extra experiments (same eval)

- Tightening the crop padding from 0.5 to 0.25 lifts the classifier from 97.96% to 98.31% and
  cuts cascade miss from 13.9% to 9.8% at FA<=1. The diagnosis was right and the fix is real,
  but it still loses to direct at 3.5x the latency.
- Pose gating (YOLO11x-pose, untrained on this data, keeping only alerts with head-keypoint or
  person-box support): at identical FA, miss worsens from 4.4% to 7.8-11.1%. The gate removed
  zero real false positives — they all sit on actual heads, which pose confirms — and only
  suppressed true back-of-head and occluded alerts.
- Fully fine-tuning the classifier instead of the frozen probe drops it to 96.0% vs 98.3%; 18k
  crops aren't enough to beat frozen DINOv2 features.
- Raising resolution from 728 to 896 at a matched GPU budget jumps AP50 from 0.943 to 0.983, yet
  miss at FA<=1 worsens to 5.3% (only 4 epochs fit the budget). The mAP gain didn't transfer to
  the operating point.

## Where the models got it wrong, and how I caught it

1. The label-noise flagger was wrong 71% of the time. cleanlab flagged 157 crops; my review
   found 41 real flips and 112 caps/hoods correctly labeled `head`. I caught this by refusing to
   auto-apply anything unreviewed — blind trust would have added 112 errors to fix 41.
2. My own eval script counted phantom false positives: predictions on the 24 leak-excluded
   images were still being counted, and 9 of "13" FPs at one operating point were phantoms. A
   separately-written bootstrap script disagreed with the point estimate, which is how I caught
   it. Fixed and re-ran everything.
3. Early stopping was watching mAP@50:95, which kept creeping on box tightness while AP50 — what
   an alert actually cares about — had plateaued 10 epochs earlier. I caught it in the per-epoch
   CSV and stopped manually, saving about 5 GPU-hours.
4. The classifier's 97.96% validation accuracy hid its real behavior: its errors cluster exactly
   on the cases that decide the business metric (helmet-adjacent heads), not uniformly. I only
   saw this by reviewing all 107 validation errors instead of trusting the average.
