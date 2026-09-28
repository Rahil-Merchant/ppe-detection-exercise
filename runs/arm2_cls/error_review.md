# Classifier val-error review (2 of 7 panels, 32 errors inspected)

107/5251 val crops misclassified (97.96% acc). Visual review of 32:
- ~85-90% genuine model errors, dominant modes:
  1. Bare head adjacent to helmeted colleagues -> predicted helmet (context bleed
     from 0.5 crop padding). Cascade consequence: missed violation (FN).
  2. Helmet worn but GT box centered on face below it -> predicted head.
     Cascade consequence: false alarm (FP) - the expensive direction.
- ~10-15% are label errors in valid (cloth caps labeled helmet, bare heads labeled helmet).

Queued experiment: retrain probe with PAD=0.25 (tighter crops) vs PAD=0.5; compare
cascade operating point. Padding jitter (0.15) already in train aug.
