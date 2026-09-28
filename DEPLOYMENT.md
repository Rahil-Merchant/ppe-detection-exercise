# Deployment note

I would ship Arm 1 (direct RF-DETR-S) in fp16 at threshold ~0.45, which is the FA<=1/100
operating point. It runs at 16.9ms per frame, about
59fps. Twelve streams sampled at one frame every 2s need 6fps, so there's roughly 10x headroom;
a T4-class edge card at half the throughput still holds ~5x, and batching the streams buys more.
The cascade takes ~60ms (the DINOv2 classifier at 518px input is the expensive stage) and misses
three times as much, so there's no case for it. Alerts should be aggregated per track before an
officer sees them — something like 3-of-5 frames over 10 seconds — because even 1 FP per 100
frames is ~216 raw false events per hour across 12 streams, and persistence collapses
single-frame errors to a handful without meaningfully delaying a real violation.

Before I'd put this in front of a safety officer I'd want four things. First, evaluation on the
client's actual cameras: this dataset is ~500x333 daytime stills, not 1080p CCTV, so night/IR,
motion blur, compression and true 15-30px heads are all unrepresented — and the tiny bucket is
where the model already misses 4 of 6. Second, a small tracked pilot with per-person dedup;
pose or tracking also filters posters, reflections and other non-person surfaces this dataset
doesn't have. Third, an agreed FA budget, since the trade is steep (5.0% down to 3.3% miss
between FA 0.5 and 2 per 100). Fourth, a labeled sample of the client's own footage adjudicated
the same way as here, because this exercise showed the model ranking flips with label quality.
