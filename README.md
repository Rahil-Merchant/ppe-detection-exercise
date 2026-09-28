# PPE Compliance — Hard-Hat Violation Detection

Take-home exercise. Detect people not wearing hard hats from CCTV; alerts to a human safety officer. Dataset: [Hard Hat Workers v10](https://public.roboflow.com/object-detection/hard-hat-workers) (raw images, 70/20/10).

## Deliverable docs

- [EVAL_PLAN.md](EVAL_PLAN.md) — protocol + falsifiable hypothesis
- [LABEL_POLICY.md](LABEL_POLICY.md) — annotation spec + audit numbers
- [DECISIONS.md](DECISIONS.md) — key decisions, alternatives, what would change my mind
- [RESULTS.md](RESULTS.md) — comparison + error analysis
- [DEPLOYMENT.md](DEPLOYMENT.md) — deployment note (section E)

## Repo map

```
src/audit/      basic_stats, find_dupes, missing_annotations, crop_confusion (label audit)
src/data/       make_variants.py  (raw v10 + fixes -> arm1 / arm2_det COCO variants)
src/train/      train_rfdetr.py   (both arms' detectors, RF-DETR DINOv2-ViT)
src/cascade/    train_classifier.py (bare-head vs helmet crop classifier, DINOv2 ViT-S)
src/eval/       predict.py, violation_metrics.py, bench_latency.py
configs/        run configs
runs/           logs, metrics, audit outputs (checkpoints excluded from git)
```

## Reproduce

```bash
# 0. data: unzip Roboflow v10 (COCO format) to data/raw/{train,valid,test}
python src/audit/basic_stats.py data/raw
python src/audit/find_dupes.py data/raw
python src/audit/missing_annotations.py data/raw          # GPU
python src/audit/crop_confusion.py data/raw               # GPU
# 1. build variants (with label fixes produced by audit review)
python src/data/make_variants.py data/raw data/variants --fixes runs/audit/fixes.json
# 2. train
python src/train/train_rfdetr.py data/variants/arm1     runs/arm1     --model small --res 728
python src/train/train_rfdetr.py data/variants/arm2_det runs/arm2_det --model small --res 728
python src/cascade/train_classifier.py data/variants/arm1 runs/arm2_cls
# 3. predict + evaluate
python src/eval/predict.py data/variants/arm1/test --arm direct  --det runs/arm1/checkpoint_best_ema.pth --target-id 0 --res 736 --out runs/arm1/preds_test.json
python src/eval/predict.py data/variants/arm1/test --arm cascade --det runs/arm2_det/checkpoint_best_ema.pth --cls runs/arm2_cls_pad025/best.pth --target-id 0 --res 736 --pad 0.25 --out runs/arm2_det/preds_test_pad025.json
python src/eval/violation_metrics.py data/variants/arm1/test/_annotations.coco.json --preds runs/arm1/preds_test.json --name direct_trusted --exclude runs/audit/test_leaky_images.json
python src/eval/violation_metrics.py data/variants/arm1/test/_annotations.coco.json --preds runs/arm2_det/preds_test_pad025.json --name cascade_trusted --exclude runs/audit/test_leaky_images.json
python src/eval/bench_latency.py --det runs/arm1/checkpoint_best_ema.pth --res 736 --cls runs/arm2_cls_pad025/best.pth
# DEIMv2 reference (run inside deimv2 repo): see src/eval/deim_predict.py header
```
