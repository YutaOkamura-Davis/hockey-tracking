# Benchmark Results

## Detection

| Model | Dataset | Split | Threshold | Recall | Precision | F1 | Notes |
|-------|---------|-------|-----------|--------|-----------|----|-------|
| RF-DETR-Base (COCO) | VIP-HTD | train | 0.5 | — | — | — | Baseline |
| RF-DETR-Base (hockey FT) | VIP-HTD | train | 0.5 | — | — | — | If fine-tuned |

## Tracking (v0 Baseline — Week 2)

| Tracker | Detector | Space | HOTA | IDF1 | MOTA | AssA | DetA | IDsw | Frag | Notes |
|---------|----------|-------|------|------|------|------|------|------|------|-------|
| ByteTrack | RF-DETR | pixel | — | — | — | — | — | — | — | v0 baseline |

## Tracking (v1 — Phase 1)

| Tracker | Detector | Space | Homography | Re-ID | HOTA | IDF1 | MOTA | AssA | DetA | IDsw | Frag |
|---------|----------|-------|------------|-------|------|------|------|------|------|------|------|
| ByteTrack | RF-DETR | rink | yes | no | — | — | — | — | — | — | — |
| ByteTrack | RF-DETR | rink | yes | OSNet | — | — | — | — | — | — | — |
| ByteTrack | RF-DETR-Seg | rink | yes (mask foot) | OSNet | — | — | — | — | — | — | — |

## Ablations (Phase 1–3)

| # | Ablation | Metric | Baseline | With Change | Delta | Phase |
|---|----------|--------|----------|-------------|-------|-------|
| 1 | RF-DETR vs RF-DETR-Seg | IDF1 | — | — | — | 1 |
| 2 | Box center vs mask foot | Reproj err | — | — | — | 1 |
| 3 | Pixel vs hybrid rink-space | IDF1/IDsw | — | — | — | 1 |
| 4 | No re-ID vs OSNet | IDF1/IDsw | — | — | — | 1 |
| 5 | No jersey vs + jersey | Player-ID | — | — | — | 2 |
| 6 | Box crop vs mask crop | Jersey acc | — | — | — | 2 |
| 7 | PARSeq vs PARSeq+VLM | Jersey acc | — | — | — | 3 |
| 8 | No legibility gate vs gated | Jersey acc | — | — | — | 2 |
| 9 | No linking vs pairwise | IDF1/Frag | — | — | — | 3 |
| 10 | No roster vs roster | Player-ID | — | — | — | 3 |
| 11 | No shift vs shift | Player-ID | — | — | — | 3 |
| 12 | RF-DETR-Seg vs SAM2.1 | Reproj/Jersey/Team | — | — | — | 3 |

## Identity (Phase 2–3)

| Config | Player-ID Acc | Top-3 Acc | Coverage | Abstention | Confidence (avg) |
|--------|--------------|-----------|----------|------------|------------------|
| Track ID only | — | — | — | — | — |
| + team + jersey | — | — | — | — | — |
| + roster constraint | — | — | — | — | — |
| + shift constraint | — | — | — | — | — |
| + pairwise linking | — | — | — | — | — |
