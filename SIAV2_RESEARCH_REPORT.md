# YOLOv7/SIAV2 품질·성능 개선 조사 보고서

- 조사일: 2026-08-12
- 대상: `yolov7-main`의 SIAV2 16-class, 1280 입력, W6 teacher 및 P3/P4/P5/P6 경량 student 후보
- 결론: 현재 구조는 **속도 최적화와 response distillation의 기반은 갖췄지만**, 실제 품질을 결정하는 데이터 검증·재현성·작은 객체 recall·배포 후 score/NMS 최적화가 아직 주요 미확인 영역이다. 논문 모듈을 무작정 추가하기보다 아래 순서로 ablation을 수행하는 것이 안전하다.

## 1. 현재 저장소에서 확인된 상태

### 이미 반영된 강점
- YOLOv7의 E-ELAN/auxiliary head/OTA 계열 학습을 유지한다.
- `train_aux.py`에 EMA, AMP, gradient accumulation, freeze, class/image weighting, auto-anchor, close-mosaic가 있다.
- `utils/loss.py`에는 CIoU, focal loss, label smoothing, small-object box/class weighting, OTA 계열 동적 positive assignment가 있다.
- W6 teacher와 P4/P5/P6, P3-lite 후보 및 같은 stride/cross-stride response distillation workflow가 있다.
- `SIAV2_TRAINING_READY.md` 기준으로 크기별 AP(small/medium/large), EDA, TensorRT 8.6.1/10.14.x latency guardrail이 있다.

### 품질 관점에서 남은 공백
1. **데이터 gate가 아직 최우선**: `data/siav2.yaml`과 실제 dataset이 이 작업공간에 없으므로 class별 AP, 누락 라벨, 중복/유사 이미지, train-val leakage, class imbalance를 아직 판단할 수 없다.
2. **P3 제거의 정보 손실**: `P4/P5/P6`는 stride-8 출력을 inference에서 제거한다. 문서가 명시하듯 `8:16` cross-stride distillation은 학습 보조일 뿐 P3의 공간 해상도를 대체하지 않는다. 작은 객체 recall이 핵심이면 P3-lite가 우선 후보이다.
3. **feature distillation 미구현**: 현재는 response/cross-response distillation 중심이다. teacher feature를 student neck/head에 전달하는 feature distillation은 아직 실험 대상이다.
4. **배포 파라미터 미최적화**: 학습 mAP와 별개로 confidence threshold, NMS IoU, top-k, class-agnostic 여부, FP16/INT8 calibration을 model/dataset별로 튜닝해야 한다. 현재 `test.py` 기본 NMS IoU는 0.65이고 `export.py` 기본값은 0.45라서 동일 조건이 아니면 비교가 왜곡될 수 있다.
5. **평가 설계 보강 필요**: mAP50-95만으로는 작은 객체/실제 운영 오류를 설명하기 어렵다. class AP, recall, precision-recall curve, size/occlusion/crowding bucket, false-positive taxonomy, seed별 평균·표준편차를 필수로 기록해야 한다.
6. **훈련 recipe의 검증 부족**: close-mosaic 시점, `paste_in`, mixup, scale, label smoothing, focal gamma, small-object gain은 존재하지만 조합별 근거/ablation 결과가 없다. 특히 copy-paste는 segmentation mask가 없으면 적용할 수 없다.

## 2. 관련 논문과 적용 가능성

| 우선순위 | 논문 | 핵심 | SIAV2 적용 판단 |
|---|---|---|---|
| A | Wang et al., **YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors**, arXiv:2207.02696 | E-ELAN, planned re-parameterization, coarse-to-fine lead-guided label assignment, auxiliary head | 현재 baseline의 기준. 구조를 더 바꾸기 전에 동일 recipe/weights로 재현성 확보 |
| A | Ge et al., **OTA: Optimal Transport Assignment for Object Detection**, arXiv:2103.14259 | dynamic label assignment | 기존 OTA/SimOTA와 assignment 및 positive 수를 비교하는 ablation 가치가 큼 |
| A | Wang et al., **YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information**, arXiv:2402.13616 | PGI, GELAN | P3 제거 student의 학습 안정성/gradient 정보 손실 보완 후보. 단, YOLOv7 head와 직접 병합하지 말고 별도 baseline으로 비교 |
| A | Wang et al., **YOLOv10: Real-Time End-to-End Object Detection**, arXiv:2405.14458 | NMS-free consistent dual assignments, efficiency design | TensorRT latency가 핵심이면 end-to-end/NMS 비용 비교 후보. 기존 YOLOv7에 일부만 이식하면 assignment 불일치 위험 |
| A | Zhao et al., **DEIM: DETR with Improved Matching for Fast Convergence**, arXiv:2412.04234 | matching/denoising 개선 | YOLO 교체보다, 최신 detector가 동일 latency에서 앞서는지 비교하는 external baseline으로 사용 |
| B | Zhao et al., **DETRs Beat YOLOs on Real-time Object Detection**, arXiv:2304.08069 | RT-DETR의 real-time transformer detector | P6 제거/작은 객체 trade-off가 나쁘면 후보. 반드시 동일 TensorRT, 입력, batch, NMS 포함 조건으로 비교 |
| A | Akyon et al., **Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection**, arXiv:2202.06934 | SAHI slicing inference | AP_small이 병목일 때 가장 직접적인 방법. 전체 이미지 latency가 아니라 slice 수별 latency와 recall을 별도 보고 |
| A | Hinton et al., **Distilling the Knowledge in a Neural Network**, arXiv:1503.02531 | logit distillation | 현재 response distillation의 temperature/box/object/class weight를 체계적으로 sweep할 근거 |
| B | Wang et al., **Generalized Focal Loss**, arXiv:2006.04388 | quality-aware localization/classification | confidence가 IoU 품질을 잘 반영하지 못하는 경우 후보. loss 교체 후 NMS/threshold 재튜닝 필수 |
| B | Tong et al., **Focaler-IoU Loss**, arXiv:2401.10525 | IoU error 구간에 집중 | 작은/가림 객체의 localization을 개선할 수 있으나 CIoU와 단독 ablation 필요 |
| B | Tong et al., **Wise-IoU**, arXiv:2301.10051 | dynamic focusing IoU loss | 작은 객체 box regression 후보. mAP 전체보다 AP_small와 localization error로 판단 |
| A | Li et al., **Quantizing YOLOv7: A Comprehensive Study**, arXiv:2407.04943 | YOLOv7 PTQ/QAT 정량화 분석 | FP16 이후 INT8을 할 경우 직접 참고. calibration set 대표성, per-channel, QAT를 분리 검증 |
| B | Ghiasi et al., **Simple Copy-Paste**, arXiv:2012.07177 | instance copy-paste augmentation | mask가 있을 때만 유효. bbox-only SIAV2에는 현재 `paste_in`과 혼동하면 안 됨 |
| B | Bochkovskiy et al., **YOLOv4**, arXiv:2004.10934 | Mosaic 등 bag-of-freebies | 현재 mosaic/mixup recipe의 출발점. close-mosaic 시점은 데이터별 ablation 필요 |

## 3. 권장 실험 순서

### Gate 0 — 측정 신뢰성 (먼저 고정)
- dataset EDA: invalid/duplicate/near-duplicate, empty image, class별 box 수, box 크기, aspect ratio, crowding/occlusion, train-val leakage.
- 동일 seed 3개 이상으로 W6 teacher와 각 student를 재학습하고 `mAP50`, `mAP50-95`, class AP, AP_small/medium/large, recall, parameter/FLOPs를 저장.
- TensorRT는 동일 버전·opset·NMS 포함 여부·batch·warmup·input shape를 고정. PyTorch inference와 TRT output 차이도 검증.
- `test.py`와 export의 NMS/conf 설정을 하나의 실험 manifest에서 공유하도록 한다.

### Gate 1 — 가장 기대값이 높은 품질 개선
1. W6 teacher: 300 epoch 또는 현재 recipe를 3 seed로 고정.
2. student 3종: P4/P5/P6, P3-lite/P4/P5, P3-lite/P4/P5/P6를 **distillation 없음/response distill/cross-response distill** 3개씩 비교.
3. distillation sweep: temperature `{2,4,8}`, cross weight `{0,0.25,0.5}`, small-object gain `{1,2}`. 전체 mAP가 아니라 AP_small 및 recall을 1차 기준으로 한다.
4. close-mosaic `{10,20,30}` epoch, `paste_in {0,0.15,0.3}`, mixup `{0,0.15}`를 one-factor 또는 fractional factorial로 검증.

### Gate 2 — 작은 객체 전용
- P3-lite를 기본 후보로 유지하고, 입력 1280에서 stride-8 feature의 channel width만 줄이는 구조를 먼저 비교한다.
- SAHI를 training 없이 inference-only로 시험한다: slice size/overlap별 AP_small, 전체 latency, duplicate merge 오류를 기록.
- GT box 크기 기준이 letterbox 후 픽셀인지 원본 픽셀인지 통일한다. 현재 문서의 64/128 기준은 반드시 이 정의를 표에 명시한다.
- 필요할 때만 tiling fine-tuning 또는 feature distillation을 추가한다.

### Gate 3 — loss/assignment/배포
- CIoU vs Wise-IoU vs Focaler-IoU를 같은 seed/recipe로 비교.
- 기존 OTA/SimOTA와 task-aligned assignment를 positive 수, AP_small, 학습 안정성으로 비교.
- confidence/NMS grid search를 validation에서 수행하되 test set은 잠근다.
- FP16 이후 INT8 PTQ → representative calibration → QAT 순서로 진행. accuracy drop이 운영 SLA 이내인지 확인한다.

## 4. 모델 선택 의사결정

- **작은 객체가 핵심이고 latency 여유가 있다**: P3-lite/P4/P5 또는 P3-lite/P4/P5/P6를 우선.
- **최저 latency가 핵심**: P4/P5/P6를 선택하되 AP_small 하락과 recall을 명시적으로 승인해야 함. cross-stride distillation만으로 보상되었다고 가정하지 않음.
- **같은 latency에서 mAP가 목표**: YOLOv7 개조보다 YOLOv10, RT-DETR/DEIM을 external baseline으로 먼저 측정.
- **배포가 INT8 중심**: 구조 변경보다 calibration/QAT와 score/NMS 최적화가 먼저.
- **운영 장면이 단일/고정 카메라**: 일반 COCO 논문의 모듈보다 hard-negative mining, camera-specific augmentation, threshold calibration이 더 높은 기대값을 가질 가능성이 큼.

## 5. 최종 판단

현재 놓친 가장 큰 부분은 새로운 attention 모듈이 아니라 **(1) 데이터/라벨 품질을 수치화하지 않은 상태에서의 구조 선택, (2) P3 제거에 따른 AP_small·recall 손실 검증, (3) 동일 배포 조건의 NMS/INT8/latency 비교, (4) seed·ablation 재현성**이다. 위 Gate 0~2 결과 없이 YOLOv9/v10/v12의 블록을 YOLOv7에 섞는 것은 성능 향상 원인을 분리할 수 없고, 구현 리스크만 키운다.

## 링크

- YOLOv7: https://arxiv.org/abs/2207.02696
- OTA: https://arxiv.org/abs/2103.14259
- YOLOv9: https://arxiv.org/abs/2402.13616
- YOLOv10: https://arxiv.org/abs/2405.14458
- RT-DETR: https://arxiv.org/abs/2304.08069
- DEIM: https://arxiv.org/abs/2412.04234
- SAHI: https://arxiv.org/abs/2202.06934
- Knowledge Distillation: https://arxiv.org/abs/1503.02531
- Generalized Focal Loss: https://arxiv.org/abs/2006.04388
- Wise-IoU: https://arxiv.org/abs/2301.10051
- Focaler-IoU: https://arxiv.org/abs/2401.10525
- Quantizing YOLOv7: https://arxiv.org/abs/2407.04943
- Simple Copy-Paste: https://arxiv.org/abs/2012.07177
- YOLOv4: https://arxiv.org/abs/2004.10934
