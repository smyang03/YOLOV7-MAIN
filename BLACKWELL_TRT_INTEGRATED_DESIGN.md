# YOLOv7 개선 통합 설계 및 Blackwell TensorRT 평가 계획

작성일: 2026-08-13  
대상 환경: Blackwell GPU, CUDA 13.0, TensorRT 10.16 이후  
대상 모델: YOLOv7-L 640/1280, W6 1280, S10/SIAV2 계열, P3-lite, anchor-free 후보  
제외: Knowledge Distillation

---

## 1. 최종 목표

최종 목표는 단일 최고 mAP 모델을 고르는 것이 아니다.

```text
실제 미감지 recall 개선
×
YOLOv7-L/S10의 운영 속도 유지
×
Blackwell + CUDA 13.0 + TensorRT 10.16에서 재현 가능한 배포
```

따라서 모든 후보는 다음 세 가지를 동시에 평가한다.

1. 품질: precision, recall, mAP50, mAP50-95, 작은 객체 recall, 실제 miss recovery
2. 속도: 엔진 build 성공 여부, TensorRT GPU latency, end-to-end latency, 평균/P95
3. 운영성: 메모리, NMS 처리, dynamic/static shape, FP16/FP8 지원 여부, 실패 시 fallback 비용

---

## 2. 설계 공간 분리

### A. 학습이 필요하지 않은 설계

모델 weight를 재학습하지 않고 기존 YOLOv7-L/S10을 이용한다.

#### A-1. 후처리 조정

- class별 confidence threshold
- class별 NMS
- NMS IoU sweep
- Soft-NMS/DIoU-NMS 비교
- confidence calibration

기대 효과: 후보가 이미 있지만 score 또는 NMS 때문에 탈락하는 경우  
한계: detector가 후보를 전혀 만들지 못한 FN은 복구하지 못함

#### A-2. 정적 cascade

```text
1차: YOLOv7-L/S10 640
→ 조건을 만족하는 프레임만 960/1280 재추론
→ box 좌표 복원 및 class별 merge
```

초기 trigger:

- person은 검출됐지만 head/helmet이 없음
- person 수 대비 head 수가 비정상적으로 낮음
- 작은 person box가 다수
- person 상단 ROI에 head 후보가 없음
- 낮은 score margin
- 연속 프레임에서 detection이 불안정

person 자체를 놓친 경우에는 person ROI cascade가 복구하지 못하므로 tile/full-frame fallback을 함께 둔다.

#### A-3. 영상 temporal 보정

- 짧은 프레임 window의 box association
- 이전/다음 프레임의 detection 유지
- 일시적인 low-score miss 보정
- 일정 시간 이상 미검출이면 stale box 제거

#### A-4. feature 기반 risk monitor

P3/P4 feature cosine을 새 box 생성기로 사용하지 않는다. 대신 “이 프레임을 다시 볼 가치가 있는가”를 판단하는 risk signal로 사용한다.

```text
YOLO 내부 feature + detection metadata
→ risk score
→ 선택적 고해상도 재추론
```

---

### B. 학습이 필요한 설계

#### B-1. P3-lite

작은 객체 공간 정보를 보존하면서 P3 채널과 head 계산을 줄인다.

```text
P3: 유지 또는 경량화
P4/P5/P6: baseline 유지
P4 semantic → P3 보강
```

첫 번째 구조 변경 후보로 둔다. 현재 640→1280에서 일부 miss가 회복된 결과와 가장 직접적으로 연결된다.

#### B-2. P2 비교군

P2를 추가해 작은 객체 recall의 상한을 확인한다. 최종 운영 모델로 바로 채택하지 않고, P3 해상도가 근본 병목인지 확인하기 위한 실험으로 둔다.

#### B-3. Anchor-free

anchor만 YAML에서 제거하는 방식은 불충분하다. 다음 요소를 모두 구현·검증해야 한다.

- anchor-free detection head
- center/point positive assignment
- anchor-free box decoding
- anchor-free loss
- crowded object assignment 충돌 처리

실험 순서:

```text
anchor-based baseline
→ anchor-free baseline
→ P3-lite anchor-based
→ P3-lite anchor-free
```

#### B-4. IoU-aware score

VarifocalNet/GFL 방향을 YOLOv7 head에 최소 변경으로 적용한다.

```text
final_score = objectness × class_score × predicted_iou_quality
```

후보 ranking과 NMS 품질을 개선하는 목적이며, 후보 미생성 FN의 직접 해결책은 아니다.

#### B-5. WIoU/Focaler-IoU

box regression loss만 독립적으로 교체한다.

```text
CIoU → WIoU → Focaler-IoU
```

loss·구조·sampler를 동시에 바꾸지 않는다.

#### B-6. head 중심 sampler와 hard replay

- 작은 head가 있는 이미지 oversampling
- 가림 head image replay
- 640 miss이면서 GT가 있는 image replay
- helmet false positive crop hard negative replay

---

### C. 통합 운영 설계

최종 후보는 다음이다.

```text
1차 detector: YOLOv7-L 또는 S10 640 TRT FP16/FP8
        ↓
저비용 rule/risk monitor
        ↓
위험 프레임/ROI만 960/1280 TRT
        ↓
class별 merge/NMS
        ↓
temporal association
```

#### C-1. 선택 정책

초기에는 규칙 기반으로 시작한다. 이후 miss set으로 FN-risk predictor를 학습한다.

risk 입력:

- P3/P4 pooled feature
- person/head count
- person box 크기
- person-head 거리
- score margin
- 640과 augmentation prediction 차이
- temporal inconsistency

risk predictor는 새 box를 생성하지 않고 2차 추론 trigger만 생성한다.

#### C-2. 평균 latency 모델

```text
평균 비용 = 1차 640 비용 + trigger_rate × 2차 재추론 비용
```

최종 선택 기준은 단일 latency가 아니라 다음이다.

```text
추가 recall / 추가 평균 latency
```

동일 FPS에서 recall이 높은 후보를 우선한다.

---

## 3. 학습 실험 매트릭스

### 3.1 순수 baseline

| ID | 모델 | 입력 | 목적 |
|---|---|---:|---|
| B1 | YOLOv7-L | 640 | 속도 기준 |
| B2 | YOLOv7-L | 1280 | 해상도 품질 상한 |
| B3 | W6 | 1280 | 큰 모델 품질 기준 |
| B4 | S10 | 1280 | 경량 품질 기준 |

### 3.2 구조 ablation

| ID | 변경 | 입력 | 판단 |
|---|---|---:|---|
| A1 | P3-lite | 1280 | 작은 객체·계산량 절충 |
| A2 | P2 | 1280 | 공간 해상도 상한 |
| A3 | anchor-free | 1280 | assignment 효과 |
| A4 | P3-lite + anchor-free | 1280 | 조합 효과 |

### 3.3 학습 목표 ablation

| ID | 변경 |
|---|---|
| L1 | CIoU baseline |
| L2 | WIoU |
| L3 | Focaler-IoU |
| L4 | IoU-aware ranking |
| L5 | head-heavy sampler |
| L6 | hard positive/negative replay |

구조 실험과 loss 실험을 처음부터 결합하지 않는다. 각 단계의 best 후보만 다음 단계에 전달한다.

---

## 4. 품질 평가 기준

### 4.1 표준 지표

- precision
- recall
- mAP50
- mAP50-95
- AP50/AP75
- class별 AP/recall

### 4.2 문제 특화 지표

- small object recall
- occlusion bucket recall
- image-boundary recall
- person detected/head missed
- helmet missed/head detected
- 실제 miss image recovery
- 추가 false positive
- frame-level miss rate
- object-level miss rate

### 4.3 cascade 지표

- trigger rate
- 2차 처리율
- 평균 latency
- P95/P99 latency
- 1차 대비 추가 recall
- 2차 추가 false positive
- 사람 자체 miss와 head-only miss 분리

---

## 5. Blackwell + CUDA 13.0 + TensorRT 10.16 평가 규칙

### 5.1 환경 고정

측정 시작 시 아래 정보를 저장한다.

```bash
nvidia-smi
nvcc --version
trtexec --version
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -c "import tensorrt as trt; print(trt.__version__)"
```

추가 기록:

- GPU 이름과 compute capability
- driver version
- GPU clock/power mode
- TensorRT builder version
- ONNX opset
- input shape
- precision
- workspace
- optimization level
- batch size
- NMS 포함 여부

### 5.2 엔진 종류

각 후보에 대해 가능한 범위에서 다음을 만든다.

1. FP16 decoded engine
2. FP16 raw-head engine
3. FP8 engine 또는 explicit FP8 graph
4. FP4/NVFP4는 명시적 quantization graph가 준비된 경우에만 별도 평가

`trtexec --fp4` 같은 단순 플래그만으로 FP4가 된다고 간주하지 않는다. Blackwell TensorRT 10.16에서 지원되는 explicit quantization 방식과 layer fallback을 확인한 뒤 성공 엔진만 평가한다.

### 5.3 TensorRT build 조건

기본 build 조건:

```text
--fp16
--builderOptimizationLevel=5
--workspace=4096 또는 서버 허용값
--profilingVerbosity=detailed
```

정적 shape와 dynamic shape를 분리한다. 운영 입력이 640/1280으로 고정이면 정적 엔진을 우선한다. dynamic engine은 flexibility와 latency를 별도로 측정한다.

### 5.4 latency 측정

각 엔진마다:

- engine build time
- warmup 200회 이상
- 측정 1000회 또는 30초 이상
- GPU compute latency
- H2D/D2H 포함 latency
- end-to-end preprocessing + inference + NMS
- median, mean, p95, p99

`trtexec` GPU compute latency만으로 운영 FPS를 주장하지 않는다. Python/서비스 wrapper의 preprocessing, memory copy, NMS를 별도 측정한다.

### 5.5 품질 측정

TRT engine의 동일한 input preprocessing과 동일한 NMS 설정으로 validation set을 평가한다.

```text
동일 images/labels
동일 letterbox
동일 confidence
동일 NMS IoU
동일 class mapping
```

PyTorch FP16, ONNXRuntime, TRT FP16/FP8 결과를 비교해 export 과정의 출력 차이도 확인한다.

---

## 6. 예상 장단점

| 설계 | 품질 기대 | 속도 기대 | 장점 | 단점 |
|---|---|---|---|---|
| 640 baseline | 기준 | 최고 | 단순·안정 | 작은 객체 miss |
| 1280 full | 높음 | 낮음 | 구현 쉬움 | 전체 latency 증가 |
| P3-lite | 중상 | 양호 | 구조 내부 개선 | 개선폭 불확실 |
| P2 | 높음 | 낮음~중간 | 극소 객체 강점 | 메모리·후보 증가 |
| anchor-free | 중간 | 양호 | anchor 민감도 감소 | assignment 재설계 필요 |
| WIoU | 중간 | 동일 | inference 비용 없음 | 후보 미생성 해결 못함 |
| IoU-aware score | 중간 | 소폭 비용 | ranking 개선 | head 수정 필요 |
| full DINO/YOLOE | 불확실 | 나쁨 | semantic 후보 가능 | 운영 속도와 충돌 |
| 640 + selective 1280 | 높음 | 평균 양호 | 품질/속도 절충 | trigger 누락·P95 증가 |
| temporal aggregation | 영상에서 높음 | 낮은 추가비용 | 일시적 miss 보완 | stale box 위험 |
| FN-risk cascade | 높음 | trigger율에 좌우 | 필요한 경우만 재추론 | miss monitor 학습 필요 |

---

## 7. 의사결정 게이트

### Gate 1: P3-lite 채택 여부

다음 조건을 모두 본다.

- head/helmet recall 증가
- 평균 latency 증가가 허용범위 이내
- TRT build/export 성공
- false positive 증가가 제한적

### Gate 2: anchor-free 채택 여부

- P3-lite와 동일 조건 비교
- 작은 객체 recall 개선
- crowded scene duplicate/miss 변화
- TensorRT export 성공

### Gate 3: cascade 채택 여부

- 640 대비 miss recovery 증가
- trigger rate가 운영 허용범위 이내
- 평균 latency가 목표 FPS 이내
- P95 latency가 서비스 한계 이내

### Gate 4: FP8 채택 여부

- Blackwell에서 엔진 build 성공
- PyTorch/FP16 대비 recall 하락이 허용범위 이내
- layer fallback이 과도하지 않음
- 실제 end-to-end speedup 확인

### Gate 5: FP4/NVFP4 채택 여부

- 명시적 quantization graph 준비
- unsupported layer fallback 확인
- 품질 손실 측정
- FP8 대비 실제 이득이 존재

---

## 8. 권장 최종 모델 선택

### 단일 모델 배포가 필요할 때

```text
P3-lite 또는 anchor-free 후보 중
TRT FP16/FP8에서 recall-latency Pareto가 좋은 모델
```

### 속도와 품질을 모두 최우선으로 할 때

```text
S10/YOLOv7-L 640 1차
+ FN-risk 또는 규칙 기반 trigger
+ 위험 영역 960/1280 TRT
+ temporal merge
```

### 가장 안정적인 개발 순서

```text
1. 기존 baseline TRT 재측정
2. P3-lite TRT
3. anchor-free TRT
4. WIoU/IoU-aware 학습
5. 640→selective 1280 cascade
6. FN-risk predictor
7. FP8
8. FP4/NVFP4는 별도 feasibility gate
```

---

## 9. 측정 결과 리포트 형식

최종 리포트에는 모델별로 다음 표를 작성한다.

| Model | Input | Precision | Build | mAP50 | mAP50-95 | Precision | Recall | Mean ms | P95 ms | FPS | Mem | 비고 |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | 640 | FP16 | pass/fail | | | | | | | | | |
| baseline | 1280 | FP16 | pass/fail | | | | | | | | | |
| P3-lite | 1280 | FP16 | pass/fail | | | | | | | | | |
| anchor-free | 1280 | FP16 | pass/fail | | | | | | | | | |
| best | 640+cascade | FP8 | pass/fail | | | | | | | | |

결론은 최고 mAP 하나가 아니라 다음 세 가지 Pareto 후보로 나눈다.

1. 최고 품질 모델
2. 최고 속도 모델
3. 운영 품질/속도 균형 모델

---

## 10. 최종 판단

현재까지의 실험과 연구를 합치면 가장 유망한 통합 설계는 다음이다.

```text
YOLOv7-L/S10 640 TRT
→ P3/P4 feature와 box metadata로 miss-risk 판단
→ 위험한 프레임만 960/1280 TRT 또는 tile
→ class-aware merge/NMS
→ 짧은 temporal aggregation
```

학습 모델 쪽에서는 `P3-lite → anchor-free → IoU-aware/WIoU` 순서로 독립 검증한다. FP8은 Blackwell + TensorRT 10.16에서 실제 layer 지원과 latency를 확인한 후 채택하며, FP4는 명시적 NVFP4 quantization graph가 없으면 단순 플래그로 시도하지 않는다.

이 설계가 만족해야 할 최종 조건은 다음이다.

```text
baseline 대비 miss recovery 증가
평균 latency 목표 이내
P95 latency 운영 한계 이내
TRT engine build 재현 가능
FP16/FP8 품질 차이 측정 완료
모델별 장단점과 실패 원인 기록
```
