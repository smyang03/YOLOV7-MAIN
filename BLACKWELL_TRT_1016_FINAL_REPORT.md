# YOLOv7 개선 설계 및 Blackwell TensorRT 10.16 최종 보고서

작성일: 2026-08-13  
브랜치: `research/yolov7-improvements`  
환경: NVIDIA RTX PRO 4000 Blackwell (GPU 1, CC 12.0), CUDA 13.0, TensorRT 10.16.1  
평가 모델: S10 (`person`, `head`), 1280 입력, 정상 decoded ONNX export

## 1. 결론

이번 측정에서는 FP8을 선택할 근거가 없다. FP8은 FP16과 품질이 사실상 같았지만 더 느렸다.

| 항목 | TRT FP16 | TRT FP8 | FP8 변화 |
|---|---:|---:|---:|
| 전체 검증 precision | 0.2673 | 0.2682 | +0.0009 |
| 전체 검증 recall | 0.7867 | 0.7868 | +0.0001 |
| 전체 검증 mAP50 | 0.7291 | 0.7295 | +0.0004 |
| 전체 검증 mAP50-95 | 0.4554 | 0.4558 | +0.0005 |
| Python evaluator 평균 | 4.70 ms | 6.31 ms | +34.2% |
| Python evaluator P95 | 4.92 ms | 6.51 ms | +32.3% |
| `trtexec` GPU median | 1.569 ms | 3.353 ms | 2.14배 |
| `trtexec` throughput | 622.8 qps | 295.2 qps | -52.6% |

따라서 현재 S10 graph에서는 **FP16을 운영 기준**으로 삼고, FP8은 다른 export graph 또는 레이어별 FP8 최적화가 확인될 때만 별도 후보로 남긴다. 단순히 `--fp8` 플래그를 추가하는 것만으로는 속도 개선이 보장되지 않는다.

## 2. 평가 조건과 데이터 검증

품질 평가는 다음 조건으로 전체 4,370장을 사용했다.

- images: `V:\00.영상파트\101.학습DB\00.안전환경\dataset\crowndataset\valid\JPEGImages`
- labels: `V:\00.영상파트\101.학습DB\00.안전환경\dataset\crowndataset\valid\labels-origin`
- input: 1280, confidence 0.01, NMS IoU 0.65
- 후보 수: low-confidence NMS 폭주 방지를 위해 이미지당 상위 3,000개
- metric: mAP50 및 mAP50-95, 고정 threshold precision/recall

`valid\labels`는 class ID가 `0,2`였고, S10의 `nc=2`와 맞지 않았다. 동일 이미지의 `labels-origin`은 `0,1`로 구성되어 있어 S10의 `person/head` 평가에는 후자를 사용했다. 이 라벨 선택을 섞으면 품질 비교가 무효가 된다.

품질 수치는 `tools/eval_trt_yolo.py`로 산출했다. 평가기의 class count는 기존 17-class 고정값을 제거하고 engine/label에서 동적으로 계산하며, NMS는 OpenCV native implementation을 사용한다.

## 3. Blackwell TensorRT 측정

`trtexec` 조건:

```text
--device=1 --warmUp=200 --iterations=1000 --duration=10
--noDataTransfers --useSpinWait --useCudaGraph --percentile=50,90,95,99
```

이 값은 preprocessing, H2D/D2H, CPU NMS를 제외한 GPU compute latency다.

| Engine | Median | P95 | P99 | Throughput |
|---|---:|---:|---:|---:|
| S10 1280 decoded FP16 | 1.569 ms | 1.612 ms | 1.616 ms | 622.814 qps |
| S10 1280 decoded FP8 | 3.353 ms | 3.367 ms | 3.372 ms | 295.235 qps |

FP8 engine build 자체는 성공했지만, 이 결과만으로 모든 convolution이 FP8 Tensor Core로 실행됐다고 해석하면 안 된다. reformat, unsupported layer fallback, graph partition을 함께 확인해야 한다. 현재 결과는 FP8의 운영 이득이 없다는 증거이며, FP8 지원 가능성 자체와는 구분한다.

## 4. 학습이 필요 없는 개선 설계

### 4.1 Threshold/NMS calibration

class별 confidence, NMS IoU, score calibration을 validation set에서 sweep한다. 특히 `person`이 검출됐지만 `head`가 없거나 score margin이 작은 경우를 별도 risk feature로 기록한다.

효과가 있는 경우는 후보 box가 이미 생성됐지만 score 또는 NMS에서 탈락하는 false negative다. detector가 box를 전혀 생성하지 못한 경우는 이 방법으로 복구할 수 없다.

### 4.2 조건부 고해상도 재추론

```text
640 또는 기본 S10 추론
  -> risk gate
  -> 선택된 프레임만 960/1280 또는 tile 재추론
  -> 좌표 복원
  -> class-aware merge/NMS
```

초기 gate는 다음 신호의 조합으로 만든다.

- person은 있는데 head가 없음
- person 수 대비 head 수가 비정상적으로 낮음
- 작은 person box 비율이 높음
- person 상단 ROI에 head 후보가 없음
- 연속 프레임에서 box가 끊김
- low-score 후보와 top-1 score 간 margin이 작음

person 자체를 놓친 경우에는 ROI 재추론만으로 복구되지 않으므로, 전체 프레임 또는 타일 fallback 조건을 별도로 둔다.

### 4.3 Temporal aggregation

영상에서는 짧은 프레임 window의 box association을 사용해 일시적인 miss를 보정한다. 단, stale box가 안전 판정을 오염시키지 않도록 TTL과 이동/appearance consistency gate를 둔다.

### 4.4 Feature similarity의 역할

이미지 프롬프트에서 뽑은 helmet/head feature와 YOLO feature의 cosine similarity는 새 detection box를 직접 생성하는 방식보다 **재추론 여부를 결정하는 risk signal**로 사용하는 것이 안전하다. feature similarity만으로 box를 만들면 localization과 background false positive를 제어하기 어렵다.

## 5. 학습이 필요한 개선 설계

증류는 사용자 결정에 따라 제외한다.

### 5.1 P3-lite / P2 ablation

작은 head가 핵심이면 P3를 완전히 제거하지 말고 P3-lite를 기준선으로 둔다. P2 추가는 작은 객체 recall에는 유리할 수 있지만 메모리와 latency를 증가시키므로 S10 속도 목표와 함께 측정한다.

### 5.2 Anchor-free 비교

anchor-free head는 anchor 설정과 autoanchor 의존성을 줄이고 crowded/scale variation에 대한 후보 생성을 개선할 가능성이 있다. 다만 YOLOv7의 head, loss, decode, ONNX/TRT export를 한 묶음으로 바꿔야 하므로 단순 cfg 변경으로 판단하지 않는다.

비교 순서는 다음과 같다.

1. S10 anchor baseline
2. S10 anchor-free
3. YOLOv7-L anchor baseline
4. YOLOv7-L anchor-free
5. W6 anchor baseline
6. W6 anchor-free

각 실험은 같은 seed/epoch/image size/batch/data split으로 비교하고, 작은 객체 recall과 miss subset을 별도로 기록한다.

### 5.3 Loss와 hard-example replay

IoU loss(WIoU/Focaler-IoU 계열)는 crowded box와 quality ranking 개선 후보로 검증한다. 전체 데이터 재학습보다 다음 hard subset을 반복 노출하는 것이 효율적이다.

- head 미감지
- helmet/head가 겹치거나 가려진 샘플
- 작은 box
- person은 맞지만 head class가 빠진 샘플
- false positive가 반복되는 배경

단, hard replay는 validation leakage가 없도록 train-only mining으로 운영한다.

## 6. 통합 운영 설계

최종 운영 후보는 다음 구조다.

```text
TRT FP16 S10 640/기본 해상도
  -> cheap metadata + feature risk gate
  -> 대부분은 즉시 종료
  -> 위험 프레임만 960/1280 또는 tile TRT FP16
  -> temporal association
  -> class-aware merge/NMS
  -> 안전 판정
```

학습 측면에서는 S10 P3-lite, P2, anchor-free, loss/hard-replay를 독립 ablation으로 검증한 뒤 가장 좋은 한 가지를 1차 모델로 채택한다. 그 모델을 640 1차 엔진으로 만들고, 1280 엔진은 miss recovery branch로 사용한다.

이 구조가 목표 속도와 품질을 동시에 만족시키는 이유는 모든 프레임에 비싼 1280 추론을 적용하지 않고, 모델이 확신하지 못하는 경우에만 계산을 추가하기 때문이다. risk gate가 과도하게 발동하면 평균 속도가 악화되므로 trigger rate와 recovery recall을 함께 최적화해야 한다.

## 7. 최종 판단과 다음 검증 gate

### 7.1 RTX 4090 추가 측정

동일한 decoded ONNX에서 RTX 4090용 engine을 별도로 build하고 GPU 0에서 측정했다. 4090 engine은 Blackwell engine을 재사용하지 않았다.

| 항목 | RTX 4090 FP16 | RTX PRO 4000 Blackwell FP16 |
|---|---:|---:|
| Compute Capability | 8.9 | 12.0 |
| TRT GPU median | 1.053 ms | 1.569 ms |
| TRT GPU P95 | 1.315 ms | 1.612 ms |
| Throughput | 894.5 qps | 622.8 qps |
| 200장 mAP50-95 | 0.5495 | 0.5497 (동일 조건 표본) |

따라서 이 S10 graph에서는 4090이 단일 stream GPU-only 추론에서 약 33% 빠르게 측정됐다. 실제 운영 속도는 preprocessing, H2D/D2H, NMS, clock/power 상태에 따라 달라지므로 이 표를 end-to-end 성능으로 해석하면 안 된다.

RTX 4090에서 `--fp8`도 시도했지만 TensorRT 10.16.1 build log에 `Unsupported data type FP8`이 반복됐다. build 명령은 종료 성공했으나 FP8 layer를 정상 실행하는 engine으로 확인되지 않았고, 생성 engine은 FP16보다 큰 49.6 MiB였다. 따라서 4090 FP8 latency/품질 수치는 유효한 FP8 비교로 기록하지 않는다.

4090 engine은 builder optimization level 3, Blackwell 기준 engine은 level 5로 생성되어 build tactic 조건이 완전히 같지는 않다. GPU 비교는 동일 ONNX와 동일 inference command의 참고값으로 사용한다.

### 7.2 W6 추가 측정

프로젝트에 존재하는 `runs/crowdhuman_train/w6_crowdhuman_teacher2/weights/best.pt`를 decoded ONNX로 export한 뒤 Blackwell GPU 1에서 TRT 10.16.1 FP16으로 측정했다. 이 checkpoint는 순수 COCO pretrained weight가 아니라 `person/head` CrowdHuman 학습 checkpoint다.

| 항목 | S10 FP16 | W6 FP16 |
|---|---:|---:|
| TRT GPU median | 1.569 ms | 5.350 ms |
| TRT GPU P95 | 1.612 ms | 5.558 ms |
| Throughput | 622.8 qps | 184.1 qps |
| 200장 mAP50-95 | 0.5497 | 0.5976 |

W6는 표본 품질이 더 높았지만 S10보다 약 3.4배 느렸다. 따라서 W6는 품질 우선의 고해상도 fallback 후보이고, 실시간 1차 검출기는 S10이 적합하다. 200장 품질 수치는 전체 validation 대체가 아닌 동일 조건의 빠른 비교값이다.

현재 Z 드라이브에는 YOLOv7-L의 실제 checkpoint/정상 ONNX가 없고, 기존 `yolov7l_640` engine은 ModelOpt/NVFP4 graph parity가 확인되지 않아 YOLOv7-L 결과로 사용하지 않았다. L checkpoint를 확보하면 같은 decoded export와 Blackwell 측정 조건으로 추가해야 한다.

### 7.3 D:\code\yolov7 640 export 추가 측정

사용자가 제공한 `D:\code\yolov7\gop` 폴더에서 배포용 ONNX를 확인하고 Blackwell GPU 1에서 TRT 10.16.1 FP16으로 측정했다. 두 파일 모두 EfficientNMS 출력이 포함되어 있다.

| 모델 | ONNX | 실제 입력 | TRT GPU median | P95 | Throughput |
|---|---|---|---:|---:|---:|
| YOLOv7-L export | `large-640360.onnx` | 640×384 | 1.321 ms | 1.358 ms | 736.6 qps |
| YOLOv7-W6 export | `w6-640360.onnx` | 640×384 | 1.303 ms | 1.321 ms | 753.7 qps |

따라서 이 두 export에서는 W6가 L보다 약간 빠르게 측정됐다. 다만 S10 1280, W6 1280과는 입력 크기와 graph가 다르므로 latency를 직접적인 모델 품질 비교로 해석하지 않는다. 또한 이 ONNX들은 현재 평가 데이터셋과 class mapping이 확인되지 않아 이번에는 속도 측정만 유효한 비교로 기록한다.

현재 확정할 수 있는 내용:

1. S10 정상 decoded export는 TRT 10.16.1/Blackwell에서 FP16 및 FP8 engine build가 가능하다.
2. 현재 graph에서는 FP16이 FP8보다 확실히 빠르다.
3. FP8 품질 차이는 전체 4,370장 기준 사실상 없다.
4. 따라서 운영 배포 기준은 TRT FP16이며, 품질 개선은 precision 변경보다 risk-gated high-resolution cascade와 학습 ablation에서 찾는 것이 타당하다.

아직 별도 gate가 필요한 항목:

- 640 1차 + 1280 fallback의 실제 영상 end-to-end latency
- preprocessing, H2D/D2H, NMS 포함 P95
- anchor-free/P3-lite/P2 학습 모델의 동일 TRT export parity
- 실제 miss subset에서 recovery recall과 gate 발동률
- FP8 layer profile을 통한 실제 FP8 실행 비율

관련 구현:

- [통합 설계](BLACKWELL_TRT_INTEGRATED_DESIGN.md)
- [TRT 평가기](tools/eval_trt_yolo.py)
- [decoded ONNX export](tools/export_decoded_onnx.py)
- [Blackwell benchmark script](scripts/run_blackwell_trt_benchmark.sh)
