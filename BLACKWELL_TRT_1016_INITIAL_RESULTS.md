# Blackwell TensorRT 10.16.1 초기 측정 결과

측정일: 2026-08-13  
GPU: NVIDIA RTX PRO 4000 Blackwell  
Compute Capability: 12.0  
CUDA: 13.0  
TensorRT: 10.16.1  
GPU index: 1

## 1. 먼저 확인된 호환성 문제

기존에 TensorRT 10.14 계열에서 생성된 engine을 TensorRT 10.16.1로 바로 로드하면 다음 오류가 발생했다.

```text
The engine plan file is not compatible with this version of TensorRT
expecting library version 10.16.1.11
```

따라서 TensorRT engine은 ONNX에서 Blackwell와 실제 배포 TensorRT 버전으로 다시 build해야 한다. 이전 engine의 latency를 10.16.1 결과로 사용하면 안 된다.

## 2. TensorRT 10.16.1에서 새로 build한 결과

### 2.1 GPU compute-only latency

조건:

- `--device=1`
- warmup 200회
- duration 10초
- iterations 1000
- `--noDataTransfers`
- `--useCudaGraph`
- `--useSpinWait`
- percentile 50/90/95/99

| Engine | Build | Median | P95 | Throughput | Engine size | 상태 |
|---|---|---:|---:|---:|---:|---|
| YOLOv7-L 640 FP16 | pass | 1.930 ms | 1.951 ms | 508.5 qps | 71.9 MiB | 정상 |
| YOLOv7-L 640 FP8 plain ONNX | pass | 5.046 ms | 5.160 ms | 196.3 qps | 140.9 MiB | 정상 build, 기대와 다른 성능 |
| YOLOv7-L 640 NVFP4 graph | pass | 약 5.045 ms | 약 5.162 ms | 약 196.3 qps | 140.9 MiB | 실제 FP4로 확인되지 않음 |

FP8 plain engine은 `--fp8` build 자체는 성공했지만 FP16보다 느렸다. TensorRT log의 출력은 `FP32+FP8`이며, 모든 layer가 FP8 tensor core로 실행된다는 뜻은 아니다. layer fallback/reformat과 graph 구성 확인이 필요하다.

NVFP4 graph는 build log에서 precision이 `FP32`로 표시되었으므로, 이 결과를 NVFP4 성능으로 부르면 안 된다. 명시적 NVFP4 quantization graph와 layer precision 확인이 선행되어야 한다.

## 3. 품질 평가에서 확인된 export parity 문제

TensorRT 10.16.1 FP16 engine을 Dongseo 2,610장에 평가했을 때:

```text
precision: 0.1726
recall:    0.5929
mAP50:     0.1838
mAP50-95:  0.03793
```

이 값은 기존 TensorRT 10.14 engine 결과와 직접 비교할 수 없다. 새 engine의 출력 tensor를 확인한 결과 decoded output으로 보이는 `[1,25200,22]`는 생성되지만, 첫 후보 값과 출력 범위가 기존 evaluator가 기대하는 확률/box 표현과 일치하지 않았다.

```text
output range: 0.0 ~ 637.0
```

현재 사용한 `yolov7l_640_nvfp4_plain.onnx`는 ModelOpt NVFP4 export 과정에서 생성된 graph이며, 일반 YOLOv7 export와 동일한 decoded-head parity가 보장되지 않는다. 따라서 이 품질값은 “새 engine의 품질이 낮다”가 아니라 “ONNX export/decode/evaluator parity 검증 실패”로 분류한다.

## 4. 이전 10.14 engine의 참고 결과

다음은 기존 engine과 TensorRT 10.14 Python evaluator에서 얻은 참고값이다.

| Engine | Recall | mAP50 | mAP50-95 | 비고 |
|---|---:|---:|---:|---|
| FP16 | 0.9931 | 0.9906 | 0.8213 | 10.14 engine |
| FP8 | 0.9931 | 0.9906 | 0.8212 | 10.14 engine |
| NVFP4 | 0.0000 | 0.0000 | 0.0000 | 당시 graph/출력 문제 |

이 수치는 모델 품질 참고용이며, Blackwell TensorRT 10.16.1의 최종 수치가 아니다. 10.16.1용 정상 export를 다시 만든 뒤 동일 evaluator로 재측정해야 한다.

## 5. 현재 판단

### 확정된 것

1. 이 PC에서 Blackwell GPU 1과 CUDA 13.0, TensorRT 10.16.1이 정상 인식된다.
2. 이전 TRT engine은 10.16.1에서 재사용할 수 없다.
3. YOLOv7-L 640 FP16 engine은 TRT 10.16.1/Blackwell에서 약 1.93ms GPU compute로 실행된다.
4. 단순 `--fp8` build가 FP16보다 빠르다는 보장은 없다.
5. NVFP4 ONNX를 TensorRT에 넣었다고 실제 NVFP4 engine이 되는 것은 아니다.

### 아직 확정할 수 없는 것

1. 정상 decoded export 기준 FP16/FP8 품질
2. FP8이 모든 주요 convolution을 실제 FP8로 실행하는지
3. 진짜 NVFP4의 품질/latency
4. H2D/D2H, preprocessing, NMS를 포함한 end-to-end latency
5. S10/P3-lite/anchor-free 학습 모델의 Blackwell TRT 결과

## 6. 다음 조치

1. 원래 FP16 engine을 만들었던 동일한 custom YOLOv7-L checkpoint/정상 export 경로를 확보한다.
2. export 단계에서 detection module의 `export=True`, decoded output, output shape를 검증한다.
3. PyTorch FP16 vs ONNXRuntime vs TRT FP16의 동일 이미지 출력 오차를 비교한다.
4. parity가 통과한 ONNX에서 FP16/FP8을 다시 build한다.
5. FP8 layer precision/profile을 확인한다.
6. NVFP4는 ModelOpt explicit graph와 TensorRT 10.16 지원 여부를 별도 gate로 평가한다.
7. 정상 engine에 대해 Dongseo full validation과 miss recovery를 다시 수행한다.

## 7. 관련 파일

- 통합 설계: [BLACKWELL_TRT_INTEGRATED_DESIGN.md](BLACKWELL_TRT_INTEGRATED_DESIGN.md)
- 측정 스크립트: [scripts/run_blackwell_trt_benchmark.sh](scripts/run_blackwell_trt_benchmark.sh)
- TRT 품질 evaluator: [tools/eval_trt_yolo.py](tools/eval_trt_yolo.py)
