# YOLOv7 계열 약점 개선을 위한 폭넓은 연구 및 적용 설계

작성일: 2026-08-13

## 0. 결론부터

현재 목표가 **YOLOv7-L TRT의 속도를 유지하면서 안전모/머리 같은 작은 객체의 미감지를 줄이는 것**이라면, 가장 가능성이 높은 방향은 다음 순서다.

1. **기본 모델의 입력 해상도와 작은 객체 픽셀 크기 분포를 먼저 고정 측정**한다.
2. 전체 이미지에 무거운 두 번째 모델을 돌리지 않고, **저비용 미감지 위험도(risk) 판단기**를 붙인다.
3. 위험한 프레임에만 crop/tile 또는 960/1280 재추론을 수행한다.
4. 학습 가능한 경우에는 P3를 유지한 경량 head, 선택적 P2, IoU-aware score, occlusion-aware assignment/loss를 각각 ablation한다.
5. 모델 구조를 크게 바꾸기 전에 **score calibration, NMS, hard-negative mining, 작은 객체 전용 샘플링**을 검증한다.

반대로 다음은 현재 목적에 우선순위가 낮다.

- 단순 attention 블록을 백본에 계속 추가하는 것
- YOLOv7에 YOLOv9/YOLOv10 구조를 부분적으로 섞는 것
- feature cosine prototype만으로 누락 객체를 복원하는 것
- 전체 프레임마다 CLIP/DINO/YOLOE 같은 대형 비전 인코더를 실행하는 것
- 실제 miss 원인 분석 없이 conf만 낮추는 것

이 결론은 “라벨 품질이 완벽하다”는 뜻이 아니다. 라벨 문제는 이번 비교에서 base와 개선 모델에 동일하게 존재하므로 **모델 개선 효과를 판단하는 변수에서 일단 분리**한다.

---

## 1. 현재 실험에서 이미 확인된 사실

Dongseo-food 데이터의 실제 YOLOv7 miss set 기준으로 확인된 결과:

| 방법 | miss 복구 | 해석 |
|---|---:|---|
| YOLOv7-L 640, conf 0.25 | 기준 | 36개 miss image 생성 |
| conf를 0.01까지 하향 | 0/36 추가 복구 | 단순 score threshold 문제가 아님 |
| P3 prototype bank, 640 | 4/36 | feature 유사도만으로는 false positive가 커짐 |
| person ROI rescue | 0/36 | 사람 crop만으로는 후보 생성 실패 |
| 동일 모델 960 재추론 | 8/36 | 해상도 증가가 일부 유효 |
| 동일 모델 1280 재추론 | 13/36 | 작은 객체의 공간 정보 손실이 주요 원인일 가능성 |

이 결과는 현재 문제를 다음처럼 좁힌다.

> “못 잡은 객체의 feature를 찾아 분류하는 문제”라기보다, 640 입력에서 **객체가 detection head의 유효한 후보로 올라오지 못하는 문제**가 상당수다.

따라서 feature prompt는 1차 detector가 이미 후보를 만든 경우에는 보조 신호가 될 수 있지만, 후보 자체가 없는 FN을 안정적으로 복구하는 주력 방법으로 보기 어렵다.

---

## 2. YOLO 계열의 약점을 원인별로 분해

### 2.1 작은 객체와 downsampling

작은 객체는 backbone/neck의 stride가 커질수록 몇 개의 feature cell에만 남는다. 이 과정에서 경계, 색상, 모양 신호가 사라지고 주변 배경과 섞인다. 작은 객체 연구 survey들은 공통적으로 약한 feature representation, downsampling에 따른 세부 정보 소실, clutter, scale imbalance를 핵심 문제로 분류한다.

관련 방향은 네 가지다.

- P3 유지 또는 P3-lite: 작은 객체에 가장 직접적이고 YOLOv7 구조와 호환성이 높다.
- P2 추가: recall 잠재력은 가장 크지만 FLOPs와 메모리, NMS 후보가 증가한다.
- 입력 해상도 증가: 구조 변경 없이 효과를 확인하기 가장 쉽다.
- tile/sparse high-resolution: 전체 고해상도 비용을 피하고 작은 영역만 확대한다.

FPN은 top-down multi-scale feature fusion을 제안했지만, 단순 fusion은 서로 다른 stride의 의미/공간 정렬 문제를 일으킬 수 있다. EFPN, Trident Pyramid 등은 이 문제를 feature fusion 또는 multi-branch scale representation으로 다룬다. 그러나 YOLOv7-L TRT 속도 목표에서는 full multi-branch보다 P3-lite와 selective high-resolution이 더 현실적이다.

### 2.2 혼잡, 가림, 중복 객체

CrowdHuman처럼 사람이 겹치는 데이터에서는 다음 세 문제가 같이 발생한다.

1. 한 cell/anchor가 여러 인접 객체를 대표하기 어려움
2. 일부만 보이는 객체의 objectness가 낮아짐
3. NMS가 겹친 후보 중 하나를 제거하면서 정상 객체까지 사라짐

YOLO의 anchor 기반 positive assignment는 데이터의 크기/종횡비에 민감하고, 밀집 객체에서는 위치 경쟁이 심해진다. anchor-free가 항상 더 정확한 것은 아니지만, anchor 설정에 대한 민감도와 작은/비정형 객체의 매칭 실패를 줄일 수 있는 비교군이다.

NMS는 별도의 약점이다. detector가 후보를 잘 만든 뒤에도 겹친 상자를 제거할 수 있다. 따라서 구조 변경과 별개로 class-aware NMS, Soft-NMS, DIoU-NMS, NMS IoU/score sweep을 반드시 따로 평가해야 한다. 특히 안전모는 사람/머리와 공간적으로 가까우므로 일반적인 global NMS 설정이 최적이라는 보장이 없다.

### 2.3 classification score와 localization quality의 불일치

YOLO의 class/objectness 점수가 “그 후보가 실제 객체인지”와 “box가 얼마나 정확한지”를 완전히 같은 의미로 표현하지 않는다. 높은 class score지만 box가 나쁜 후보가 먼저 남거나, 실제 객체지만 localization이 불안정한 작은 후보가 낮은 score로 밀릴 수 있다.

VarifocalNet과 Generalized Focal Loss 계열은 class confidence와 IoU quality를 하나의 ranking signal로 결합하려는 접근이다. GFLv2는 box distribution 자체에서 localization quality를 추정한다. 이는 단순히 confidence threshold를 낮추는 것보다 **후보 순위와 NMS 전 score를 개선**하는 방향이다.

YOLOv7에 적용할 때는 full GFL head를 바로 이식하기보다 다음의 저위험 ablation이 낫다.

- objectness/class score에 predicted IoU quality를 곱하는 IoU-aware ranking head
- positive/negative imbalance에 Focal/Varifocal류 weighting 적용
- box loss를 CIoU baseline과 WIoU/Focaler-IoU로 비교
- training 후 temperature scaling 또는 class/size별 calibration

### 2.4 loss와 positive assignment

Loss를 바꾸면 validation loss가 좋아져도 실제 miss recall이 좋아진다는 보장은 없다. 작은 객체와 가림 객체에서는 positive sample 수 자체가 부족할 수 있기 때문이다.

검증해야 할 축:

- anchor threshold와 autoanchor 결과
- center/neighbor assignment 범위
- 작은 box를 positive로 받는 head의 비율
- crowded object 간 assignment 충돌
- box regression loss와 objectness loss의 상대적 크기

추천 실험은 loss를 한꺼번에 바꾸지 않고, **동일 seed/epoch에서 assignment 로그를 먼저 저장**하는 것이다. 특히 작은 객체의 positive 수, head별 positive 수, GT당 매칭 후보 수를 기록하면 “loss 문제”와 “후보 생성 문제”를 구분할 수 있다.

### 2.5 label noise와 missing annotation

이번 모델 비교에서는 라벨 품질을 공통 조건으로 취급하지만, 실제 학습 안정성에는 여전히 영향을 준다. 문헌상 detection annotation noise는 missing labels, extra labels, class shift, inaccurate boxes로 나눌 수 있다. noisy box 연구들은 box 회귀가 classification보다 라벨 오차에 더 민감하다고 보고한다.

따라서 전체 라벨을 다시 만드는 것만이 방법은 아니다.

- 학습 초반 model prediction과 GT 불일치가 큰 sample을 audit queue로 보낸다.
- 작은 객체/가림 객체의 box를 별도 검수한다.
- 확신이 낮은 GT를 즉시 삭제하지 말고 loss weight를 낮춘다.
- empty image와 실제 negative image를 구분한다.
- positive가 없는 이미지가 “라벨 누락”인지 “진짜 배경”인지 split한다.

이 영역은 성능의 절대 상한을 높이지만, 이번 목표의 1차 모델 구조 비교와는 분리해 두는 것이 맞다.

### 2.6 class imbalance와 hard negative

CrowdHuman에서 person이 head보다 압도적으로 많은 경우, dense detector의 gradient가 person/background에 편중될 수 있다. long-tail detection 연구는 많은 샘플을 가진 클래스가 feature learning에 더 큰 영향을 주며, 클래스별 샘플 균형화가 중요하다고 지적한다.

우리 데이터에서 필요한 것은 단순 oversampling보다 다음이다.

- head가 작은 이미지/가림 이미지의 batch 비율 증가
- person은 잘 맞지만 head가 빠지는 장면을 hard positive로 재샘플링
- 안전모 false positive가 나는 배경을 hard negative로 재투입
- class별 AP가 아니라 size/occlusion/scene별 recall을 함께 기록

Focal류 loss만 적용하면 easy negative는 줄지만, 누락된 head의 feature가 좋아진다는 보장은 없다. sampler와 crop 정책을 함께 봐야 한다.

### 2.7 domain shift와 영상 조건

학습 데이터와 실제 현장의 카메라, 조명, 압축, 각도, 작업복, 안전모 종류가 다르면 같은 detector라도 confidence와 feature 분포가 바뀐다. domain-adaptive YOLO 연구들은 detector의 domain shift를 별도 문제로 다룬다.

다만 사용자 목표가 고정된 영상 환경이라면 복잡한 adversarial domain adaptation보다 비용이 낮은 방법이 우선이다.

- train-time style augmentation: 밝기, blur, compression, glare, color temperature
- target 영상의 unlabeled frame으로 confidence/feature 분포만 측정
- test-time에는 BN 통계나 weight를 무분별하게 업데이트하지 않음
- calibration set으로 camera별 threshold를 분리

온라인 self-training은 pseudo-label 오류가 누적되고 confirmation bias가 생길 수 있으므로, 운영 안정성이 중요하면 기본 선택으로 삼지 않는다.

### 2.8 calibration과 uncertainty

confidence 0.01에서도 실제 miss가 회복되지 않았던 실험은 “threshold만 조절하면 해결”되는 유형이 아니라는 근거다. 그러나 confidence가 실제 실패 확률을 잘 반영하는지도 별도 검증해야 한다.

신뢰도 보정은 다음 용도로 유용하다.

- 낮은 confidence 후보를 재추론 trigger로 사용
- 위험 프레임을 triage
- camera/조명별 threshold 조정
- false positive와 false negative 비용을 반영한 operating point 선택

주의할 점은 calibration이 detector의 새 객체를 만들어주지는 않는다는 것이다. calibration은 **언제 detector를 믿지 말아야 하는지**를 더 잘 알려주는 장치다.

---

## 3. 구조 변경 후보 비교

| 후보 | 기대하는 약점 개선 | 속도 영향 | 현재 목표 적합도 | 판단 |
|---|---|---:|---:|---|
| YOLOv7-L baseline | 기준 | 기준 | 매우 높음 | 반드시 유지 |
| P3-lite head | 작은 객체 공간 정보 | 낮음~중간 | 매우 높음 | 1순위 구조 실험 |
| P2 full head | 극소 객체 recall | 중간~큼 | 중간 | recall 상한 확인용 |
| Anchor-free head | anchor/assignment 민감도 | 낮음~중간 | 높음 | 별도 100 epoch 비교 |
| attention 추가 | 배경 억제/문맥 | 낮음~중간 | 중간 이하 | 효과 검증 없이는 후순위 |
| deformable convolution | 변형/가림/정렬 | 중간~큼, TRT 주의 | 중간 | 실제 export 가능성 먼저 확인 |
| transformer neck | 전역 문맥 | 중간~큼 | 낮음~중간 | 속도 목표와 충돌 가능 |
| full high-res branch | 작은 객체 | 큼 | 낮음 | selective 방식이 대안 |
| YOLOE/DINO feature prompt | open-vocabulary/semantic 후보 | 매우 큼 | 낮음 | 후보 복구 주력으로는 부적합 |
| DETR 계열 외부 baseline | NMS/assignment 대안 | 모델별 상이 | 중간 | 별도 기준선으로만 비교 |

### P3-lite가 중요한 이유

현재 960/1280 재추론에서 회복된 miss가 있으므로, 작은 객체 정보가 해상도에서 손실되는 비중이 확인됐다. 그렇다고 full P2를 바로 넣으면 전체 추론 비용이 증가한다. P3-lite는 다음 절충이다.

- P3 feature의 채널 수만 축소
- P3 head를 작은 객체 클래스 중심으로 유지
- P4/P5는 기존 YOLOv7-L 경로 유지
- export/TRT graph를 단순하게 유지

다만 P3를 제거하거나 약화하면 작은 객체 AP/recall이 떨어질 수 있다는 연구 관찰과 일치하므로, P3 removal은 개선 방향이 아니라 반드시 피해야 할 대조군이다.

### Anchor-free는 만능 해결책이 아니다

anchor-free는 anchor prior와 매칭 민감도를 줄일 수 있지만, objectness/assignment/NMS 문제를 자동으로 해결하지 않는다. crowded scene에서 center가 겹치면 오히려 별도 assignment 설계가 필요하다. 따라서 anchor-based baseline, autoanchor, anchor-free를 **같은 input/epoch/batch/augmentation으로 비교**해야 한다.

---

## 4. 속도를 지키는 핵심 방향: 조건부 재추론

전체 프레임을 1280으로 올리는 대신, 640 1차 추론에서 실패 위험이 높은 프레임만 960/1280 또는 tile로 재추론한다.

### 4.1 trigger 신호

trigger는 하나의 confidence threshold가 아니라 여러 저비용 신호를 결합해야 한다.

- person 수는 많은데 head 수가 비정상적으로 적음
- person box 내부에 head 후보가 전혀 없음
- person/head 비율이 camera별 정상 범위에서 이탈
- 작은 head 후보의 objectness/IoU가 불안정함
- augmentation 또는 짧은 temporal window에서 prediction이 불일치
- feature/box uncertainty가 높은 영역
- frame difference 또는 optical flow가 큰 영역

### 4.2 재추론 방식

- 고정 960/1280 full-frame: 구현이 가장 쉽고 효과 상한 확인용
- person 주변 crop: 사람 탐지 자체가 맞을 때만 유효
- head 상단 prior crop: 도메인 지식이 강한 경우 비용이 낮음
- 2x2/3x3 tile: 사람 검출이 실패하는 경우도 복구 가능
- QueryDet식 sparse high-resolution: coarse feature에서 관심 위치를 고르고 고해상도 feature는 sparse하게 계산

QueryDet은 저해상도 feature로 coarse location을 얻은 뒤 고해상도 feature를 sparse query로 계산하여 작은 객체 성능과 효율을 함께 노리는 대표적 방향이다. YOLOv7에 그대로 이식하기보다, 먼저 외부 crop/tile cascade로 효과를 확인한 후 필요하면 내부화하는 것이 안전하다.

### 4.3 목표 latency 계산

평균 latency는 다음으로 계산해야 한다.

`평균 비용 = 1차 640 비용 + trigger_rate × 재추론 비용`

예를 들어 640 비용을 1이라고 할 때, 1280 full이 3이라면 trigger rate 20%에서 평균은 1.4다. 따라서 전체 1280보다 훨씬 저렴할 수 있다. 단, 실제 TRT latency와 trigger rate를 함께 기록해야 한다.

---

## 5. feature prompt / KGFP / failure prediction의 정확한 위치

Knowledge-Guided Failure Prediction(KGFP) 계열과 “Did You Miss the Sign?” 계열은 detector 자체를 바꾸지 않고 내부 feature 또는 외부 vision embedding으로 **검출 실패 가능성**을 예측한다. 이들은 “누락된 객체의 box를 직접 생성하는 detector”라기보다 monitor/risk predictor에 가깝다.

따라서 현재 문제에 다음처럼 적용하는 것이 맞다.

1. YOLOv7-L 640 결과와 중간 feature를 입력으로 받는 작은 FN-risk predictor 학습
2. risk가 높을 때만 960/1280 또는 tile 재추론
3. 재추론 결과를 기존 결과와 NMS/merge
4. risk predictor의 목표는 box 생성이 아니라 “이 frame/ROI를 다시 볼 가치가 있는가”

현재 P3 cosine prototype에서 false positive가 많이 발생하고 실제 miss 복구가 제한적이었던 결과는 이 해석과 맞는다. feature similarity를 detector 대신 쓰기보다, **재추론 trigger**로 격하하면 실용성이 올라간다.

이 방식은 소량의 라벨만으로도 만들 수 있지만, “miss set”과 “정상 detection”을 분리한 학습/evaluation split이 필요하다. 평가 지표는 risk predictor accuracy가 아니라 다음이어야 한다.

- 같은 평균 FPS에서 recall 증가
- trigger rate
- miss recovery / 추가 false positive
- 95/99 percentile latency
- frame 단위와 object 단위의 recall

---

## 6. 실제로 진행할 실험 매트릭스

### 단계 A: 기준선과 진단

1. YOLOv7-L 640 TRT baseline
2. YOLOv7-L 960/1280 PyTorch 및 TRT latency/recall
3. confidence, NMS IoU, class별 threshold sweep
4. GT box 크기, aspect ratio, occlusion, image edge, person-head distance 분포
5. head별(P3/P4/P5) positive 수와 miss 위치 로그

### 단계 B: 가장 비용 대비 효과가 좋은 학습 변경

1. P3-lite
2. anchor-free head
3. P3-lite + anchor-free
4. IoU-aware ranking score
5. CIoU vs WIoU/Focaler-IoU
6. head-heavy sampler + hard negative replay

각 실험은 같은 seed 2~3개가 이상적이며, 최소한 동일한 100/300 epoch 조건과 동일한 데이터 순서를 유지해야 한다. loss와 구조와 sampler를 한 번에 바꾸면 원인을 알 수 없다.

### 단계 C: 학습 없이 속도를 유지하는 cascade

1. 640 baseline
2. 고정 960/1280
3. person/head ratio trigger
4. feature/score instability trigger
5. trigger 조합 + 960 crop
6. trigger 조합 + tile

최종 선택은 최고 AP가 아니라 다음 조건으로 한다.

`recall 개선 / 평균 latency 증가`가 가장 큰 방법

### 단계 D: 운영 안정화

- camera별 calibration
- 영상 temporal consensus
- frame 간 box tracking/association
- 동일 객체가 여러 프레임에서 계속 miss되는지 monitor
- drift 감지 후 재학습 queue 생성

영상에서는 단일 프레임 재검출보다 짧은 temporal aggregation이 값싸게 false negative를 줄일 수 있다. 다만 안전모 착용 여부가 빠르게 변하는 장면에서는 지연과 stale box를 제한해야 한다.

---

## 7. 우선순위 최종판

### 가장 먼저 할 것

1. 640/960/1280 TRT 실제 latency를 같은 환경에서 재측정
2. NMS/threshold sweep으로 post-processing 손실 분리
3. P3-lite와 anchor-free를 별도 학습
4. miss image의 크기/가림/경계/조명별 bucket 분석
5. 조건부 1280 cascade prototype 작성

### 다음으로 할 것

6. IoU-aware ranking 또는 Varifocal류 head
7. hard-negative replay와 head-heavy sampler
8. 작은 객체 augmentation 및 targeted crop
9. 짧은 temporal consistency/tracking
10. FN-risk predictor를 재추론 trigger로 학습

### 후순위

11. P2 full head
12. deformable convolution
13. transformer neck
14. YOLOE/DINO feature encoder를 전체 프레임에 적용
15. anchor-free와 여러 attention/loss를 한 번에 혼합

## 8. 연구적으로 가장 유망한 신규 설계

현재 상황에 가장 잘 맞는 설계는 다음이다.

### “YOLOv7-L 640 + Failure-Aware Selective High-Resolution Cascade”

- 1차: 기존 YOLOv7-L 640 TRT
- monitor: YOLO feature, person/head count relation, score margin, temporal inconsistency를 이용한 작은 risk head
- 2차: risk ROI만 960/1280 또는 tile 재검출
- merge: class-aware Soft-NMS 또는 IoU-aware score
- training: 실제 miss set을 hard positive로 사용
- no distillation: teacher/student 구조 없이 독립적으로 학습

이 설계는 detector의 약점을 “feature 유사도만으로 새 box를 상상”하려 하지 않고, detector가 놓칠 가능성이 있는 상황에서만 공간 정보를 추가로 투입한다. 따라서 현재 확인된 640→1280 회복 효과를 활용하면서 평균 속도는 trigger rate로 제어할 수 있다.

## 9. 참고 논문 및 자료

- Small-object YOLO survey: [A Comprehensive Literature Review on YOLO-Based Small Object Detection](https://www.sciencedirect.com/org/science/article/pii/S1546221826001943)
- FPN: [Feature Pyramid Networks for Object Detection](https://arxiv.org/pdf/1612.03144)
- EFPN: [Enhancing Feature Pyramid Networks for Small Object Detection](https://arxiv.org/abs/2003.07021)
- QueryDet: [Cascaded Sparse Query for Accelerating High-Resolution Small Object Detection](https://openaccess.thecvf.com/content/CVPR2022/papers/Yang_QueryDet_Cascaded_Sparse_Query_for_Accelerating_High-Resolution_Small_Object_Detection_CVPR_2022_paper.pdf)
- VarifocalNet: [An IoU-Aware Dense Object Detector](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_VarifocalNet_An_IoU-Aware_Dense_Object_Detector_CVPR2021_paper.pdf)
- GFL: [Generalized Focal Loss](https://proceedings.neurips.cc/paper_files/paper/2020/file/f0bda020d2470f2e74990a07a607ebd9-Paper.pdf)
- Wise-IoU: [Bounding Box Regression Loss with Dynamic Focusing Mechanism](https://arxiv.org/abs/2301.10051)
- Long-tail detection: [Factors in Finetuning Deep Model for Object Detection With Long-Tail Distribution](https://openaccess.thecvf.com/content_cvpr_2016/html/Ouyang_Factors_in_Finetuning_CVPR_2016_paper.html)
- Noisy boxes: [Robust Object Detection With Inaccurate Bounding Boxes](https://arxiv.org/abs/2207.09697)
- Domain adaptation: [Domain Adaptive YOLO](https://proceedings.mlr.press/v157/zhang21c.html)
- Failure prediction: [Did You Miss the Sign?](https://arxiv.org/abs/1903.06391)
- KGFP: [Knowledge-Guided Failure Prediction](https://openaccess.thecvf.com/content/CVPR2026W/SAIAD/html/Zimmermann_Knowledge-Guided_Failure_Prediction_Detecting_When_Object_Detectors_Miss_Safety-Critical_Objects_CVPRW_2026_paper.html)
- Introspective detection: [Introspective False Negative Prediction for Object Detection](https://pmc.ncbi.nlm.nih.gov/articles/PMC8073889/)
- Self-aware detector calibration: [Towards Building Self-Aware Object Detectors](https://openaccess.thecvf.com/content/CVPR2023/papers/Oksuz_Towards_Building_Self-Aware_Object_Detectors_via_Reliable_Uncertainty_Quantification_and_CVPR2023_paper.pdf)

## 최종 판단

현재 방향은 “백본을 계속 키우는 것”에서 “작은 객체가 후보가 되기 전 어디서 정보가 사라지는지 측정하고, 필요한 순간에만 고해상도 계산을 추가하는 것”으로 바꾸는 것이 맞다.

학습 모델만 고른다면 `YOLOv7-L baseline → P3-lite → anchor-free → P3-lite+anchor-free → IoU-aware ranking` 순서가 가장 합리적이다. 운영 속도까지 포함한 최종 후보는 `640 baseline + selective 960/1280 cascade`다. 이 구조가 현재의 640 miss와 1280 회복 결과를 가장 직접적으로 이용한다.
