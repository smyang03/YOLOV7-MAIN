# YOLOv7 튜닝 약점 개선 상세 연구 정리

작성일: 2026-08-13  
대상: YOLOv7-L / YOLOv7-W6 계열, SIAV2, CrowdHuman 기반 사람·머리·안전모 검출  
목표: YOLOv7-L 또는 S10의 추론 속도를 크게 해치지 않으면서 미감지와 작은 객체 recall 개선

> 이 문서는 지금까지 조사한 논문과 현재 프로젝트에서 수행한 실험을 연결해 정리한 연구 문서다. 논문 수치는 대부분 각 논문의 원래 benchmark 결과이며, Dongseo/CrowdHuman 성능으로 직접 해석하면 안 된다.

---

## 1. 문제 정의

현재 문제는 단순히 mAP를 높이는 것이 아니다.

1. 640 입력에서 작은 head/helmet이 미감지된다.
2. confidence를 낮춰도 일부 miss는 복구되지 않는다.
3. 960/1280 재추론에서는 일부 miss가 회복된다.
4. 전체 프레임을 항상 고해상도로 처리하면 속도가 느려진다.
5. 외부 vision encoder나 feature cosine만으로는 후보가 없는 객체를 안정적으로 생성하기 어렵다.
6. Knowledge distillation은 설계에서 제외한다.

따라서 해결해야 할 핵심은 다음 세 가지다.

```text
작은 객체 feature가 사라지는 문제
후보는 있지만 score/NMS에서 탈락하는 문제
어떤 프레임을 고해상도로 재검출해야 하는지 모르는 문제
```

---

## 2. 현재 실험에서 확인된 사실

Dongseo-food 데이터의 실제 YOLOv7 miss set을 기준으로 수행한 기존 실험:

| 방법 | 결과 | 해석 |
|---|---:|---|
| YOLOv7-L 640, conf 0.25 | 기준 miss 36개 | 기준선 |
| conf 0.01까지 하향 | 추가 복구 0/36 | 단순 threshold 문제가 아님 |
| P3 prototype bank, 640 | 4/36 회복 | 일부 유사 feature는 있으나 false positive가 큼 |
| person ROI rescue | 0/36 회복 | person crop만으로는 후보 생성 실패 miss를 못 고침 |
| 동일 모델 960 | 8/36 회복 | 입력 해상도 개선 효과 확인 |
| 동일 모델 1280 | 13/36 회복 | 작은 객체의 공간 정보 소실이 주요 원인일 가능성 |

이 결과로부터 다음을 추론할 수 있다.

- confidence calibration만으로는 해결되지 않는 miss가 존재한다.
- feature similarity는 detector를 대체하기보다 재추론 trigger에 적합하다.
- 사람 box가 이미 있어야 하는 person ROI cascade만으로는 전체 FN을 해결할 수 없다.
- P3와 선택적 고해상도 처리가 핵심 후보이다.

---

## 3. YOLO 계열의 주요 약점 분류

### 3.1 작은 객체와 downsampling

작은 객체는 backbone의 stride가 커질수록 몇 개의 grid cell에만 표현된다. 경계와 색상, 모양 정보가 사라지고 배경과 합쳐진다. 작은 객체는 다음 조건에서 특히 취약하다.

- 객체가 입력에서 차지하는 픽셀이 작음
- 사람이나 장비에 의해 가려짐
- 이미지 경계에 위치함
- 배경과 색상·질감이 비슷함
- 여러 객체가 겹쳐 있음

직접적인 개선 방법은 P3/P2 보존, 입력 해상도 증가, crop/tile, sparse high-resolution이다.

### 3.2 Feature semantic gap

고해상도 feature는 위치가 정확하지만 의미 정보가 약하고, 저해상도 feature는 의미 정보가 강하지만 위치가 거칠다. 단순한 upsample/concat은 두 feature의 semantic gap과 spatial misalignment를 충분히 해결하지 못할 수 있다.

### 3.3 Anchor와 positive assignment

anchor-based detector는 데이터의 box 크기와 aspect ratio에 영향을 받는다. 작은 객체가 적절한 anchor에 매칭되지 않으면 gradient가 약해지거나 후보가 부족해질 수 있다. Anchor-free는 anchor prior를 제거할 수 있지만, crowded scene에서 center 충돌과 assignment 설계 문제가 새로 생긴다.

### 3.4 Score와 localization quality 불일치

classification/objectness score가 box의 위치 정확도를 완전히 표현하지 못하면 다음 문제가 생긴다.

- box는 부정확하지만 confidence가 높은 후보가 살아남음
- box는 괜찮지만 confidence가 낮아 NMS 전에 밀림
- score threshold를 낮추면 false positive만 증가

### 3.5 NMS와 혼잡 장면

겹친 사람·머리·안전모에서는 한 객체의 후보가 다른 객체 후보를 제거할 수 있다. Detector가 후보를 만들었더라도 post-processing이 FN을 만들 수 있다.

### 3.6 클래스·샘플 불균형

person이 head보다 많고 쉬운 person 샘플이 다수이면 head의 gradient가 상대적으로 약해질 수 있다. 단순 Focal loss보다 head 중심 sampler와 hard positive/negative replay를 같이 봐야 한다.

### 3.7 Domain shift

학습 데이터와 실제 현장의 카메라·조명·압축·각도·작업복·안전모가 다르면 feature 분포와 confidence가 변한다. 이 문제는 구조 변경만으로 해결되지 않는다.

### 3.8 라벨 noise

객체검출 annotation noise는 missing label, extra label, class shift, inaccurate box로 분류할 수 있다. 작은 객체는 box가 몇 픽셀만 틀려도 IoU 손실이 커진다. 이번 모델 비교에서는 base와 개선 모델에 같은 라벨 조건을 적용해 공통 변수로 취급하되, 장기적으로는 별도 관리해야 한다.

---

## 4. 논문별 상세 요약

## 4.1 FPN — Feature Pyramid Networks for Object Detection

논문: [Lin et al., 2017](https://arxiv.org/pdf/1612.03144)

### 해결하려는 문제

서로 다른 크기의 객체를 하나의 feature map으로 처리하면 작은 객체와 큰 객체를 모두 안정적으로 검출하기 어렵다. 이미지 pyramid를 사용하면 정확도는 좋아지지만 계산량이 커진다.

### 핵심 방법

- backbone의 계층형 feature를 활용
- 높은 semantic 정보를 가진 저해상도 feature를 위로 upsample
- 낮은 단계의 고해상도 feature와 lateral connection으로 결합
- 각 pyramid level에서 독립적으로 prediction

### 핵심 의미

고해상도 feature에 semantic 정보를 주입하면서 multi-scale representation을 만든다. 논문은 단일 입력 해상도에서도 pyramid를 구성해 image pyramid보다 효율적으로 처리하는 방향을 제시했다.

### YOLOv7 적용

YOLOv7의 P3/P4/P5/P6는 FPN/PAN 계열의 발전된 형태로 이해할 수 있다. 현재 문제에서는 P3를 제거하거나 과도하게 축소하면 안 된다.

### 적용 우선순위

매우 높음. P3-lite 설계의 이론적 기반이다.

### 한계

FPN만으로 후보가 생성되지 않는 모든 FN을 해결하지는 못한다. 고해상도 feature를 유지하면 계산량이 증가할 수 있다.

---

## 4.2 EFPN — Extended Feature Pyramid Network

논문: [Enhancing Feature Pyramid Networks for Small Object Detection](https://arxiv.org/abs/2003.07021)

### 해결하려는 문제

기존 FPN의 단순 feature fusion은 서로 다른 깊이의 feature가 가진 semantic gap을 충분히 줄이지 못한다. 작은 객체에서는 고해상도 feature의 의미 정보가 약해지는 문제가 더 크다.

### 핵심 방법

feature pyramid 연결과 융합을 개선해 작은 객체 위치 정보와 semantic 정보를 함께 강화한다. 논문 계열의 공통 방향은 고해상도 경로에 더 강한 의미 정보를 전달하는 것이다.

### YOLOv7 적용

full EFPN을 그대로 이식하기보다 다음의 경량화가 현실적이다.

```text
P4 → P3 semantic 보강
P3 채널 수 축소
P3 전용 lightweight conv
기존 P4/P5/P6 경로 유지
```

### 적용 우선순위

높음. 단, 구조를 크게 바꾸기 전에 P3-lite를 먼저 실험한다.

### 한계

feature fusion을 복잡하게 만들면 TensorRT graph와 latency가 악화될 수 있다.

---

## 4.3 QueryDet — Cascaded Sparse Query

논문: [Yang et al., CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Yang_QueryDet_Cascaded_Sparse_Query_for_Accelerating_High-Resolution_Small_Object_Detection_CVPR_2022_paper.pdf)

### 해결하려는 문제

작은 객체를 보기 위해 전체 이미지를 고해상도로 처리하면 계산량이 크게 증가한다.

### 핵심 방법

1. 낮은 해상도 feature에서 객체가 있을 위치를 coarse하게 예측
2. 해당 위치만 sparse query로 선택
3. 선택된 위치의 고해상도 feature를 계산
4. 저해상도와 고해상도 결과를 결합

### YOLOv7 적용

처음에는 내부 feature 구조를 바꾸지 않고 외부 cascade로 검증한다.

```text
640 1차 검출
→ miss-risk 영역 선택
→ 선택 영역만 960/1280 crop 또는 tile
→ 원본 좌표로 복원
→ 결과 merge
```

### 적용 우선순위

매우 높음. 속도와 품질을 동시에 목표로 할 때 가장 직접적인 연구 방향이다.

### 한계

1차 검출에서 관심 영역을 잘못 선택하면 2차 검출도 놓친다. person ROI 방식은 person 자체가 miss된 경우를 복구하지 못하므로 tile fallback이 필요하다.

---

## 4.4 SnipeDet

논문: [SnipeDet](https://www.sciencedirect.com/science/article/abs/pii/S0167865521003858)

### 핵심 아이디어

작은 객체가 포함될 가능성이 높은 영역을 중심으로 고해상도 처리를 수행해 전체 입력의 계산 부담을 줄인다.

### YOLOv7 적용

QueryDet과 유사하게 640 detector의 후보·person 위치·위험 영역을 이용해 crop/tile을 선택하는 외부 cascade로 시작할 수 있다.

### 적용 우선순위

중상. QueryDet보다 구체적 구현을 그대로 옮기기보다는 selective crop 설계의 참고로 사용한다.

### 한계

관심 영역 proposal이 실패하면 작은 객체를 놓친다.

---

## 4.5 Trident Pyramid Networks

논문: [Trident Pyramid Networks](https://openreview.net/forum?id=327eol9Xgyi)

### 핵심 아이디어

서로 다른 dilation/receptive field를 가진 branch를 사용해 다양한 객체 크기와 문맥을 처리한다. branch 간 backbone을 공유해 계산 중복을 줄이려는 방향이다.

### YOLOv7 적용

작은 객체와 중간 객체에 서로 다른 receptive field가 필요할 때 참고할 수 있다. 그러나 multi-branch는 TensorRT latency와 메모리에 부담을 줄 수 있다.

### 적용 우선순위

중간 이하. P3-lite와 cascade 검증 이후 고려한다.

---

## 4.6 Adaptive Object Detection using Adjacency and Zoom Prediction

논문: [Lu et al., CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/html/Lu_Adaptive_Object_Detection_CVPR_2016_paper.html)

### 핵심 아이디어

객체 주변 관계와 위치를 이용해 어떤 영역을 확대할지 결정한다. 전체 이미지를 동일하게 확대하지 않고 필요한 영역에 계산을 집중한다.

### YOLOv7 적용

person-head 관계, 사람 수 대비 head 수, 상단 영역의 후보 부재를 zoom trigger로 사용할 수 있다.

### 적용 우선순위

높음. 규칙 기반 cascade trigger 설계에 적합하다.

### 한계

초기 detector의 adjacency 정보에 의존한다. person 자체가 miss된 경우에는 별도 tile trigger가 필요하다.

---

## 4.7 AdaZoom

논문: [AdaZoom](https://arxiv.org/abs/2106.10409)

### 핵심 아이디어

모델이 객체가 있을 가능성이 높은 영역을 선택적으로 확대하는 adaptive zoom 방식이다.

### YOLOv7 적용

640 결과와 low-cost risk score를 이용해 crop 위치와 확대 배율을 선택하는 구조로 연결할 수 있다.

### 적용 우선순위

중상. 학습형 zoom controller보다 먼저 규칙 기반 trigger로 개념을 검증한다.

### 한계

zoom controller 학습을 위한 데이터와 안정적인 merge가 필요하다.

---

## 4.8 VarifocalNet

논문: [VarifocalNet](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_VarifocalNet_An_IoU-Aware_Dense_Object_Detector_CVPR2021_paper.pdf)

### 해결하려는 문제

classification score가 객체 존재 여부와 localization quality를 동시에 정확히 표현하지 못한다.

### 핵심 방법

IoU-aware classification score를 학습해 높은 품질의 box가 높은 ranking score를 받도록 한다.

### YOLOv7 적용

가장 간단한 실험은 predicted IoU quality를 추가해 다음과 같이 ranking하는 것이다.

```text
final_score = objectness × class_score × iou_quality
```

### 효과가 기대되는 경우

- 후보는 존재하지만 confidence가 낮음
- box 품질이 낮은 후보가 NMS에서 살아남음
- 좋은 box가 낮은 score로 밀림

### 적용 우선순위

중상. 후보 미생성 FN은 해결하지 못하지만 NMS 전후 ranking 개선에 유용하다.

### 한계

head 구조와 학습 loss를 수정해야 하며, predicted IoU가 부정확하면 오히려 ranking이 악화될 수 있다.

---

## 4.9 Generalized Focal Loss

논문: [GFL](https://proceedings.neurips.cc/paper_files/paper/2020/file/f0bda020d2470f2e74990a07a607ebd9-Paper.pdf)

### 핵심 아이디어

- classification quality와 localization quality를 함께 학습
- box 좌표를 단일 값이 아니라 분포로 예측

작은 객체에서는 1~2픽셀의 좌표 오차도 IoU에 큰 영향을 줄 수 있기 때문에 분포형 회귀가 의미가 있다.

### YOLOv7 적용

full GFL head를 바로 이식하기보다는 IoU-aware score와 box distribution 중 하나만 단계적으로 실험한다.

### 적용 우선순위

중간. 연구 가치는 높지만 TensorRT export와 head 변경 비용이 크다.

### 한계

구조 변경 폭이 크고, 후보가 생성되지 않은 FN을 직접 복구하지 못한다.

---

## 4.10 GFLv2

논문: [GFLv2](https://arxiv.org/abs/2011.12885)

### 핵심 아이디어

box regression distribution으로부터 localization quality를 추정해 NMS ranking을 개선한다.

### YOLOv7 적용

Varifocal/GFL보다 복잡하므로 2차 연구로 둔다. 먼저 validation에서 candidate score와 실제 IoU 상관관계를 측정해야 한다.

### 적용 우선순위

중간 이하.

---

## 4.11 Wise-IoU

논문: [Wise-IoU](https://arxiv.org/abs/2301.10051)

### 핵심 아이디어

box regression 샘플의 품질에 따라 loss 집중도를 동적으로 조절한다. 쉬운 샘플과 지나치게 이상한 샘플에 gradient가 불균형하게 집중되는 것을 줄이는 목적이다.

### YOLOv7 적용

```text
CIoU baseline
→ WIoU
→ Focaler-IoU
```

한 번에 여러 loss를 섞지 않고 작은 객체 recall과 AP75를 비교한다.

### 적용 우선순위

중상. 추론 속도 영향이 적어 비교적 안전한 실험이다.

### 한계

positive assignment와 objectness 문제를 직접 해결하지 않는다.

---

## 4.12 Localization/Confidence Calibration

자료: [Localization Calibration](https://arxiv.org/abs/1811.11210), [Multiclass Confidence and Localization Calibration](https://openaccess.thecvf.com/content/CVPR2023/papers/Pathiraja_Multiclass_Confidence_and_Localization_Calibration_for_Object_Detection_CVPR2023_paper.pdf)

### 핵심 아이디어

confidence가 실제 검출 정확도와 일치하도록 보정한다. 특히 class confidence와 localization correctness가 분리되어 있다는 점을 다룬다.

### YOLOv7 적용

- camera별 threshold
- class별 threshold
- box size별 threshold
- temperature scaling
- calibration curve, ECE, precision-recall operating point 측정

### 적용 우선순위

높음. 재학습 없이 가능한 실험이다.

### 한계

calibration은 새 객체를 생성하지 않는다. detector가 후보를 전혀 만들지 못한 miss는 복구하지 못한다.

---

## 4.13 Self-Aware Object Detectors

논문: [Towards Building Self-Aware Object Detectors](https://openaccess.thecvf.com/content/CVPR2023/papers/Oksuz_Towards_Building_Self-Aware_Object_Detectors_via_Reliable_Uncertainty_Quantification_and_CVPR2023_paper.pdf)

### 핵심 아이디어

detector가 자신의 예측을 얼마나 믿을 수 있는지 uncertainty로 표현한다.

### YOLOv7 적용

저비용 uncertainty 후보:

- weak/strong augmentation 결과 차이
- score margin
- box regression variance
- P3/P4 activation instability
- 시간축 prediction inconsistency

이 값은 2차 재추론 trigger로 사용할 수 있다.

### 적용 우선순위

중상.

### 한계

ensemble이나 dropout 기반 uncertainty는 속도를 떨어뜨릴 수 있다. 처음에는 augmentation consistency부터 사용한다.

---

## 4.14 Did You Miss the Sign?

논문: [Did You Miss the Sign?](https://arxiv.org/abs/1903.06391)

### 핵심 아이디어

detector 내부 feature로 false negative 가능성을 별도 예측한다. detector를 교체하지 않고 detector가 실패했을 가능성을 판단한다.

### YOLOv7 적용

```text
YOLOv7 P3/P4 feature
+ prediction metadata
→ FN risk score
→ 위험한 ROI만 고해상도 재검출
```

### 적용 우선순위

높음.

### 한계

FN predictor 자체가 새 box를 생성하는 것은 아니다. 반드시 2차 detector 또는 tile과 결합해야 한다.

---

## 4.15 Introspective False Negative Prediction

논문: [Introspective False Negative Prediction for Object Detection](https://pmc.ncbi.nlm.nih.gov/articles/PMC8073889/)

### 핵심 아이디어

detector의 내부 상태와 prediction을 이용해 어느 영역에서 false negative가 발생할지 예측한다.

### YOLOv7 적용 입력

- P3/P4 pooled feature
- objectness/class score
- box 크기
- 사람 수와 head 수 관계
- image boundary 여부
- augmentation 간 차이

### 적용 우선순위

매우 높음. 현재 feature prompt 아이디어를 가장 실용적으로 바꾸는 방향이다.

### 한계

miss set과 정상 set을 분리한 학습 데이터가 필요하다. predictor 성능이 아니라 cascade 전체 평균 latency와 recall로 평가해야 한다.

---

## 4.16 KGFP — Knowledge-Guided Failure Prediction

논문: [KGFP](https://openaccess.thecvf.com/content/CVPR2026W/SAIAD/html/Zimmermann_Knowledge-Guided_Failure_Prediction_Detecting_When_Object_Detectors_Miss_Safety-Critical_Objects_CVPRW_2026_paper.html)

### 핵심 아이디어

안전과 관련된 객체를 detector가 놓칠 가능성을 내부 detector feature와 외부 vision representation을 활용해 판단한다.

### YOLOv7 적용

초기 버전은 외부 encoder 없이 다음만 사용한다.

```text
YOLO P3/P4 feature
person/head 관계
score margin
box 크기
temporal inconsistency
```

외부 DINO/YOLOE feature는 offline 분석 또는 낮은 빈도의 fallback으로 제한한다.

### 적용 우선순위

매우 높음.

### 한계

KGFP는 detector가 아니다. miss 위험을 판단하고, 실제 복구는 2차 고해상도 detector가 수행해야 한다.

---

## 4.17 DECIDER

논문: [DECIDER](https://arxiv.org/abs/2408.00331)

### 핵심 아이디어

모델의 prediction이 실패할 가능성을 별도로 판단하는 failure detection 계열이다.

### YOLOv7 적용

직접 detector head를 바꾸기보다 “현재 결과를 믿어도 되는가?” 판단 모듈의 설계 참고로 사용한다.

### 적용 우선순위

중간. 분류 실패 예측 중심이므로 object detection FN cascade에는 KGFP/Introspective FN이 더 직접적이다.

---

## 4.18 NMS와 End-to-End Detection

자료: [End-to-End Object Detection](https://proceedings.mlr.press/v139/sun21b.html)

### 핵심 문제

NMS는 후보를 제거하는 후처리이고, crowded scene에서는 정상적인 중복 객체까지 삭제할 수 있다. NMS는 학습과 직접 연결되지 않는 비미분 단계이므로 detector score와 post-processing 사이에 불일치가 발생할 수 있다.

### YOLOv7 적용

구조 변경 전에 다음을 평가한다.

- class-aware NMS
- class별 NMS
- NMS IoU sweep
- Soft-NMS prototype
- duplicate candidate count
- NMS 전후 recall 변화

### 적용 우선순위

높음. 재학습 없이 확인할 수 있다.

### 한계

NMS는 후보가 있을 때만 효과가 있다. 후보 자체가 없으면 아무리 좋은 NMS도 복구하지 못한다.

---

## 4.19 QueryProp와 Temporal Aggregation

자료: [QueryProp](https://arxiv.org/abs/2207.10959), [Video Object-Level Temporal Aggregation](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/2107_ECCV_2020_paper.php)

### 핵심 아이디어

영상에서는 한 프레임에서 confidence가 낮거나 잠시 miss된 객체가 앞뒤 프레임에는 검출될 수 있다. 이전/이후 프레임의 feature와 box를 이용해 일시적인 FN을 완화한다.

### YOLOv7 적용

- 짧은 temporal window
- box association/tracking
- 이전 프레임 후보 유지
- 연속 프레임 consensus
- 갑작스러운 miss에만 재추론

### 적용 우선순위

높음. 영상 입력이라면 구조를 크게 바꾸지 않고 recall을 높일 가능성이 있다.

### 한계

객체가 빠르게 움직이거나 착용 상태가 변하면 stale box가 생길 수 있다. latency와 temporal window를 제한해야 한다.

---

## 4.20 Long-Tail Object Detection

논문: [Factors in Finetuning Deep Model for Object Detection With Long-Tail Distribution](https://openaccess.thecvf.com/content_cvpr_2016/html/Ouyang_Factors_in_Finetuning_CVPR_2016_paper.html)

### 핵심 아이디어

샘플 수가 많은 클래스가 feature learning에 더 큰 영향을 준다. detection에서는 각 클래스가 서로 다른 task이므로 클래스 불균형을 별도로 관리해야 한다.

### YOLOv7 적용

- head 포함 image oversampling
- 작은 head image 중심 batch
- head miss hard positive replay
- helmet false positive hard negative replay
- class별이 아닌 size/occlusion별 recall 측정

### 적용 우선순위

높음. 구조 변경 없이 적용할 수 있다.

### 한계

단순 oversampling은 과적합과 batch 다양성 저하를 일으킬 수 있다.

---

## 4.21 Equalized Focal Loss

논문: [Equalized Focal Loss](https://mlanthology.org/cvpr/2022/li2022cvpr-equalized/)

### 핵심 아이디어

long-tail dense detection에서 클래스별 positive-negative imbalance를 고려해 focal weighting을 조정한다.

### YOLOv7 적용

head와 person의 class imbalance가 실제로 큰 경우 후보가 될 수 있다. 다만 현재 데이터의 클래스·크기 분포를 먼저 계산해야 한다.

### 적용 우선순위

중상.

### 한계

head가 작아서 생기는 공간 정보 손실은 직접 해결하지 못한다.

---

## 4.22 Noisy Annotation 연구

논문: [Towards Noise-resistant Object Detection with Noisy Annotations](https://arxiv.org/abs/2003.01285)

### 핵심 아이디어

missing label, class 오류, box 오류를 분리해 label correction과 box correction을 수행한다.

### YOLOv7 적용

- prediction-GT 불일치 sample audit
- 낮은 품질 GT의 loss weighting
- 작은 box 별도 검수
- empty image와 missing-label image 분리

### 적용 우선순위

현재 모델 구조 비교에서는 중간. 장기적으로 높음.

### 한계

이번 비교에서는 모든 모델이 동일 라벨 조건을 가지므로 모델 구조의 순수 효과와 분리해야 한다.

---

## 4.23 Robust Detection With Inaccurate Bounding Boxes

논문: [Robust Object Detection With Inaccurate Bounding Boxes](https://arxiv.org/abs/2207.09697)

### 핵심 아이디어

부정확한 box annotation에 대해 classification 신호와 object-aware instance selection을 사용해 더 안정적인 box를 학습한다.

### YOLOv7 적용

작은 head box가 지속적으로 prediction과 어긋나는 경우 audit/loss weighting 대상으로 사용한다.

### 적용 우선순위

현재는 중간.

---

## 4.24 Domain Adaptive YOLO

논문: [Domain Adaptive YOLO](https://proceedings.mlr.press/v157/zhang21c.html)

### 핵심 아이디어

source domain과 target domain의 feature/appearance 차이를 줄여 실제 환경 일반화를 개선한다.

### YOLOv7 적용

복잡한 adversarial adaptation 전에 다음을 적용한다.

- 밝기·색온도 augmentation
- blur·compression·glare augmentation
- target 영상 calibration
- camera별 threshold
- target 객체 크기 분포 분석

### 적용 우선순위

실제 카메라와 학습 데이터 차이가 큰 경우 높음.

### 한계

test-time self-training은 noisy pseudo-label과 confirmation bias가 누적될 수 있다.

---

## 4.25 Test-Time Adaptation

자료: [Test-Time Adaptive Object Detection](https://arxiv.org/abs/2209.07601), [TeST](https://openaccess.thecvf.com/content/WACV2023/papers/Sinha_TeST_Test-Time_Self-Training_Under_Distribution_Shift_WACV2023_paper.pdf)

### 핵심 아이디어

label 없는 target 영상에 맞춰 추론 시점에 모델 또는 feature statistics를 적응시킨다.

### 적용 우선순위

현재는 낮음~중간.

### 이유

운영 중 모델이 잘못된 pseudo-label을 학습하면 성능이 누적 악화될 수 있다. 고정된 현장에서는 offline calibration과 augmentation이 더 안전하다.

---

## 5. 구조 변경 후보 비교

| 후보 | 개선 대상 | 속도 영향 | 장점 | 주요 위험 | 우선순위 |
|---|---|---:|---|---|---:|
| P3-lite | 작은 객체 feature | 낮음~중간 | YOLO 구조 호환성이 높음 | 개선폭 제한 가능 | 1 |
| P2 full | 극소 객체 | 중간~큼 | recall 상한 확인 | latency/메모리 증가 | 2 |
| Anchor-free | anchor matching | 낮음~중간 | anchor prior 제거 | assignment/NMS는 별도 문제 | 3 |
| IoU-aware score | 후보 ranking | 낮음 | post-NMS 품질 개선 | 후보 미생성은 못 고침 | 4 |
| WIoU/Focaler-IoU | box regression | 거의 없음 | 학습만 변경 | recall 직접 개선 불확실 | 5 |
| Attention | 배경 억제 | 낮음~중간 | feature 선택 가능 | 근본 원인 불명확 | 후순위 |
| Deformable conv | 변형/정렬 | 중간~큼 | 가림·변형 대응 | TRT 호환성 | 후순위 |
| Transformer neck | 전역 문맥 | 중간~큼 | 문맥 강화 | 속도 충돌 | 후순위 |
| Full high-res | 작은 객체 | 큼 | 효과 확인 쉬움 | 속도 저하 | 비교군 |
| Selective cascade | 작은 객체+속도 | 평균 낮음 | 필요한 경우만 비용 증가 | trigger 누락 | 최종 후보 |

---

## 6. 최종 제안 설계

## 6.1 모델 학습 경로

```text
Baseline YOLOv7-L/S10 640
        ↓
P3-lite 640
        ↓
Anchor-free 640
        ↓
P3-lite + Anchor-free
        ↓
IoU-aware score 또는 WIoU
```

각 단계는 같은 dataset split, seed, epoch, augmentation 조건에서 독립 평가한다.

## 6.2 운영 추론 경로

```text
1차 YOLOv7-L 또는 S10 640 TRT
        ↓
저비용 miss-risk 판단
        ↓
위험 프레임/ROI만 960·1280 또는 tile
        ↓
원본 좌표 복원
        ↓
class-aware merge/NMS
```

### 초기 rule-based trigger

- person은 있는데 head/helmet이 없음
- person 대비 head 수가 camera별 정상 범위보다 낮음
- person box가 작음
- person 상단 ROI에 후보가 없음
- score margin이 낮음
- augmentation 간 prediction이 불일치
- 이전/현재 프레임 간 prediction이 불일치

person 자체 miss까지 복구해야 하는 경우에는 person ROI가 아니라 tile/full-frame fallback을 사용한다.

## 6.3 FN-risk predictor

초기에는 YOLO 내부 feature만 사용한다.

```text
P3/P4 pooled feature
+ detection metadata
+ person-head spatial relation
+ temporal inconsistency
→ risk score
→ selective high-resolution trigger
```

외부 DINO/YOLOE feature는 전체 프레임에 매번 실행하지 않는다. 필요하면 offline 분석 또는 낮은 빈도 fallback으로 제한한다.

---

## 7. 실험 순서와 성공 기준

### Phase A: 기준선

1. YOLOv7-L 640 TRT
2. S10 640 TRT
3. YOLOv7-L 960/1280 실제 latency
4. NMS IoU/threshold sweep

### Phase B: 구조

1. P3-lite
2. Anchor-free
3. P3-lite + Anchor-free
4. P2 비교군

### Phase C: 학습 목표

1. CIoU baseline
2. WIoU/Focaler-IoU
3. IoU-aware ranking
4. head-heavy sampler
5. hard positive/negative replay

### Phase D: cascade

1. fixed full-frame 1280
2. person ROI 960/1280
3. rule-based trigger
4. tile fallback
5. FN-risk predictor trigger

### 최종 선택 기준

```text
목표 = recall 증가 / 평균 latency 증가
```

반드시 함께 기록한다.

- 전체 AP/precision/recall
- class별 AP/precision/recall
- 작은 객체 recall
- occlusion bucket recall
- 실제 miss recovery 수
- 추가 false positive 수
- 평균 FPS
- P95/P99 latency
- trigger rate
- GPU memory

---

## 8. 현재 적용하지 않을 방향

### Knowledge distillation

사용자 결정에 따라 제외한다.

### 전체 프레임 DINO/YOLOE

속도 목표와 맞지 않는다. feature prompt는 risk trigger로만 사용한다.

### Attention 무제한 추가

실제 약점 진단 없이 백본만 무거워질 위험이 있다.

### YOLOv7과 YOLOv9/YOLOv10 혼합

구조·loss·assignment·export가 동시에 바뀌어 원인 분석이 어렵다. 외부 baseline으로만 비교한다.

### Test-time self-training 기본 적용

pseudo-label confirmation bias와 catastrophic forgetting 위험 때문에 후순위로 둔다.

---

## 9. 최종 판단

현재까지의 실험과 문헌을 종합하면 가장 가능성이 높은 최종 구조는 다음이다.

```text
YOLOv7-L/S10 640 TRT
+ P3 보존 또는 P3-lite
+ class/NMS/score calibration
+ 조건부 960/1280 또는 tile cascade
+ 필요 시 FN-risk predictor
```

학습 모델만 선택해야 한다면 다음 순서다.

```text
YOLOv7-L/S10 baseline
→ P3-lite
→ Anchor-free
→ P3-lite + Anchor-free
→ IoU-aware score
```

운영 모델까지 고려하면 `단일 고정 1280 모델`보다 `640 1차 + 위험 영역 선택적 고해상도 재추론`이 현재 목표에 더 적합하다. 이는 640에서 1280으로 올렸을 때 일부 실제 miss가 회복된 기존 결과를 직접 활용하면서, 모든 프레임의 비용 증가를 피할 수 있기 때문이다.

---

## 10. 참고 논문 목록

### 작은 객체·feature pyramid

- [FPN — Feature Pyramid Networks for Object Detection](https://arxiv.org/pdf/1612.03144)
- [EFPN — Extended Feature Pyramid Network](https://arxiv.org/abs/2003.07021)
- [QueryDet — Cascaded Sparse Query](https://openaccess.thecvf.com/content/CVPR2022/papers/Yang_QueryDet_Cascaded_Sparse_Query_for_Accelerating_High-Resolution_Small_Object_Detection_CVPR_2022_paper.pdf)
- [SnipeDet](https://www.sciencedirect.com/science/article/abs/pii/S0167865521003858)
- [Trident Pyramid Networks](https://openreview.net/forum?id=327eol9Xgyi)
- [Adaptive Object Detection using Adjacency and Zoom Prediction](https://openaccess.thecvf.com/content_cvpr_2016/html/Lu_Adaptive_Object_Detection_CVPR_2016_paper.html)
- [AdaZoom](https://arxiv.org/abs/2106.10409)

### score·loss·calibration

- [VarifocalNet](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_VarifocalNet_An_IoU-Aware_Dense_Object_Detector_CVPR2021_paper.pdf)
- [Generalized Focal Loss](https://proceedings.neurips.cc/paper_files/paper/2020/file/f0bda020d2470f2e74990a07a607ebd9-Paper.pdf)
- [GFLv2](https://arxiv.org/abs/2011.12885)
- [Wise-IoU](https://arxiv.org/abs/2301.10051)
- [Localization Calibration](https://arxiv.org/abs/1811.11210)
- [Multiclass Confidence and Localization Calibration](https://openaccess.thecvf.com/content/CVPR2023/papers/Pathiraja_Multiclass_Confidence_and_Localization_Calibration_for_Object_Detection_CVPR2023_paper.pdf)
- [Self-Aware Object Detectors](https://openaccess.thecvf.com/content/CVPR2023/papers/Oksuz_Towards_Building_Self-Aware_Object_Detectors_via_Reliable_Uncertainty_Quantification_and_CVPR2023_paper.pdf)

### failure prediction

- [Did You Miss the Sign?](https://arxiv.org/abs/1903.06391)
- [Introspective False Negative Prediction](https://pmc.ncbi.nlm.nih.gov/articles/PMC8073889/)
- [KGFP](https://openaccess.thecvf.com/content/CVPR2026W/SAIAD/html/Zimmermann_Knowledge-Guided_Failure_Prediction_Detecting_When_Object_Detectors_Miss_Safety-Critical_Objects_CVPRW_2026_paper.html)
- [DECIDER](https://arxiv.org/abs/2408.00331)

### 영상·NMS·데이터·domain

- [End-to-End Object Detection](https://proceedings.mlr.press/v139/sun21b.html)
- [QueryProp](https://arxiv.org/abs/2207.10959)
- [Video Object-Level Temporal Aggregation](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/2107_ECCV_2020_paper.php)
- [Long-Tail Object Detection](https://openaccess.thecvf.com/content_cvpr_2016/html/Ouyang_Factors_in_Finetuning_CVPR_2016_paper.html)
- [Equalized Focal Loss](https://mlanthology.org/cvpr/2022/li2022cvpr-equalized/)
- [Noise-resistant Object Detection](https://arxiv.org/abs/2003.01285)
- [Robust Detection With Inaccurate Bounding Boxes](https://arxiv.org/abs/2207.09697)
- [Domain Adaptive YOLO](https://proceedings.mlr.press/v157/zhang21c.html)
- [Test-Time Adaptive Object Detection](https://arxiv.org/abs/2209.07601)

### survey

- [YOLO-Based Small Object Detection Survey](https://www.sciencedirect.com/org/science/article/pii/S1546221826001943)
- [Small Object Detection Comprehensive Survey](https://www.sciencedirect.com/science/article/pii/S2667305325000870)
