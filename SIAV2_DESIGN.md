# siav2 검출 모델 개발 설계문서 — YOLOv7-W6 경량화 @1280

> **문서 지위:** 이 문서 **하나가** 개발·검증의 기준(source of truth). 무엇을·어떻게·언제 하는지 전부 여기에 있다. 이후 모든 실험/검증은 이 문서의 목표치·방안·게이트로 판정한다.
> **작성 근거:** 실제 레포 코드(`cfg/`, `models/yolo.py`, `models/common.py`, `utils/loss.py`, `train_aux.py`, `data/hyp.scratch.p6.yaml`) 확인 + v8~v26 검증 기법 검토 + 6인 라운드테이블 2회.
> **라이선스:** 의사결정에서 **제외**(사용자 지시) → 최신 버전 채택은 라이선스 제약 없이 순수 측정값으로만 판단.
> **환경 주의:** 현 개발환경(Windows)은 torch cuDNN 페이징 오류로 실행 불가 → 모든 학습/실측/빌드검증은 **사용자 GPU 머신**에서. cfg는 w6 구조 복사본이라 빌드 안전성은 w6와 동일.

---

## 0. 진행 현황 체크리스트 (한 것 / 안 한 것 / 해야 할 것)

> 상태의 단일 출처. 작업할 때마다 여기부터 갱신한다.

### ✅ 한 것 (이번 설계 작업)
- [x] 본 기준 문서 작성 (목표·아키텍처·레버·스케줄·방안·검증·리스크·채택규칙)
- [x] `cfg/training/yolov7-l6-siav2.yaml` (기본 P3~P6, width 0.75) — YAML 파싱 OK
- [x] `tools/make_random_weights.py` (Phase 0 랜덤 weight + `--width` 스윕) — `py_compile` 통과
- [x] Phase −1~5 스케줄 + 각 단계 방안(명령/hyp/판정) 상세화
- [x] 검토 반영: Phase −1(W6 baseline), Phase 0 공정성 규정, close-mosaic/copy_paste 정정
- [x] 사용자 확정값 반영: 속도 2배 / nc 16(cfg) / 운영지표 나중 / GPU(A4000·A6000×8) / anchor-free 이식+토글

### ⏸️ 의도적으로 안 한 것 (지금 하면 안 됨 — 이유)
- [ ] head 변형 `-s`(P2~P5)/`-l`(P3~P5) cfg — P2/P6 라우팅 재구성 → **빌드검증 필수인데 현 환경 torch 미로드.** Phase 1(EDA 후) 생성.
- [ ] close-mosaic 코드 패치 — yolov7 native 아님(§4 L2). Phase 3 필요 시 소량 구현.
- [ ] W6-siav2 baseline 학습 / 실제 latency·mAP 실측 — **검증은 이 작업 이후**(사용자 지시).

### ⬜ 해야 할 것 (다음 착수 순서)
- [ ] **[먼저]** cfg 빌드 확인: `python models/yolo.py --cfg cfg/training/yolov7-l6-siav2.yaml` (A4000)
- [ ] **[남은 입력]** `data/siav2.yaml` 경로·val split (데이터 준비 시) / 운영지표 SLA(추후)
- [ ] Phase −1 → 0 → 1 → 2 → 3 →(필요시)4 → 5 (아래 §5)

---

## 1. 목적 & 배경 & 핵심 결정

**문제의식:** siav2는 근거리+원거리 전체를 커버하는 16채널 단일 검출 모델 필요. 640은 소형·원거리에 한계, 1280이 맞으나 W6-1280(~360 GFLOPs)은 무겁다.

**해법 가설:** yolov7-w6(1280, ReOrg stem, P3~P6)를 **경량화**해 W6에 근접한 정확도를 훨씬 빠른 속도로 낸다.

**핵심 통찰(3-레버가 하나로 엮임):** 경량화(L1)는 속도를 벌지만 mAP를 까먹는다 → 그 갭을 검증된 **학습기법(L2)** 으로 되사온다 → 그래도 부족하면 **anchor-free(L3)** 를 조건부 투입.

**확정된 핵심 결정:**
1. v7에 최신 기능(PGI/anchor-free/어텐션) 손이식 = 최신 버전을 열등하게 재구현 → **원칙적으로 안 함.** v7 정당한 커스터마이징은 **cfg/학습기법 레벨까지.**
2. v7은 이미 **SimOTA**(`ComputeLossAuxOTA`, `loss_ota:1`)를 씀 → v7→v11 P차이는 "할당 없음→있음"이 아니라 **SimOTA→TaskAligned**. anchor-free 기대이득은 생각보다 좁다.
3. 라이선스 제외 후, **v7 잔류의 유일한 합법적 근거 = "동일 latency에서 최신보다 빠르거나 정확함"을 실측 증명.**

---

## 2. 최종 목표 (정량 스펙)

| 항목 | 목표치 | 판정 |
|---|---|---|
| **정확도 하락** | W6 대비 mAP **< 2%** | 동결 val셋, 동일 스크립트 |
| **속도 향상** | W6 대비 latency **2배**(= 지연 반토막) **확정** | RTX 4090 TRT **FP16**, 1280, batch·warmup 고정 |
| **소형 검출** | AP_small 하락 **< 2%** (1급 지표) | COCO AP_small 별도 |
| **운영 지표** | ⏸️ **나중**(현장 SLA 미정) — 지금은 **모델 지표(mAP/AP_small)만** | 추후 정의 |

> ⚠️ 속도 목표 = **2배 확정**(latency 반토막). 트리거 A/게이트 판정선도 "W6 대비 2배".
> ⚠️ 클래스 수 **nc=16 확정** (siav2). cfg 반영 완료.
> ⚠️ 지표 원칙 = GFLOPs 아님, **4090 TRT FP16 latency**가 유일한 속도 진실(근거: v26-L이 낮은 GFLOPs에도 30% 느렸음).

---

## 3. 아키텍처 설계

### 3.1 베이스
yolov7-w6 (1280, `ReOrg` space-to-depth stem, Conv-s2 다운샘플, P3~P6, `IAuxDetect` + `ComputeLossAuxOTA`). ReOrg가 1280의 순수 해상도발 4배 연산을 실질 ~2배로 완화.

### 3.2 경량화 방식 (모델 코드 수정 0줄 — 의도)
- `width_multiple`이 모든 Conv 채널을 `make_divisible(c2*gw, 8)`로 스케일(ReOrg 무영향) → **한 줄로 폭 조절, `models/*` 안 건드림.**
- config: `cfg/training/yolov7-l6-siav2.yaml` (= w6 구조 + `width_multiple: 0.75`).
- width 스윕은 파일 복제 없이 런타임 override: `tools/make_random_weights.py --width 0.5`.

### 3.3 Head 변형 (배치별 선택, 백본 공유)
| 변형 | Head | 용도 | 상대 연산 | 상태 |
|---|---|---|---|---|
| `-s` | P2~P5 | 소형·원거리 특화 | 높음(P2 320²) | Phase 1 생성 |
| 기본 | P3~P6 | 전체 커버 | 중 | **존재** |
| `-l` | P3~P5 | 대형 드묾, 경량 | 낮음 | Phase 1 생성 |

→ 어느 변형이 필요한지는 Phase 1 bbox 분포 EDA로 확정. **변형 생성 시 라우팅 인덱스 재구성 → 즉시 `python models/yolo.py --cfg`로 빌드검증.**

---

## 4. 3-레버 상세 방안

### L1 — 경량화 (속도 확보 / cfg만)
- **width 스윕:** 0.5 / 0.625 / 0.75. 낮을수록 빠르고 mAP↓.
- **head 축소/확장:** P6 제거(경량) 또는 P2 추가(소형). §3.3.
- **anchor 재계산:** siav2 분포에 kmeans. yolov7는 학습 시작 시 autoanchor 자동 실행(끄려면 `--noautoanchor`), 또는 `utils/autoanchor.py`의 `kmean_anchors` 수동 호출.

### L2 — mAP 되사기 (학습기법 / 대부분 hyp만, 재구현 아님)
`data/hyp.scratch.p6.yaml` 기준 실제 필드/플래그:

| 기법 | 조작 | 현재값 | 방안 | 비용 |
|---|---|---|---|---|
| MixUp | hyp `mixup` | 0.15 | 0.15→0.2 상향(소형 도움) | hyp만 |
| **paste_in** (bbox copy-paste) | hyp `paste_in` | 0.15 | 상향(소형·원거리 핵심) | hyp만 |
| copy_paste (segment) | hyp `copy_paste` | 0.0 | **segmentation 마스크 있을 때만.** bbox-only면 `paste_in` 사용 | hyp만 |
| scale jitter | hyp `scale` | 0.9 | 유지/조정 | hyp만 |
| multi-scale | `--multi-scale` 플래그 | off | on(±50%). **단 1280에선 메모리 큼** → batch 조정 | 플래그 |
| 긴 스케줄 | `--epochs` | 300 | 300→ (수렴 볼 때까지) | 플래그 |
| SimOTA | hyp `loss_ota` | 1 | 유지(이미 dynamic) | — |
| **close-mosaic** | ⚠️ yolov7 **native 없음** | — | 필요 시 dataloader에 "마지막 N epoch mosaic off" **소량 코드** 추가(선택) | 소량 코드 |

> **정정:** close-mosaic는 v8 기능이지 yolov7 기본 제공이 아님. copy_paste는 마스크 필요. → L2의 "거의 공짜"는 mixup·paste_in·scale·multi-scale·긴 스케줄. close-mosaic만 소량 코드.

### L3 — anchor-free (조건부 스트레치 / 아키텍처, 큼) — 경로 (a) 이식 확정
- **발동:** L1+L2로 2% 못 지킬 때만.
- **경로 확정:** (a) **v7-light 백본에 anchor-free decoupled head 직접 이식.** (v9 채택 경로 b는 폐기 — 사용자 지시.)
- **⭐ 이식/미이식 토글 구성(필수):** 동일 백본에서 head만 바꿔 **A/B 비교 가능하게** 만든다. yolov7은 head가 cfg 마지막 모듈로 결정되므로:
  - **미이식(baseline):** `yolov7-l6-siav2.yaml` — `IAuxDetect`(anchor-based) + `ComputeLossAuxOTA` (기존).
  - **이식(anchor-free):** `yolov7-l6-siav2-af.yaml` — 백본 섹션 동일, **마지막 head만 새 anchor-free decoupled 모듈**(예: `models/yolo.py`에 신규 클래스 + `utils/loss.py`에 대응 loss). 스위칭 = cfg 교체 한 줄.
  - → 같은 데이터·같은 백본으로 **anchor-free 켠 것 vs 끈 것**을 동일 조건에서 학습·비교. 이식 이득을 순수 분리 측정.
- **비용/가드:** 신규 head 모듈 + target-builder + loss + NMS decode 필요(= 부분 v9 재구현). **2주 타임박스 + 정량 롤백 기준.**

---

## 5. 단계별 스케줄 & 방안 (진입/종료/실패 포함)

> 기간은 4090급 1~2대 가정, GPU 가용량 따라 가변.

### Phase −1 — W6-siav2 baseline 확보 (필수 사전조건) · 3~5일
- **목적:** 모든 "W6 대비" 판정의 기준값. 없으면 Phase 2/3 종료조건 판정 불가.
- **방안:** 기존 W6-siav2 체크포인트 있으면 지표 재측정만. 없으면 학습:
  ```bash
  python train_aux.py --cfg cfg/training/yolov7-w6.yaml --img 1280 1280 \
    --data data/siav2.yaml --weights '' --hyp data/hyp.scratch.p6.yaml \
    --epochs 300 --name w6-siav2-base
  ```
  이어서 mAP/AP_small(§6) + 4090 TRT FP16 latency(§5 Phase0 방식) 측정 → **기준값 동결.**
- **GPU:** 실학습 → **A6000 8개**.
- **종료:** `W6-siav2` 체크포인트 + 기준 지표표 존재.

### Phase 0 — 속도 게이트 G0 (학습 없음) · 2~3일
- **목적:** latency는 weight 무관 → 학습 전에 속도 목표 도달 여부 판정.
- **방안(코드=`tools/make_random_weights.py`):**
  ```bash
  # ① 랜덤 weight (width 0.5 / 0.75)
  python tools/make_random_weights.py --cfg cfg/training/yolov7-l6-siav2.yaml --width 0.5  --nc <nc>
  python tools/make_random_weights.py --cfg cfg/training/yolov7-l6-siav2.yaml --width 0.75 --nc <nc>
  # ② ONNX (전 모델 동일 opset·end2end 통일)
  python export.py --weights yolov7-l6-siav2-w500-random.pt --img-size 1280 1280 --grid --end2end --simplify
  # ③ TRT FP16 latency (전 모델 동일 flag·batch·warmup)
  trtexec --onnx=yolov7-l6-siav2-w500-random.onnx --fp16 --shapes=images:1x3x1280x1280
  # 비교군: W6(기존 cfg 동일 파이프라인), v9·v11(각 레포 동일 조건)
  ```
- **⚠️ 공정성 규정(필수):** 다른 레포 비교 시 **동일 opset, 동일 NMS 처리(전 모델 end-to-end 포함 또는 전 모델 제외로 통일), 동일 trtexec 플래그·batch·warmup·TRT 버전.** v26 논란이 후처리 포함 latency였으므로 **end-to-end 측정 기본.** 못 맞추면 트리거 A 판정 보류.
- **종료(GO):** v7-l6이 W6 대비 **2배**(지연 반토막) AND 동일 latency대 v9/v11보다 안 느림.
- **GPU:** 이 Phase는 **A4000 1개**로 충분(export/프로파일만, 학습 없음).
- **실패(NO-GO):** **트리거 A**(§7) → v7 중단, 최신 채택 검토.

### Phase 1 — 데이터·평가 준비 (Phase 0과 병렬 가능) · 3~5일
- **① bbox 분포 EDA:** siav2 라벨(`*.txt`의 w·h, 정규화)에서 **픽셀 크기 히스토그램** 산출. 판정: 32px 미만 비율 높음→P2(`-s`) 필요, 대형(화면 큰 비중) 거의 없음→P6 제거(`-l`) 가능. AP_small 대상 비율 확정.
- **② 평가 프로토콜 동결:** val split 고정, `test.py`로 mAP·AP_small(pycocotools, `--save-json` COCO eval) 계산법 고정. **이후 변경 금지.** (운영지표(미탐율@오경보율)는 현장 SLA 미정 → **나중**. 지금은 모델 지표만.)
- **③ anchor:** autoanchor 자동 or `kmean_anchors` 수동으로 siav2 anchor 확정.
- **④ head 변형 cfg 생성**(EDA 결과 따라): `-s`/`-l` 파생 → **즉시 `python models/yolo.py --cfg`로 빌드검증.**
- **종료:** head 변형 확정+빌드검증 / 평가셋·스크립트 동결 / siav2 anchor 확정.

### Phase 2 — 경량 baseline & 폭/head 스윕 학습 · 1~2주
- **목적:** mAP-latency **knee**(속도 목표 만족하며 mAP 갭 최소인 width) 확정.
- **방안:** width **0.5부터** baseline 학습 → mAP 확인 → 필요 시 0.625/0.75. 확정 head 변형 사용. GPU 부족 시 0.5 단일 우선, 조기종료로 가지치기.
  ```bash
  python train_aux.py --cfg cfg/training/yolov7-l6-siav2.yaml --img 1280 1280 \
    --data data/siav2.yaml --weights '' --hyp data/hyp.scratch.p6.yaml --epochs 300 --name l6-w050
  # width override 방식은 별도 cfg 사본 또는 동일 원리(로드시 width_multiple 덮기)로 처리
  ```
- **GPU:** 실학습 단계 → **A6000 8개**(width/head 병렬 스윕에 활용).
- **종료:** 속도 목표(2배) 만족 width에서 mAP 갭(vs W6) 측정됨 = L2가 되사올 양.

### Phase 3 — mAP 되사기 (L2) · 1~2주
- **목적:** Phase 2 갭을 학습기법으로 2% 이내로 되사기.
- **방안:** §4 L2 총동원. `hyp.scratch.p6.yaml` 사본 만들어 `mixup`·`paste_in`↑, `--multi-scale`, `--epochs`↑. 각 기법 ablation으로 기여도 측정. close-mosaic 필요 시 소량 코드.
- **종료(성공):** mAP·AP_small 하락 **< 2%** → **목표 충족, L3 불필요, Phase 5로.**
- **실패:** 2% 못 지킴 → **트리거 B**(§7) → Phase 4.

### Phase 4 — anchor-free 이식 (조건부, 타임박스) · ≤2~3주
- **발동:** Phase 3이 2% 미달일 때만.
- **방안(§4 L3, 경로 a 확정):**
  1. anchor-free decoupled head 모듈 신규 작성(`models/yolo.py`) + 대응 loss(`utils/loss.py`) + 추론 decode.
  2. `yolov7-l6-siav2-af.yaml` 생성 = 기존 백본 섹션 + head만 교체. **즉시 `python models/yolo.py --cfg`로 빌드검증.**
  3. **A/B 학습:** `yolov7-l6-siav2.yaml`(미이식) vs `-af.yaml`(이식)을 **동일 데이터·백본·hyp**로 학습 → mAP·AP_small·latency 비교. anchor-free 순수 이득 분리 측정.
- **GPU:** A6000 8개. **착수 전 정량 롤백 기준 확정**(예: latency 악화 >X% or mAP 이득 <Y → 폐기). **2주 타임박스.**
- **종료:** 이식본이 2% 달성 or 롤백 기준 초과 → 트리거 C/최신 채택.

### Phase 5 — 최종 검증 & 채택 결정 · 3~5일
- **방안:** 최종 후보를 **동일 latency 예산의 최신(v9/v11/v26)과 정면 비교**(mAP@same-latency). INT8 검토(후순위, QAT는 별도 트랙). 운영지표 최종 측정.
- **종료:** §7 규칙으로 **v7-light 확정 vs 최신 채택** 판정 + 배포 대상별 head 변형 확정.

### 의존성
```
Phase −1 (W6 baseline) ── 모든 "W6 대비" 기준값
Phase 0 (속도게이트) ─┐
                      ├─> Phase 2 ─> Phase 3 ─(실패)─> Phase 4 ─> Phase 5
Phase 1 (데이터/평가)─┘               └─(성공)──────────────────> Phase 5
```
> **가장 먼저:** `python models/yolo.py --cfg cfg/training/yolov7-l6-siav2.yaml` (유일한 미검증 가정) → 이후 −1/0 착수.

---

## 6. 검증 프로토콜

- **속도:** RTX 4090, TensorRT **FP16**, 1280², **batch·warmup·TRT버전 고정**, 동일 스크립트. INT8 후순위(Phase 5).
- **정확도:** Phase 1 동결 val split + `test.py`(pycocotools COCO eval, `--save-json`). **mAP + AP_small(1급)** 항상 함께. 운영지표(미탐율@오경보율)는 SLA 확정 후 추가(현재 모델 지표만).
- **공정성:** 전 후보(W6/v7-light/최신)를 동일 TRT 조건·동일 val셋. 최종 비교축 = "동일 latency에서 mAP".
- **재현성:** 실험 = {cfg, hyp, seed, weight, latency 로그, mAP 로그} 한 세트 저장.

---

## 7. 버전 채택 결정 규칙 (라이선스 제외)

**하나라도 켜지면 v7 경량화 접고 최신(v9/v11/v26) 채택.**

| 트리거 | 조건 | 시점 |
|---|---|---|
| **A 속도게이트 실패** | v7-l6이 W6 대비 <2배거나 동일 latency대 최신보다 안 빠름 | Phase 0 |
| **B 되사기 실패** | L2 총동원 후에도 mAP 하락 ≥2%, anchor-free로도 미해결 | Phase 3~4 |
| **C 정면비교 패배** | 동일 latency 예산 최신이 v7-light보다 mAP 높음 | Phase 5 |

**비용 비대칭:** v7 잔류=지속 커스텀+upstream 못 받음. 최신 채택=1회 이전 후 무료 상속. → **비기면 최신 유리. v7은 측정값에서 "이겨야" 정당화.**

---

## 8. 리스크 & 대응

| # | 리스크 | 신호 | 대응 |
|---|---|---|---|
| R1 | 경량화 속도 미달 | Phase0 <2배 | width↓ or head 축소(P6 제거) → 안되면 트리거 A |
| R2 | 2배 위해 폭 과축소로 mAP 붕괴 | Phase2 급락 | L2/L3로 되사기 시도 → 2%·2배 동시 불가 확정 시 목표 재검토/트리거 |
| R3 | L2가 갭 못 메움 | Phase3 ≥2% | mixup/paste_in·스케줄 재튜닝 → 안되면 Phase4 |
| R4 | anchor-free=열등한 v9 | Phase4 ROI 음수 | 타임박스·롤백 즉시 중단 → v9 채택 |
| R5 | GFLOPs↓인데 실측 느림 | TRT 이상 | ONNX 그래프·RepConv fuse·특수연산자 점검 |
| R6 | bbox 분포 미확인 head 오선택 | AP_small 저조 | Phase1 EDA를 게이트로 강제 |
| R7 | 평가기준 흔들림 | 비교 무의미 | Phase1 동결 엄수, 변경 시 전체 재측정 |
| R8 | INT8 정확도 급락 | Phase5 | FP16 배포, INT8은 QAT 별도 트랙 |
| R9 | GPU 부족 지연 | 일정 초과 | width 0.5 단일 우선, 조기종료 |
| R10 | copy_paste 무효(마스크 없음) | aug 효과 0 | `paste_in`으로 대체 |

---

## 9. 산출물

**존재:** `SIAV2_DESIGN.md`, `cfg/training/yolov7-l6-siav2.yaml`(nc=16), `tools/make_random_weights.py`. (모델 코드 미수정 — 의도)
**예정:** W6-siav2 baseline(−1), head 변형 `-s`/`-l`(1), latency 비교표(0), EDA 리포트·동결 평가 스크립트·anchor(1), 실험별 {cfg,hyp,로그,weight}(2~4), **anchor-free head 모듈 + loss + `yolov7-l6-siav2-af.yaml`(이식/미이식 토글, 조건부 4)**, 최종 채택 판정서·배포 weight(5).

---

## 10. 확정값 / 남은 미확정

**✅ 확정(사용자 답변):**
1. **속도목표 = 2배**(latency 반토막) 확정.
2. **nc = 16** 확정 (cfg 반영 완료).
3. **운영지표 = 나중**(현장 SLA 미정) → 현재 **모델 지표(mAP/AP_small)만**.
4. **GPU 예산** — 학습 전(Phase 0/준비) = **A4000 1개**, 실학습(Phase −1/2/3/4) = **A6000 8개**.
5. **anchor-free 경로 = (a) v7 이식** 확정. + **이식/미이식 토글 가능하게 구성**(§4 L3, §5 Phase 4).

**⬜ 남은 미확정:**
- `data/siav2.yaml` 경로 및 val split (데이터 준비 시 확정)
- 운영지표 SLA 임계값 (현장 확정 후 §6에 추가)

---

## 부록 A — 명령 레퍼런스
```bash
# cfg 빌드/파라미터/GFLOPs 확인 (가장 먼저)
python models/yolo.py --cfg cfg/training/yolov7-l6-siav2.yaml
# Phase 0 랜덤 weight → export → TRT
python tools/make_random_weights.py --cfg cfg/training/yolov7-l6-siav2.yaml --width 0.5 --nc <nc>
python export.py --weights yolov7-l6-siav2-w500-random.pt --img-size 1280 1280 --grid --end2end --simplify
trtexec --onnx=yolov7-l6-siav2-w500-random.onnx --fp16 --shapes=images:1x3x1280x1280
# Phase −1/2/3 학습 (P6 계열 → train_aux.py, P6 hyp)
python train_aux.py --cfg <cfg> --img 1280 1280 --data data/siav2.yaml \
  --weights '' --hyp data/hyp.scratch.p6.yaml --epochs 300 --name <run>
# 평가 (mAP + AP_small)
python test.py --data data/siav2.yaml --img 1280 --weights <ckpt>.pt --save-json
```

## 부록 B — 파일 맵
| 파일 | 역할 | Phase |
|---|---|---|
| `SIAV2_DESIGN.md` | 기준 문서 | 전체 |
| `cfg/training/yolov7-l6-siav2.yaml` | 경량 W6 기본(P3~P6, w0.75) | 0~3 |
| `tools/make_random_weights.py` | 랜덤 weight 생성(+width) | 0 |
| `data/hyp.scratch.p6.yaml` | P6 hyp(mixup/paste_in/scale) 기준 | −1,2,3 |
| `train_aux.py` | P6 계열 학습(aux head) | −1,2,3 |
| `export.py` / `test.py` | ONNX export / 평가 | 0,5 |

---

**한 문장:** *W6를 폭·head로 깎아 4090 TRT에서 **2배** 빠르게 만들고(Phase 0에서 학습 없이 먼저 검증, A4000), 잃은 mAP를 실제 hyp 기반 학습기법으로 2% 이내로 되산다(Phase 3, A6000×8). 부족하면 조건부로 anchor-free를 **이식/미이식 토글**로 A/B 이식(Phase 4). 어느 시점이든 트리거 A/B/C가 켜지면 최신 버전 채택.*
