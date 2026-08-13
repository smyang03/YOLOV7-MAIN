# siav2 작업지시서 (다른 PC의 AI 에이전트용) — 학습 직전까지 완료

> **당신(에이전트)에게:** 이 문서는 작업 지시서다. **먼저 `SIAV2_DESIGN.md`를 처음부터 끝까지 읽어라.** 그 문서가 설계·목표·방안의 기준이고, 이 문서는 "이번에 어디까지 하고 어디서 멈추는지"를 지정한다.
> **핵심 경계:** **학습(train) 직전까지의 모든 작업과 검증을 완료**하고 멈춘다. **어떤 `train_aux.py`/`train.py`도 실행하지 마라.**
> **환경:** 이 PC는 torch/CUDA/TensorRT가 정상 동작하는 GPU 머신이라고 가정(원 설계 PC는 torch 미로드로 실행검증을 못 했음 → 그 미검증분을 당신이 실제로 검증하는 것이 이번 임무의 핵심).

---

## 0. 임무 요약 (Definition of Done)

아래가 **전부** 끝나면 완료:
1. 환경 sanity 확인 (torch 로드, GPU 인식, trtexec 사용 가능)
2. cfg 빌드 검증 (`yolov7-l6-siav2.yaml`, nc=16)
3. **Phase 0 속도 게이트**: 랜덤 weight로 latency 측정 → **트리거 A 판정** (학습 불필요)
4. **Phase 1 준비**: bbox EDA → head 변형 확정·생성·빌드검증, anchor 재계산, 평가 프로토콜 동결 (학습 불필요)
5. `SIAV2_DESIGN.md` §0 체크리스트 갱신 + 결과 리포트 작성
6. **STOP** — 학습은 시작하지 않는다. 결과를 사람에게 보고하고 승인 대기.

> 확정 파라미터(설계문서 §2/§10): **입력 1280, nc=16, 속도목표 = W6 대비 2배(FP16), 정확도 하락 <2%, anchor-free는 조건부(지금 범위 아님).** GPU: 이 준비단계는 **A4000 1개**로 충분.

---

## 1. 작업 범위 (포함/제외)

**✅ 포함 (이번에 한다):**
- 환경 검증, cfg 빌드 검증
- Phase 0 전체 (랜덤 weight → ONNX → TRT latency, 비교표, 트리거 A 판정)
- Phase 1 전체 (EDA, head 변형 cfg 생성+빌드검증, anchor kmeans, 평가 프로토콜 동결)

**⛔ 제외 (절대 하지 마라 — 학습이므로):**
- Phase −1 W6-siav2 **학습**
- Phase 2/3/4 **학습**
- anchor-free head 이식 (Phase 4, 조건부 — 지금 범위 아님)
- INT8 (후순위)

> W6의 **latency**는 랜덤 weight로 측정 가능하므로 Phase 0 비교군에 포함(학습 아님). W6의 **mAP**는 학습이 필요하므로 제외.

---

## 2. 단계별 작업 지시 (순서대로, 각 완료조건 포함)

### Task 0 — 환경 sanity
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
which trtexec   # 또는 trtexec --version
```
- **완료조건:** torch가 CUDA=True로 로드, trtexec 실행 가능. 안 되면 여기서 멈추고 환경 문제 보고.

### Task 1 — cfg 빌드 검증 (설계문서 "가장 먼저" 항목)
```bash
python models/yolo.py --cfg cfg/training/yolov7-l6-siav2.yaml
```
- **확인:** 에러 없이 빌드, 로그 마지막 줄의 layers/parameters/GFLOPs 기록. nc=16 반영 확인.
- **완료조건:** 빌드 성공 + 파라미터/GFLOPs 수치 확보. 실패 시 에러 전문 보고(설계문서상 유일한 미검증 가정이므로 중요).

### Task 2 — Phase 0 속도 게이트 (설계문서 §5 Phase 0)
목표: **v7-l6이 W6 대비 2배 빠른가?** 를 학습 없이 판정.
```bash
# ① 랜덤 weight (width 스윕: 0.5 / 0.625 / 0.75)
python tools/make_random_weights.py --cfg cfg/training/yolov7-l6-siav2.yaml --width 0.5   --nc 16
python tools/make_random_weights.py --cfg cfg/training/yolov7-l6-siav2.yaml --width 0.625 --nc 16
python tools/make_random_weights.py --cfg cfg/training/yolov7-l6-siav2.yaml --width 0.75  --nc 16
# W6 비교군(기존 cfg, 랜덤 weight)
python tools/make_random_weights.py --cfg cfg/training/yolov7-w6.yaml --nc 16 --out w6-siav2-random.pt

# ② ONNX (전부 동일 opset·동일 옵션으로 통일)
python export.py --weights yolov7-l6-siav2-w500-random.pt --img-size 1280 1280 --grid --end2end --simplify
# (w625/w750/w6 도 동일하게 반복)

# ③ TRT FP16 latency (전부 동일 flag·batch·warmup)
trtexec --onnx=yolov7-l6-siav2-w500-random.onnx --fp16 --shapes=images:1x3x1280x1280 --warmUp=500 --iterations=1000
# (나머지 onnx 동일 옵션으로 반복)
```
- **⚠️ 공정성 규정 (설계문서 §5 Phase 0):** 모든 모델을 **동일 opset·동일 NMS 처리(end-to-end 포함으로 통일)·동일 trtexec flag/batch/warmup/TRT버전**으로. 이 규정 못 지키면 판정 보류하고 보고.
- **v9/v11 비교(선택):** 각 레포가 이 PC에 있으면 동일 조건으로 측정해 표에 추가. 없으면 "미측정"으로 표기(트리거 A의 2배 판정은 W6 기준만으로도 가능).
- **완료조건 = latency 비교표 + 트리거 A 판정:**
  - v7-l6(어느 width)이 **W6 대비 ≥2배** → **GO**. 그 width를 Phase 1/학습 후보로.
  - 어떤 width도 2배 미달 → **NO-GO = 트리거 A**. 설계문서 §7대로 "v7 중단, 최신 채택 검토" 사유를 보고(임의로 진행하지 말 것).

### Task 3 — Phase 1 준비 (설계문서 §5 Phase 1)
> siav2 데이터셋이 이 PC에 있어야 함. `data/siav2.yaml`(train/val 경로, nc=16, names) 없으면 먼저 만들 것.

1. **bbox 크기 분포 EDA**
   - siav2 라벨(`labels/**/*.txt`의 정규화 w,h)에서 **픽셀 단위 bbox 크기 히스토그램** 산출(1280 기준). 32px 미만(소형) 비율, 대형 비율 리포트.
   - **판정:** 32px 미만 다수 → `-s`(P2~P5) 필요 / 대형 거의 없음 → `-l`(P3~P5) 가능 / 애매하면 기본 P3~P6 유지.
2. **head 변형 cfg 생성**(EDA 결과에 따라 필요한 것만)
   - `-s` = P2~P5 (P2 stride4 추가), `-l` = P3~P5 (P6 제거). 백본 섹션은 기본과 공유하되 **head 라우팅 인덱스/Concat/IAuxDetect 입력을 정확히 재구성**.
   - **생성 즉시 반드시 빌드검증:** `python models/yolo.py --cfg cfg/training/yolov7-l6-siav2-s.yaml` (실패하면 인덱스 수정, 통과할 때까지). ← **이게 원 설계 PC가 못 한 검증. 당신이 반드시 통과시켜라.**
3. **anchor 재계산 (kmeans)**
   - `utils/autoanchor.py`의 `kmean_anchors`로 siav2에 맞춰 재계산, 또는 학습 시 autoanchor 자동실행 확인. 결과 anchor를 cfg에 반영/기록.
4. **평가 프로토콜 동결**
   - val split 고정, 평가 명령 고정: `python test.py --data data/siav2.yaml --img 1280 --weights <ckpt>.pt --save-json` (pycocotools COCO eval, **AP_small 포함**).
   - **운영지표(미탐율@오경보율)는 이번 범위 아님** — SLA 미정(설계문서 §10). 모델 지표(mAP/AP_small)만 동결.
- **완료조건:** EDA 리포트 + head 변형 확정·빌드검증 통과 + siav2 anchor + 동결된 평가 명령/split.

---

## 3. 완료 후 (STOP & 보고)

- **학습을 시작하지 마라.** Task 3까지 끝나면 멈춘다.
- **`SIAV2_DESIGN.md` §0 체크리스트를 갱신**(한 것/검증결과 반영).
- **리포트 작성**(별도 `SIAV2_PHASE0_1_REPORT.md` 권장) — 최소 포함:
  1. 환경(torch/CUDA/TRT 버전, GPU)
  2. Task 1: 빌드 성공 여부, params/GFLOPs
  3. Task 2: **latency 비교표**(모델·width·측정조건·ms), **트리거 A 판정(GO/NO-GO)**, 추천 width
  4. Task 3: bbox EDA 요약, 확정 head 변형(+빌드검증 통과 로그), anchor, 동결 평가 명령
  5. 발견된 문제/이상(있으면)
- 그 리포트를 사람에게 제시하고 **학습 착수 승인**을 받는다.

---

## 4. 문제 발생 시 (설계문서 §8 리스크표 참조)

| 상황 | 조치 |
|---|---|
| Task 1 cfg 빌드 실패 | 에러 전문 확보 → yaml 라우팅/채널 점검. 임의 우회 말고 원인 보고 |
| Task 2 어떤 width도 2배 미달 | **트리거 A** — 진행 멈추고 "최신 채택 검토" 사유와 실측표 보고 |
| GFLOPs 낮은데 latency 느림 | ONNX 그래프·RepConv fuse·특수연산자 점검(설계문서 R5) |
| head 변형 빌드 실패 | 인덱스/Concat/IAuxDetect 입력 재정렬, 통과까지 반복 |
| siav2 데이터/`data/siav2.yaml` 없음 | 데이터 위치 확인 후 yaml 작성. 없으면 사람에게 요청 |
| v9/v11 레포 없음 | 비교 생략, W6 기준으로 트리거 A 판정(표에 "미측정" 명시) |

---

## 5. 참조 문서
- **`SIAV2_DESIGN.md`** — 설계·목표·방안·스케줄·리스크·채택규칙 (반드시 먼저 정독)
- `cfg/training/yolov7-l6-siav2.yaml` — 경량 W6 기본(nc=16, P3~P6, w0.75)
- `tools/make_random_weights.py` — Phase 0 랜덤 weight 생성기
- `data/hyp.scratch.p6.yaml` — (학습 단계용, 이번엔 미사용)

> **다시 한 번:** 이번 임무는 **학습 직전까지**. 속도 게이트(트리거 A)를 통과하고 Phase 1 준비물을 검증까지 마친 뒤 멈춰서 보고하라. 학습은 사람 승인 후 다음 단계다.
