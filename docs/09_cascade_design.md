# Phase 10 — 캐스케이드 탐지기 설계/결정 문서 (제안 CNN 1차 + char-CNN 2차)

> 이 문서는 사람이 읽는 **설계·의사결정 기록**입니다.
> 실제 수치는 `experiments/results/*_cascade_*.json`, 그림은 `docs/figures/models/cascade_curve_*.png`.
> **상태: GPU 실측 완료(2026-07-25, RTX 4080 SUPER) — 결과와 판정은 §9.**

착수 동기: *"char-CNN 의 탐지 정확도와 제안 CNN 의 속도 장점을 결합할 수 없나?"*

> ⚠️ **용어 주의 — `hybrid` 와 헷갈리지 말 것.**
> 이 프로젝트에는 이미 `hybrid` 라는 **모델**이 있다(Phase 9/docs 08: CNN stem + Transformer,
> `src/models/vit.py`). 그건 **하나의 신경망 아키텍처**다.
> 이 문서의 **캐스케이드(`cascade`)** 는 아키텍처가 아니라 **두 모델을 잇는 배치 구조**이며,
> 학습을 하지 않고 기존 체크포인트 2개를 조합한다. 논문·코드 모두 이름을 섞지 않는다.

---

## 1. 출발점 — RQ1 실측이 만든 딜레마

docs/04 §5(payload_4class, 2026-07-18 재실측) 와 docs/05 §13.2(csicnorm) 의 실측이
두 사실을 동시에 못 박았다.

| 모델 | clean Macro-F1 | MCC | **추론 처리량** | 위치 |
|---|---|---|---|---|
| 제안 CNN — gray | 0.9714 / csicnorm 0.967 | 0.9609 | **92,466 /s** | 정확도 하위 · **비용 최저** |
| 제안 CNN — RGB rb/cc/bd | 0.9836 | 0.9775 | **81,964 /s** | RGB 로 격차 좁힘, 여전히 4위 |
| char-CNN | **0.9969** / csicnorm 0.995 | **0.9957** | 6,010 /s | **정확도 최상위** · **13.6배 느림** |

즉 제안 모델의 실질적 강점은 "탐지 우월성"이 아니라 **비용**이다. docs/04 §5 는 이를
*"제안 기법이 실제로 이기는 축은 정확도가 아니라 처리량 … Pareto front 는 LogReg → RGB CNN →
char-CNN"* 으로 정리했고, docs/04 §8 상보성 분석도 "clean 탐지 상보성 이득 없음"을 확인했다.

**이 Phase 의 위치**: 그 Pareto front 위의 **점 3개를 연속 곡선으로 바꾼다.** 제안 CNN 의
비용 우위를 버리지 않으면서 정확도 격차를 원하는 만큼만 사서 메우는 구조를 만든다.

---

## 2. ⚠️ 왜 '특징 융합'이 아니라 '캐스케이드'인가 — 가장 중요한 설계 결정

"하이브리드"라고 하면 보통 **two-branch 특징 융합**을 떠올린다:
이미지 CNN 특징 ⊕ char-CNN 특징 → concat → FC. 그러나 이 구조는 **모든 입력이 두 브랜치를
전부 통과**하므로 비용이

```
C_fusion = C_cnn + C_charcnn          (> char-CNN 단독)
```

이 되어, **애초 목표였던 "제안 CNN 의 속도 장점"이 사라진다**. 정확도만 얻고 속도는 잃으며,
심지어 char-CNN 단독보다 느리다. → **채택하지 않는다.**

대신 **선택적 실행(cascade)** 을 채택한다.

```
페이로드
  └─▶ 1차: 제안 CNN (모든 트래픽)
        confidence = max softmax
        ├─ confidence ≥ τ  →  1차 판정 채택          (fast path, 대다수)
        └─ confidence <  τ  →  2차: char-CNN 재판정   (slow path, 소수)
```

```
C_cascade = C_cnn + r · C_charcnn      (r = 에스컬레이션 비율)
```

r 이 작으면 **정확도는 char-CNN 급, 지연은 CNN 급**이 된다. τ 를 움직이면 정확도-지연
곡선(Pareto)이 그려지고, 양 끝점은 정확히 두 단독 모델이다.

- τ = 0 → 아무도 넘기지 않음 = **제안 CNN 단독**
- τ > 1 → 전부 넘김 = **char-CNN 단독**

→ 캐스케이드는 두 모델을 잇는 **연속적 스펙트럼**이며, 단독 모델들은 그 특수해다.
이 성질을 단위 테스트로 못 박았다(`tests/test_cascade.py`).

**문헌 근거**: 저비용 1차 필터 + 2D-CNN 2차 정밀 판정의 하이브리드 구성
(docs/thetics 「시그니처 기반 필터링과 2D-CNN을 활용한 하이브리드 악성 트래픽 탐지 기법」).
본 설계는 1차를 시그니처가 아니라 **학습된 저비용 CNN** 으로 둔 변형이다.

---

## 3. 방법론 — 공정성을 지키는 세 가지 못

### 3.1 τ 는 반드시 **val** 에서 고른다
test 로 τ 를 고르면 임계값이 test 에 과적합돼 성능이 부풀려진다(정보 누수).
`cascade.py` 는 val 스윕으로 τ 를 확정한 뒤 **그 τ 를 test 에 한 번만** 적용한다.

선택 규칙 2종(`--select`):

| 규칙 | 의미 | 쓰는 상황 |
|---|---|---|
| `match-teacher`(기본) | val Macro-F1 이 char-CNN 단독 이상이 되는 τ 중 **에스컬레이션 최소** | "정확도는 최소한 char-CNN 만큼, 비용은 최소로" |
| `budget` | 에스컬레이션 ≤ `--budget` 제약 아래 val Macro-F1 최대 | 지연 SLA 가 먼저 정해진 배포 상황 |

`match-teacher` 로도 char-CNN 을 못 따라잡으면 val 최대 지점으로 대체하고 그 사실을
JSON `tau_selection.note` 에 남긴다(조용한 실패 금지).

### 3.2 비교 기준선 4종을 같은 test·같은 지표로 함께 낸다
1. **1차 단독**(제안 CNN) — 곡선의 좌단
2. **2차 단독**(char-CNN) — 곡선의 우단
3. **융합형 비용선**(`C_cnn + C_charcnn`) — §2 의 "안 쓴 대안"이 실제로 더 비싸다는 증거
4. **oracle 라우팅** — *1차가 틀린 샘플만* 정확히 넘겼을 때의 상한.
   어떤 확신도 게이트도 이보다 잘할 수 없으므로, **게이트 품질**(확신도가 오답을 얼마나 잘
   골라내는가)을 재는 기준이 된다. 실제 곡선이 oracle 에서 멀면 문제는 모델이 아니라 게이트다.

### 3.3 지연은 두 모델 **같은 기준**으로 잰다
`measure_latency` — warmup 후 반복 측정, CUDA 는 `synchronize()` 필수(비동기 실행이라
동기화 없이 재면 0 에 가깝게 나옴). **모델 순전파 기준**이며 전처리(텍스트→이미지/바이트)는
제외한다. 두 모델 모두 동일 기준이라 비교는 공정하고, 전처리는 µs 수준이라 결론을 바꾸지 않는다.

---

## 4. 구현물

| 파일 | 역할 |
|---|---|
| `src/models/cascade.py` (신규) | 캐스케이드 규칙·τ 스윕/선택·지연 측정·지표·그림 (CLI) |
| `src/attacks/run_evasion.py` (확장) | `--model cascade` 추가 → RQ2 회피 실험에 하이브리드 편입 |
| `tests/test_cascade.py` (신규) | 게이트 규칙·τ 격자·선택 규칙·oracle 의 순수 로직 검증 10건 |

**학습은 하지 않는다.** Phase 4/5 에서 이미 학습한 두 체크포인트를 그대로 재사용한다
→ 추가 GPU 학습 비용 0, 그리고 "같은 모델을 배치 방식만 바꿨다"는 인과가 성립한다.

### 4.1 run_evasion 확장에서 지킨 것
- `build_torch_proba` 를 원본으로 두고 `build_torch_predict` 가 그 위에 얹히도록 리팩터링
  → 단독/캐스케이드가 **같은 로딩·전처리 코드**를 공유(분기 지점 단일화).
- 캐스케이드 τ 는 `cascade_*.json` 에서 **읽어온다**. 회피 실험 데이터로 τ 를 다시 고르면
  공격 데이터에 튜닝하는 셈이라 공정성이 깨진다(운영점은 clean val 에서 한 번만 정해짐).
- 2차는 **에스컬레이션된 부분집합에만** 실행 → 실제 배포 동작·비용 구조를 그대로 재현.

---

## 5. 산출물 규약 (Phase 4 명명 관습 계승)

tag = `{track}_cascade[-{stage2}]_{text}{채널}{lr}{예산}[_bal]` — train.py 규칙 계승.

```
experiments/results/{tag}.json                지표·τ·지연·스윕 곡선
experiments/results/pred_{tag}.npz            샘플 단위 예측
docs/figures/models/cascade_curve_{tag}.png   정확도/지연 vs 에스컬레이션
docs/figures/models/cm_{tag}.png              혼동행렬
```

⚠️ **tag 에 실험을 가르는 축을 전부 넣는다** — 2차 모델·1차 채널·lr·τ 선택 규칙·균형화.
빠뜨리면 서로 덮어쓴다(커밋 5eede2f 의 RGB ablation 사고와 같은 실패 모드).
채널 접미사는 `data_image._channel_suffix` 를 재사용해 단일 진실 소스를 유지한다.

`pred_*.npz` 는 기존 규약을 따르므로 `detection_analysis.py --ref cascade` 로
**상보성 분석(docs/04 §8)에 그대로 투입**된다(추가 코드 불필요).

---

## 6. 이 실험이 답해야 할 질문과 판정 기준

결과를 미리 재단하지 않는다(docs/05 §7 원칙 계승). 어느 쪽이 나와도 논문 기여가 되도록
질문과 판정 기준을 **미리** 못 박는다.

| # | 질문 | 판정 |
|---|---|---|
| Q1 | 캐스케이드가 낮은 에스컬레이션으로 char-CNN 정확도에 도달하는가 | `escalation_rate` 와 Macro-F1/MCC. r ≤ 0.1 에서 도달하면 강한 주장 |
| Q2 | 실제 지연 이득이 있는가 | `speedup_vs_stage2_only` > 1 **그리고** `overhead_vs_stage1_only` 가 작을 것 |
| Q3 | 확신도 게이트가 좋은 라우터인가 | 실측 곡선 vs `oracle_routing` 격차 |
| Q4 | 회피(RQ2) 하에서도 유지되는가 | `evasion_*_cascade_*.json` 의 any-misclass 가 char-CNN(0.224) 쪽인가 CNN(0.594) 쪽인가 |
| Q5 | **비용이 공격당하는가** | 회피 변형 시 `escalation_rate` 상승폭 (§8 리스크) |

**주의 — 과장 금지**: docs/04 §7·§7.1 이 실측한 대로 이 트랙의 clean 지표는 **표면 토큰
포화** 상태다. 따라서 Q1 의 정확도 개선폭(최대 0.967→0.995, 약 2.8%p)을 "탐지력 향상"으로
포장하면 안 된다. 이 Phase 의 기여는 **정확도-비용 트레이드오프를 연속적으로 제어할 수 있게
만든 배치 구조**이며, 헤드라인은 정확도가 아니라 **"같은 정확도를 몇 분의 1 비용으로"** 여야 한다.

또한 docs/04 의 CV 결론(단일 split 은 실행 간 ±0.11pp 흔들림 → 우열은 5-fold CV 로 판정)에 따라,
**캐스케이드 vs 단독의 정확도 우열 주장도 단일 split 으로 내지 않는다.** 다만 캐스케이드는
설계상 두 단독 모델 사이를 보간하므로(τ 양 끝점 = 두 단독), 주장의 무게는 정확도 순위가 아니라
**같은 정확도에서의 비용 절감폭**에 둔다 — 이쪽은 단일 split 노이즈에 훨씬 둔감하다.

---

## 7. 실행 순서 (재현, GPU 권장)

```bash
# 0) 데이터 재생성(.gitignore 대상이라 매번) — 인자 없이 전체 실행(노트 유지)
python src/data/preprocess.py
python src/imaging/build_image_dataset.py --track payload_4class_csicnorm

# 1) 두 단계 모델 학습(이미 있으면 생략 — 캐스케이드는 재학습하지 않는다)
python src/models/train.py --model cnn     --track payload_4class_csicnorm --balance
python src/models/train.py --model charcnn --track payload_4class_csicnorm --balance

# 2) 캐스케이드 평가(τ 를 val 에서 확정 → test 적용) — §9 가 실제로 돌린 7종
python src/models/cascade.py --track payload_4class_csicnorm --balance
python src/models/cascade.py --track payload_4class_csicnorm --balance \
    --select budget --budget 0.05          # 지연 예산 5% 시나리오
python src/models/cascade.py --track payload_4class_csicnorm --balance \
    --select budget --budget 0.10
python src/models/cascade.py --track payload_4class_csicnorm --balance --stage2 bilstm
# 1차를 RGB CNN 으로. ⚠️ csicnorm 트랙에는 rb/cc/bd 조합(docs/04 §5 의 최고 정확도)의
#   npz·체크포인트가 없다 — 있는 것은 **기본 인코더 세트**(raw_byte,char_class,local_entropy
#   = 파일명 접미사 '_rgb')뿐이다. 그래서 아래처럼 기본값으로 돌린다.
#   rb/cc/bd 로 하려면 그 트랙의 npz 빌드 + CNN 재학습이 선행돼야 한다.
python src/models/cascade.py --track payload_4class_csicnorm --balance --channels rgb
python src/models/cascade.py --track payload_4class_csicnorm --balance --channels rgb \
    --select budget --budget 0.05
python src/models/cascade.py --track payload_4class_csicnorm --balance --channels rgb \
    --select budget --budget 0.10

# 3) RQ2 회피 실험에 하이브리드 편입(같은 공격·같은 파이프라인)
#    회피 경로는 gray 이미지 전용이므로 1차는 gray 다. 운영점을 바꿔 두 번 잰다
#    (Q5 비용 공격은 '싸게 도는' 저에스컬레이션 운영점에서만 관찰된다 — §9.5).
python src/attacks/run_evasion.py --model cascade --track payload_4class_csicnorm
python src/attacks/run_evasion.py --model cascade --track payload_4class_csicnorm \
    --tau 0.5677664577960968               # = budget 5% 운영점(cascade JSON 의 selected_tau)
python src/attacks/compare_evasion.py --track payload_4class_csicnorm

# 4) 상보성 분석에 편입(기존 스크립트 그대로)
python src/eval/detection_analysis.py --track payload_4class_csicnorm --bal \
    --ref cascade --baselines cnn,charcnn,tfidf_rf

# 스모크(코드 점검용, CPU 가능): --smoke --limit 1200
python src/models/cascade.py --track payload_4class_csicnorm --balance --smoke --limit 1200
```

---

## 8. 리스크 / 열린 결정

- [x] **⚠️ 비용 기반 공격 표면(하이브리드 고유 리스크)**: 공격자가 1차 확신도를 낮추는
  페이로드를 대량 생성하면 에스컬레이션이 치솟아 **정확도는 그대로인데 지연만 폭증**한다
  (일종의 알고리즘 복잡도 공격). 정확도 지표로는 절대 안 보인다.
  → **실측됨(§9.5)**: 저에스컬레이션 운영점에서 2.54% → 5.50%(**2.17배**)로 올라
  예산 5% SLA 가 공격 하에서 깨진다. 논문 한계/고찰에 서술한다.
- [ ] **총 학습 비용은 줄지 않는다**: 캐스케이드가 줄이는 것은 **추론 지연**이며, 두 모델을
  모두 학습해야 하므로 학습 비용은 오히려 합산이다. 근거로 인용할 수치는 **추론 처리량**
  (docs/04 §5: RGB CNN 81,964 /s vs char-CNN 6,010 /s)이지 학습 지표(s/epoch)가 아니다.
- [x] **1차를 gray 로 쓸지 RGB 로 쓸지** → **RGB 채택(§9.3)**. 정확도뿐 아니라 **비용에서도**
  RGB 가 이긴다: char-CNN 급 정확도에 도달하는 데 gray 는 78.4% 를 넘겨야 했지만 RGB 는
  27.7% 로 충분했다(1차가 정확할수록 넘길 일이 줄어 캐스케이드 전체가 싸진다).
  우려했던 FPR 열세는 운영점에서 재현되지 않았다(양쪽 모두 0.0014).
- [ ] **확신도 보정(calibration)**: 1차 CNN 은 class weight + 언더샘플링으로 학습돼 확률이
  잘 보정돼 있지 않다. 게이트는 순위만 쓰므로 원리상 문제없지만, oracle 격차가 크면
  temperature scaling 등 보정을 검토한다.
  → **격차가 실제로 크다(§9.4)**: oracle 은 1.9%(RGB)만 넘겨도 상한에 닿는데 실측 게이트는
  같은 정확도에 27.7% 를 쓴다. **후속 1순위 과제**.
- [ ] **엔트로피 게이트 대안**: 현재 게이트는 max softmax 1종. 예측 엔트로피·상위 2개 마진
  등으로 바꾸면 같은 에스컬레이션에서 더 좋은 라우팅이 될 수 있다(ablation 후보).
  → §9.4 의 oracle 격차가 이 ablation 의 **상한 이득**을 정량화해 준다(최대 −25.8pp 에스컬레이션).
- [ ] **대안 설계 — 지식 증류(knowledge distillation)**: char-CNN 을 교사로 제안 CNN 을
  재학습하면 **추론은 CNN 하나뿐**이라 속도 100% 유지 + 파라미터 증가 0. 캐스케이드가
  "두 모델 유지" 비용을 지불하는 것과 대비된다. 캐스케이드 실측 후 비교군으로 검토.
- [ ] `payload_4class`(원본 Normal) 트랙에서도 같은 곡선을 낼지 결정(현재는 csicnorm 우선).

---

## 9. GPU 실측 결과 (2026-07-25, RTX 4080 SUPER / cu124)

트랙 `payload_4class_csicnorm`, val 22,628 · test 22,629, 클래스
`[CommandInjection, Normal, SQLInjection, XSS]`. 학습은 하지 않았고 Phase 4/5 의 `_bal`
체크포인트를 그대로 재사용했다(설계 의도대로 추가 학습 비용 0).

### 9.1 arm 7종 — 전체 결과

`esc`=에스컬레이션 비율(2차 호출 비중), `spd`=2차 단독 대비 배속, `ovh`=1차 단독 대비 비용.
지연은 모두 **같은 실행 안에서** 잰 값이다(ms/sample, 배치 512 순전파 기준).

| 1차 | 2차 | τ 규칙 | τ | esc | Macro-F1 | MCC | ms/sample | spd | ovh |
|---|---|---|---|---|---|---|---|---|---|
| gray | char-CNN | match-teacher | 0.9998 | 78.36% | 0.9954 | 0.9940 | 0.0508 | 1.11× | 7.74× |
| gray | char-CNN | budget 10% | 0.7509 | 7.46% | 0.9856 | 0.9756 | 0.0108 | 5.23× | 1.64× |
| gray | char-CNN | budget 5% | 0.5678 | 2.54% | 0.9754 | 0.9579 | 0.0080 | 7.05× | 1.22× |
| gray | BiLSTM | match-teacher | 0.8099 | 9.98% | 0.9831 | 0.9714 | 0.0212 | 6.92× | 3.23× |
| **RGB** | char-CNN | match-teacher | 0.9971 | 27.73% | **0.9955** | **0.9941** | 0.0223 | 2.52× | 3.32× |
| **RGB** | char-CNN | **budget 10%** | 0.9425 | **7.99%** | **0.9934** | **0.9907** | **0.0111** | **5.07×** | 1.68× |
| **RGB** | char-CNN | budget 5% | 0.7516 | 2.71% | 0.9890 | 0.9837 | 0.0081 | 6.92× | 1.23× |

기준선(같은 test·같은 지표): **1차 단독** gray 0.9676/MCC 0.9418(0.0066 ms) · RGB
0.9804/MCC 0.9725(0.0066 ms), **2차 단독** char-CNN 0.9954/MCC 0.9940(0.0561 ms) ·
BiLSTM 0.9838/MCC 0.9736(0.1468 ms), **융합형(두 브랜치 상시)** 0.0627 ms.

설계 성질도 실측으로 확인했다 — 곡선의 양 끝점이 정확히 두 단독 모델이다
(τ=0 → esc 0.0000/F1 0.9676, τ=1.01 → esc 1.0000/F1 0.9954).

### 9.2 헤드라인 — "같은 정확도를 몇 분의 1 비용으로"

**RGB 1차 + 예산 10%** 를 대표 운영점으로 삼는다:

> 트래픽의 **8.0% 만 2차로 올려** char-CNN 대비 MCC 0.9940 → 0.9907(−0.33pp)에
> **지연은 5.07배 낮다**(0.0561 → 0.0111 ms/sample). 융합형(0.0627 ms) 대비 5.6배.

정확도를 한 치도 양보하지 않으려면 **RGB + match-teacher** 가 27.7% 에스컬레이션으로
char-CNN 을 **넘어선다**(0.9955 vs 0.9954, MCC 0.9941 vs 0.9940)면서도 2.52배 빠르다.

⚠️ docs/09 §6 의 경고대로 **정확도 개선을 주장하지 않는다**. 이 트랙은 표면 토큰 포화
상태이고(docs/04 §7), 위 차이는 단일 split 노이즈(±0.11pp)와 같은 자릿수다.
주장의 무게는 전부 **같은 정확도에서의 비용 절감폭**에 둔다.

### 9.3 Q1~Q3 판정

| # | 질문 | 판정 |
|---|---|---|
| Q1 | 낮은 에스컬레이션으로 char-CNN 정확도 도달? | **1차에 달렸다.** gray 는 78.4% 가 필요해 사실상 실패(사전 기준 r≤0.1 미달), RGB 는 27.7%. 다만 **r≤0.1 에서도** RGB 는 char-CNN MCC 의 99.7% 를 회수한다 |
| Q2 | 실제 지연 이득? | **있다.** 대표 운영점에서 `speedup_vs_stage2_only`=5.07× 이고 `overhead_vs_stage1_only`=1.68× — 1차 단독의 1.7배 비용으로 2차 단독의 정확도를 사실상 산다 |
| Q3 | 확신도 게이트가 좋은 라우터인가 | **아니다(§9.4).** oracle 대비 격차가 크다 |

Q1 의 gray 실패는 규칙 탓이 아니라 **1차 모델의 확신도 품질** 탓이다. `match-teacher` 는
"val Macro-F1 ≥ char-CNN" 을 요구하는데, gray CNN 의 확신도가 오답을 잘 골라내지 못해
그 조건을 만족시키려면 거의 전량을 넘겨야 했다. 이때 speedup 은 1.11배로 **캐스케이드를
쓸 이유가 사라진다** — 즉 이 구조의 성패는 게이트 품질이 좌우한다.

### 9.4 게이트 품질 — oracle 격차가 이 Phase 의 가장 큰 미결

oracle 라우팅(1차가 틀린 샘플만 정확히 넘겼을 때의 상한):

| 1차 | oracle esc | oracle Macro-F1 | 실측 게이트가 같은 수준에 쓰는 esc |
|---|---|---|---|
| gray | 3.96% | 0.9968 | 78.36% (**19.8배**) |
| RGB | 1.87% | 0.9982 | 27.73% (**14.8배**) |

확신도(max softmax)는 **오답을 15~20배 과잉 포착**한다. 정확도 상한이 아니라 **비용 상한**이
문제이며, 게이트를 개선하면(엔트로피·상위 2개 마진·temperature scaling) 정확도를 그대로
두고 에스컬레이션만 최대 −25.8pp 줄일 여지가 있다. §8 의 두 열린 항목이 여기서 정량화됐다.

### 9.5 Q4·Q5 — 회피(RQ2) 하에서의 거동

`run_evasion.py --model cascade` 를 **두 운영점**에서 돌렸다(k=5 누적 변형 기준).

| 운영점 | benign-evasion | any-misclass | esc(clean) | esc(공격 k=5) | 비용 증가 |
|---|---|---|---|---|---|
| τ=0.9998 (match-teacher) | 0.0001 | **0.3641** | 78.36% | 79.96%(최대) | 1.02배 |
| τ=0.5678 (budget 5%) | 0.0009 | **0.5557** | 2.54% | 5.50%(최대) | **2.17배** |
| — 참고: char-CNN 단독 | 0.0001 | 0.2239 | — | — | — |
| — 참고: 제안 CNN 단독 | 0.0022 | 0.5944 | — | — | — |

**Q4 판정**: 캐스케이드의 회피 강건성은 **τ 로 두 단독 모델 사이를 이동한다**. 높은 τ 는
char-CNN 쪽(0.364), 낮은 τ 는 CNN 쪽(0.556). 즉 §2 에서 주장한 "연속 스펙트럼"이
clean 정확도뿐 아니라 **공격 하 강건성에서도 성립**한다. 다만 τ=0.9998 에서도 char-CNN
단독(0.224)에 못 미치는데, 원인은 **넘기지 않은 21.6%** 다 — 1차가 확신했기에 남은
샘플이지만 변형되면 틀린다. "확신도가 높다"가 "변형에 강하다"를 뜻하지 않는다.

**Q5 판정(캐스케이드 고유 리스크)**: 비용 기반 공격이 **실재한다**. 저에스컬레이션
운영점에서 공격은 정확도를 흔드는 것과 **동시에** 2차 호출을 2.17배로 늘려 예산 5% SLA 를
깼다(5.50%). 정확도 지표만 보면 안 보이는 열화다. 반대로 이미 78% 를 넘기는 운영점은
증가 여지가 없어 1.02배에 그친다 — **비용 공격 표면은 캐스케이드가 싸게 돌아갈수록 커진다.**
운영 시 에스컬레이션 비율을 지표로 모니터링하고 상한 큐를 두는 완화책이 필요하다.

### 9.6 상보성 분석 — clean 에서 새로 잡는 건 없다

`detection_analysis.py --ref cascade --baselines cnn,charcnn,tfidf_rf`(공격 21,907건):

| baseline | 둘 다 탐지 | cascade 만 | baseline 만 | 둘 다 놓침 |
|---|---|---|---|---|
| cnn | 21,889 | 10 | 8 | 0 |
| charcnn | 21,899 | 0 | 0 | 8 |
| tfidf_rf | 21,899 | 0 | 8 | 0 |

char-CNN 과 **탐지 집합이 완전히 동일**하다(기본 운영점이 78% 를 넘기므로 당연한 귀결).
docs/04 §8 의 "clean 탐지 상보성 이득 없음" 결론이 캐스케이드에서도 유지된다 —
이 Phase 의 기여는 탐지 집합 확장이 아니라 **비용 축소**임을 다시 확인한다.

### 9.7 실측 중 고친 것 두 가지 (재현성)

1. **지연 측정이 실행마다 60% 흔들렸다.** 1차 측정에서 char-CNN 이 같은 조건인데
   0.0555~0.0904 ms 로 튀어 gray budget 5% 의 speedup 이 9.30배로 **부풀려졌다**(재측정 7.05배).
   원인은 단발 측정(warmup 3·repeats 10)이 GPU 클럭 부스트 상태·타 프로세스 간섭을 그대로
   받는 것. → `measure_latency` 를 **5라운드 × 20회의 중앙값**으로 바꾸고 전 arm 을 재실측했다.
   재측정 후 변동은 0.7% 이내(ms1 0.0066~0.0067, char-CNN 0.0561~0.0565).
   **헤드라인이 속도인 Phase 에서 지연 측정의 분산 관리는 정확도 지표만큼 중요하다.**
2. **`run_evasion.py` 의 tag 에 τ 축이 없었다.** `evasion_{track}_{model}` 뿐이라 운영점이
   다른 두 실행이 서로 덮어썼을 것(커밋 5eede2f 의 RGB ablation 사고와 같은 실패 모드).
   → `--tau` 를 지정한 실행은 `_tau{값}` 접미사로 분리하고(기본 운영점은 생략 — train.py 의
   lr 규칙과 같은 관습), JSON 안에도 `tau` 를 기록했다. `compare_evasion.py` 는 캐스케이드
   운영점들을 glob 으로 발견해 라벨의 τ 를 **파일에서 읽어** 그린다(τ 하드코딩 금지).

### 9.8 산출물

```
experiments/results/payload_4class_csicnorm_cascade[-bilstm]_raw[_rgb][_b0.05|_b0.1]_bal.json  (7종)
experiments/results/pred_...(같은 tag).npz                                                     (7종)
experiments/results/evasion_payload_4class_csicnorm_cascade[_tau0.567766]_{single,stacked}.json
experiments/results/detection_complementarity_payload_4class_csicnorm_cascade.json
docs/figures/models/cascade_curve_*.png, cm_*.png                                              (각 7종)
docs/figures/attacks/evasion_{single,stacked}_payload_4class_csicnorm_cascade[_tau...].png
docs/figures/attacks/evasion_compare_payload_4class_csicnorm.png                               (캐스케이드 2점 추가)
```
