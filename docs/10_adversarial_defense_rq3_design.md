# Phase 11 — 적대적 방어 학습 설계/결정 문서 (RQ3: 강건성-성능 트레이드오프)

> 이 문서는 사람이 읽는 **설계·의사결정 기록**입니다.
> 실제 수치는 `experiments/results/*_def-*.json`, 그림은 `docs/figures/defense/` 에 남긴다.
> **상태: 설계 확정 · 구현 미착수(2026-07-28).**

RQ3(설계문서 원문): *"회피 공격 샘플을 활용한 adversarial training 을 적용하면 강건성이 얼마나
개선되며, 이 과정에서 정상 트래픽 오탐률과 기존 탐지 정확도는 얼마나 희생되는가?"*

---

## 0. 착수 배경 — 왜 지금, 그리고 왜 늦었나

RQ3 는 논문 5장의 `RQ1 → RQ2 → RQ3` 축에서 **본체**인데, Phase 9(ViT)·Phase 10(캐스케이드)이
먼저 나가면서 밀렸다. docs/08 §6-3 이 *"ViT arm 이 RQ3 를 밀어내면 안 된다"* 고 스스로 경고해
뒀는데 실제로 그렇게 됐다 — 기록해 둔다.

지금 착수하는 이유는 두 곁가지(ViT·캐스케이드)가 모두 **"제안 CNN 유지"** 로 닫혀서
공격 대상 모델이 더 이상 흔들리지 않기 때문이다. 방어 실험의 대상이 고정됐다.

---

## 1. ⭐ 가장 먼저 풀어야 할 문제 — 방어할 대상이 이미 0 이다

docs/05 §13.2 실측: **주 지표 benign-evasion 이 5모델 전부 ~0** 이다(제안 CNN 0.0022,
char-CNN 0.0001, TF-IDF 0.0000). RQ3 를 원문 그대로 수행하면 **0 을 0 으로 만드는 실험**이 된다.
바닥 효과(floor effect)라 개선폭도 희생폭도 측정 불가능하고, 어떤 결과가 나와도 해석이 안 된다.

### 1.1 확정 — 주 지표를 any-misclass 로 옮긴다 (2026-07-28, 사용자 결정)

| 축 | 기존 RQ3 | **재정의 RQ3** |
|---|---|---|
| 강건성 주 지표 | benign-evasion(공격→Normal) | **any-misclass(변형 하 오분류율, k=5)** |
| 비용 주 지표 | FPR | **FPR + clean MCC/Macro-F1 희생** (원문 유지) |
| benign-evasion | 주 지표 | **보조 지표로 계속 보고**(바닥 효과 명시) |

**지표 교체는 결과를 보고 고른 것이 아니다.** 근거가 되는 실측(docs/05 §13.2)은 방어 실험을
돌리기 **전에** 이미 존재하며, 이 문서로 **사전 고정**한다(HARKing 방지). 논문에도 이 순서
그대로 쓴다 — "바닥 효과를 확인했기 때문에 지표를 옮겼다"가 정직한 서술이다.

### 1.2 ⚠️ any-misclass 를 방어 대상으로 삼는 것의 정당성 — 반드시 방어해야 할 지점

any-misclass 는 **WAF 우회가 아니다**(SQLi→XSS 로 틀려도 탐지·차단은 된다). 심사에서
*"그걸 왜 고쳐야 하나"* 가 반드시 나온다. 근거 3개를 미리 세운다.

1. **운영 비용은 실재한다.** 공격 유형은 대응 playbook·차단 규칙·포렌식 분류를 가른다.
   SQLi 를 XSS 로 라벨링하면 잘못된 대응이 자동 실행된다. 탐지는 되지만 대응은 틀린다.
2. **결정 경계 불안정성의 프록시다.** 의미보존 표면 변형에 예측이 흔들린다는 것은 그 표현이
   표면 토큰 shortcut(docs/04 §7)에 의존한다는 직접 증거다. 지금은 Normal 경계까지 안 넘을
   뿐이고, **더 강한 공격(GA·feature-space)에 대한 선행 취약 지표**로 읽는 것이 타당하다.
3. **유일하게 변별력이 있는 강건성 축이다.** benign-evasion 은 5모델 전부 0(변별력 0)인데
   any-misclass 는 표현방식별로 **2.6배 차이**(char-CNN 0.224 ~ 제안 CNN 0.594)가 난다.
   측정 가능한 것을 측정한다.

**한계도 같은 문단에 쓴다**: any-misclass 개선은 보안적 침해 감소를 직접 의미하지 않는다.
이 Phase 의 주장은 *"표현 안정성(representation stability)을 얼마의 비용으로 살 수 있는가"* 이지
*"WAF 가 더 안전해졌다"* 가 아니다.

---

## 2. 위협 모델 / 방어자 모델

- **공격자**: docs/05 §2 그대로 계승 — 블랙박스·모델 무관, 의미보존 표면 변형만. 방어 도입
  사실을 알지만 방어 모델의 가중치는 모른다(적응형 공격은 §10 열린 결정).
- **방어자**: 학습 데이터와 변형 규칙 집합에 접근 가능. 단 **공격자가 쓸 변형을 전부 알지는
  못한다** — 이 가정이 §4 의 held-out 분할을 강제한다.
- **평가 트랙**: `payload_4class_csicnorm` / raw / side=48 (docs/05 §1 계승). Normal 이 실제
  HTTP 트래픽이어야 FPR 이 의미를 갖는다.

---

## 3. 방어 3안과 채택

| 안 | 방식 | 추론 비용 | 채택 |
|---|---|---|---|
| **A. 적대적 증강 학습**(`advtrain`) | 변형 페이로드를 train 에 섞어 재학습 | **불변**(같은 아키텍처) | **주 실험** |
| **B. 입력 정규화**(`norm`) | 예측 전에 URL/HTML 디코딩(`text_decoded`) | 전처리 +α | **비학습 대조군(필수)** |
| C. 일관성 정규화(`consist`) | 원본-변형 쌍의 출력 분포를 KL 로 결속(TRADES 류) | 불변 | 여유 시 |

**B 를 반드시 넣는 이유 — 이 Phase 에서 가장 값싼 정직성**
docs/05 §3 이 *"raw 트랙은 공격에 유리한 최악 조건이며, 디코딩 정규화가 방어책이 되는지는
후속 Phase 에서"* 라고 **약속해 두고 갚지 않았다.** 여기서 갚는다. 그리고 실측상 any-misclass 를
끌어올린 것은 거의 전적으로 URL 인코딩 계열(Δ+0.59~0.62, docs/05 §12.2)이므로, **디코딩 한 줄이
학습형 방어를 이겨버릴 가능성이 상당히 높다.** 그 결과가 나오면 그것을 헤드라인으로 쓴다 —
"복잡한 방어보다 파이프라인 정규화가 낫다"는 실무적 결론이 더 강한 기여다.

**A 의 추론 비용이 불변인 점은 명시적 이득**이다. 제안 기법의 유일한 방어 가능한 우위가
처리량(docs/04 §5.1)인데, A 는 그것을 건드리지 않는다. B 는 전처리 비용이 늘어나므로
**처리량을 반드시 재측정**한다(unquote 반복은 µs 급이나 정직하게 잰다).

C 는 구현 비용 대비 정보량이 낮아 후순위 — A 가 효과 없을 때만 간다.

---

## 4. ⭐ 변형 분할(seen / held-out) — 이 Phase 의 방법론적 핵심

**학습에 쓴 변형으로 평가하면 "본 적 있는 공격을 막았다"는 자명한 결과가 나온다.**
반드시 **평가 전용(held-out) 변형**을 남긴다. 현재 변형 14종(`mutations.REGISTRY`)을 4계열로 묶는다.

| 계열 | 변형 | 실측 위력(docs/05) |
|---|---|---|
| **E. 인코딩** | `url_encode`, `double_url_encode`, `xss_html_entity`, `xss_decimal_entity` | **압도적**(Δ+0.59~0.62) |
| C. 대소문자 | `random_case`, `xss_tag_case` | 미미 |
| W. 공백·주석 | `space_to_tab`, `space_to_comment`, `sqli_inline_comment`, `sqli_version_comment`, `cmdi_ifs_substitution` | 미미 |
| S. 구문 동치 | `sqli_logical_equiv`, `cmdi_quote_insert`, `cmdi_separator_swap` | 미미 |

**분할 시나리오 2종을 함께 돌린다(둘 다 헤드라인)**

| 시나리오 | 학습에 쓰는 변형 | 평가 | 답하는 질문 |
|---|---|---|---|
| **S-0 (seen)** | 4계열 전부 | 같은 4계열 | **낙관 상한** — 방어가 원리적으로 얼마나 줄일 수 있나 |
| **S-A (leave-encoding-out)** | C+W+S 만 | **E 포함 전부** | **진짜 질문** — 미지의 강한 벡터에 일반화되나 |

**S-A 가 가혹하다는 것을 알고 넣는다.** 학습에 남는 C/W/S 는 실측상 거의 무력한 변형이라
방어가 아무것도 못 배울 수 있다. 그래서 S-0 을 **상한선으로 나란히** 둔다. 두 값의 격차
자체가 결과다: *"방어는 본 변형만 외운다"* 인지 *"표면 변형 일반에 강해진다"* 인지가 여기서 갈린다.

> ⚠️ held-out 계열(E)은 **train 에도 val 에도 절대 넣지 않는다.** val 에 넣으면 조기 종료를
> 통해 간접 누수된다(§7.3).

---

## 5. 실험 매트릭스

**대상 모델 2종으로 고정한다** — any-misclass 스펙트럼의 양 끝:

| 모델 | 채널 | clean MCC | any-misclass(k=5) | 이 모델을 넣는 이유 |
|---|---|---|---|---|
| **제안 CNN** | RGB(`_rgb`) | 0.9725 | **0.594**(최상위 동요) | 방어 이득이 가장 클 후보 = 주 대상 |
| **char-CNN** | — | 0.9940 | **0.224**(최저) | 이미 강건한 모델도 더 좋아지나, 아니면 천장인가 |

char-CNN 을 빼면 *"이미지 표현만 방어가 필요하다"* 는 해석을 배제할 수 없다. 표현방식 축을
남기려면 둘 다 필요하다. BiLSTM·TF-IDF 는 확장(§10).

**arm 목록(모델당)**

| arm | 설명 |
|---|---|
| `base` | 방어 없음 = Phase 4/5 의 기존 `_bal` 체크포인트 재사용(재학습 없음) |
| `advtrain` × {S-0, S-A} × ρ∈{0.25, 0.5} | 적대적 증강 학습 4셀 |
| `norm` | 입력 정규화(디코딩). 학습은 `--text decoded`, 평가 시 변형 후 디코딩 |
| `advtrain+norm` | 둘 다(직교하는지 확인) — 여유 시 |

ρ = 증강 비율(§7.2). 모델 2종 × (기존 재사용 + 4 + 1) = **신규 학습 10회**, GPU 1~2시간 규모.

---

## 6. 판정 기준 — 미리 못 박는다 (HARKing 방지)

| # | 가설 | 판정 기준 |
|---|---|---|
| **H3-1** | 적대적 증강이 any-misclass(k=5)를 낮춘다 | **≥10pp 절대 감소 = 효과 있음**, 5~10pp = 5-fold CV 로 판정, <5pp = **효과 없음**으로 보고 |
| **H3-2** | 그 대가가 수용 가능하다 | clean MCC 하락 **≤1pp** 이고 FPR 상승 **≤0.5pp** 이면 "수용 가능한 트레이드오프" |
| **H3-3** ⭐ | 방어가 **미지의 변형**에 일반화된다 | S-A 감소폭 ≥ S-0 감소폭의 **50%** → "일반화", 미만이면 **"본 변형 암기"** 로 보고 |
| **H3-4** | 학습형 방어가 정규화 방어보다 낫다 | `advtrain` 이 `norm` 을 못 이기면 **그 사실을 헤드라인으로** 낸다(§3) |

**단일 split 규칙**: CLAUDE.md 대로 실행 간 변동은 ±0.11pp 다. 위 기준(5pp/10pp)은 그보다
**한 자릿수 크므로 단일 split 로 보고 가능**하다. 그 미만 구간에 걸리면 `cross_validate.py`
5-fold + paired t-test 로 판정한다. **기준을 결과에 맞춰 사후 조정하지 않는다.**

**과장 금지 못**: clean 지표는 이 트랙에서 표면 토큰 포화 상태다(docs/04 §7.1). 방어로 clean
MCC 가 올라가도 **"탐지력 향상"으로 포장하지 않는다.** 이 Phase 의 주장은 강건성 축 하나다.

---

## 7. 구현 계획 — 접합점과 함정

### 7.1 ⚠️ 이미지 모델은 npz 에 원문이 없다 (가장 큰 함정)

`train.py::build_datasets` 는 이미지 모델일 때 `data/images/*.npz`(이미 이미지화된 배열)를
읽는다. **원문 텍스트가 없어 그 자리에서 변형할 수 없다.** 텍스트 모델은 CSV 를 읽으므로 문제없다.

→ **결정**: 증강은 **텍스트 단계에서만** 수행하고(변형 규칙은 문자열 함수), 이미지 모델의
증강 학습은 `train` split 을 **CSV → 변형 → `payload_to_image`/`payload_to_rgb_image` 온더플라이**
경로로 만든다. `run_evasion.py` 가 이미 검증한 경로라 신규 위험이 낮다(docs/05 §13.5).

> ⚠️ **원본까지 같은 경로로 만든다.** 원본은 npz, 증강본만 온더플라이로 만들면 두 경로의
> 미세한 차이가 "증강본 여부"와 상관되어 **새 shortcut** 이 된다. train split 전체를 온더플라이로
> 통일한다. val/test 는 기존 npz 그대로(평가 일관성 유지).
> → **회귀 테스트 필수**: 동일 텍스트에 대해 npz 이미지와 온더플라이 이미지가 **바이트 단위로
> 동일**함을 assert (`tests/test_defense.py`). 이게 깨지면 이 Phase 의 모든 수치가 무효다.

### 7.2 ⚠️ 증강은 '추가'가 아니라 '치환'이다

변형 대상은 공격 3클래스뿐이다. 변형본을 **추가**(append)하면 공격 클래스만 불어나 Normal 이
상대적 소수가 되고, **FPR·benign-evasion 이 방어 효과가 아니라 분포 변화 때문에** 움직인다.
`--balance` 언더샘플링(docs/05 §11-C)의 전제도 깨진다.

→ **결정**: **치환(replace) 방식.** 균형화 이후, 각 공격 클래스에서 비율 ρ 만큼을 골라
그 샘플을 변형본으로 **바꿔치기**한다. train 크기·클래스 비율 불변 → 방어 효과와 분포 효과가
분리된다. append 방식은 ablation 으로만(§10).

- ρ ∈ {0.25, 0.5}. 변형 예산은 회피 실험과 맞춰 **k=1~3 랜덤**(평가는 k=5 까지 — 학습보다 강한
  공격으로 평가해야 낙관 편향이 없다).
- 증강 rng 는 `--seed` 파생 고정(결정론 유지).

### 7.3 ⚠️ 조기 종료 기준을 arm 별로 맞춘다 (ViT lr 교훈의 재적용)

기존 조기 종료는 **clean val Macro-F1** 기준이다. 방어 arm 을 이 기준으로 멈추면 강건성이
좋아지는 방향으로는 학습이 진행돼도 멈춰버린다 — docs/08 §9.1 에서 겪은 *"기본값이 한쪽
아키텍처에 맞춰져 있으면 나머지가 자동으로 불리해진다"* 와 **동형의 함정**이다.

→ **결정**: `base`/`norm` 은 clean val, `advtrain` 은 **mixed val**(clean + **seen 계열만** 변형)로
조기 종료한다. 각 arm 을 자기 목적함수로 최적화하는 것이 공정하다. **held-out 계열은 val 에
넣지 않는다**(§4). 이 결정과 근거를 JSON `early_stop.criterion` 에 기록한다.

### 7.4 파일 배치 (기존 재사용 원칙)

| 파일 | 책임 |
|---|---|
| `src/defense/augment.py` (신규) | 계열 분할 상수(E/C/W/S) · 치환식 증강 데이터셋 생성. `mutations.py` 재사용, 순수 함수 |
| `src/models/train.py` (확장) | `--defense {none,advtrain,norm}`, `--aug-ratio`, `--mutation-split {S0,SA}`, `--aug-budget` 추가. 학습 루프·평가·조기종료는 그대로 |
| `src/attacks/run_evasion.py` (확장) | **`--text {raw,decoded}` 추가** — 현재 raw 고정이라 정규화 방어(B)를 평가할 수 없다. 변형 후 `preprocess.normalize_text` 를 통과시키는 경로 |
| `src/eval/cross_validate.py` (확장) | 5pp 미만 구간 판정용으로 동일 인자 전달 |
| `tests/test_defense.py` (신규) | 계열 분할 배타성 · 치환 후 클래스 비율 불변 · **npz↔온더플라이 이미지 동일성**(§7.1) · held-out 변형이 train/val 에 안 섞임 |

**train.py 에 얹는 이유**: 데이터로더·조기종료·`metrics.py`·CV 스크립트를 전부 재사용하려면
별도 학습 스크립트를 만들면 안 된다(계산 방식 차이 배제 원칙, docs/04 §3).

---

## 8. 산출물 규약 — tag 에 방어 축을 전부 넣는다

커밋 5eede2f(RGB 조합 상호 덮어쓰기)·docs/09 §9.7(τ 축 누락)과 **같은 실패 모드**를 세 번째로
겪지 않는다. 기존 규칙(`채널` + `_lr` + `_bal`)에 방어 축을 잇는다.

```
{track}_{model}_{text}{채널}{lr}_def-{advtrain|norm}[_{S0|SA}][_r{ρ}]{_bal}
예) payload_4class_csicnorm_cnn_raw_rgb_def-advtrain_SA_r0.5_bal
    payload_4class_csicnorm_charcnn_decoded_def-norm_bal
```

- `--defense none` 이면 `_def-*` 를 통째로 생략 → **기존 산출물 파일명과 100% 호환**
  (`_lr` 기본값 생략 관습과 동일).
- 규칙은 `train.py`·`cross_validate.py` **2곳** 동시 갱신(CLAUDE.md 규칙).
- 그림: `docs/figures/defense/` (신규, 추적 대상) — 강건성-비용 산점도(x=clean MCC 희생,
  y=any-misclass 감소), 예산-ASR 곡선의 방어 전/후 겹침, S-0 vs S-A 격차 막대.
- 지표 JSON 에 `defense` 블록(방식·ρ·분할·seen 계열 목록·조기종료 기준)을 기록해 **파일명만으로
  재구성 불가능한 정보를 남긴다.**

---

## 9. 실행 순서 (재현)

```bash
# --- 노트북(CPU)에서 가능 ---
python src/data/preprocess.py                      # 인자 없이 전체(노트 유지)
python src/imaging/build_image_dataset.py --track payload_4class_csicnorm
python src/imaging/build_image_dataset.py --track payload_4class_csicnorm --channels rgb
python -m pytest tests/ -q                         # test_defense.py 포함
python src/models/train.py --model cnn --track payload_4class_csicnorm --balance \
    --channels rgb --defense advtrain --mutation-split SA --aug-ratio 0.5 --smoke --limit 1200

# --- 데스크톱(GPU) 전면 학습: 모델 2종 × arm 5 = 10회 ---
for SPLIT in S0 SA; do for R in 0.25 0.5; do
  python src/models/train.py --model cnn --track payload_4class_csicnorm --balance \
      --channels rgb --defense advtrain --mutation-split $SPLIT --aug-ratio $R
  python src/models/train.py --model charcnn --track payload_4class_csicnorm --balance \
      --defense advtrain --mutation-split $SPLIT --aug-ratio $R
done; done
# 정규화 방어(B) — 학습은 decoded 컬럼으로
python src/imaging/build_image_dataset.py --track payload_4class_csicnorm --text decoded --channels rgb
python src/models/train.py --model cnn     --track payload_4class_csicnorm --balance \
    --channels rgb --text decoded --defense norm
python src/models/train.py --model charcnn --track payload_4class_csicnorm --balance \
    --text decoded --defense norm

# --- 회피 재평가(같은 공격·같은 파이프라인, 모델만 교체) ---
python src/attacks/run_evasion.py --model cnn --track payload_4class_csicnorm --defense ...   # arm 별
python src/attacks/run_evasion.py --model cnn --track payload_4class_csicnorm --text decoded  # norm arm
python src/attacks/compare_evasion.py --track payload_4class_csicnorm
```

⚠️ GPU 는 1대뿐 → **순차 실행**. 백그라운드 작업이 "killed" 로 보여도 파이썬 자식은 살아 있을 수
있으므로 재실행 전 `Get-CimInstance Win32_Process -Filter "Name='python.exe'"` 로 확인(docs/08 §9.6).

---

## 10. 리스크 / 열린 결정

- [ ] **⚠️ 적응형 공격(adaptive attack) 미수행**: 방어 모델을 알고 그에 맞춰 재설계된 공격은
  다루지 않는다. 방어 논문의 표준 비판 지점이므로 **한계로 명시**한다. 정식 대응은 GA 탐색을
  방어 모델에 대해 다시 도는 것(docs/05 §4.5-③, 미착수).
- [ ] **S-A 가 무력할 위험**: 학습에 남는 C/W/S 계열이 실측상 거의 무효라 방어가 배울 게 없을 수
  있다. 그 경우에도 S-0 과의 격차가 결과이며, "인코딩 계열은 학습으로 못 막고 정규화로 막는
  종류"라는 결론(§3 B안)으로 수렴한다 — **어느 쪽이든 논문에 남는다.**
- [ ] **치환(replace)이 데이터를 줄인다**: ρ=0.5 면 원본 공격 샘플의 절반이 사라진다. clean 성능
  하락이 방어 부작용인지 데이터 감소인지 헷갈릴 수 있다 → append 방식 ablation 1셀로 분리 확인.
- [ ] **B(정규화)의 부작용 미검증**: 디코딩은 정상 트래픽도 바꾼다. CSIC 실트래픽 Normal 이
  디코딩 후 공격과 가까워지면 **FPR 이 오를 수 있다** — H3-2 기준으로 반드시 확인.
- [ ] **캐스케이드와의 상호작용**(Phase 10 연결): 방어 학습된 CNN 을 1차로 쓰면 확신도 분포가
  바뀌어 게이트 품질·에스컬레이션이 달라진다. docs/09 §9.4 의 oracle 격차(15~20배 과잉 포착)가
  방어로 줄어드는지는 **후속 과제**. 이 Phase 에서는 τ 재선택 없이 건드리지 않는다.
- [ ] **C(일관성 정규화) 착수 여부** — A 의 효과가 <5pp 일 때만.
- [ ] BiLSTM·TF-IDF 로 확장할지(현재 2종 고정) — 표현방식 축 결론이 흔들리면 추가.
- [ ] (운영) 브랜치 정리 — 여전히 `phase1-data-acquisition` 에 전 Phase 누적.

---

## 11. 요약 — 이 문서가 확정한 것

1. **RQ3 주 지표를 any-misclass 로 재정의**하고(바닥 효과 근거는 사전 실측), benign-evasion·FPR·
   clean MCC 는 보조/비용 축으로 함께 보고한다.
2. 방어는 **A(적대적 증강, 주) + B(입력 정규화, 비학습 대조군)**. B 가 이기면 그 사실이 헤드라인.
3. **seen(S-0) / held-out(S-A) 분할**로 "본 변형 암기"와 "일반화"를 분리한다 — 이 Phase 의
   방법론적 핵심.
4. 증강은 **치환식**(클래스 비율 불변), 조기 종료는 **arm 별 자기 목적함수**(ViT lr 교훈).
5. 판정 기준(10pp/5pp/50%)을 **실험 전에** 고정했다. 사후 조정하지 않는다.
