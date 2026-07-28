# Phase 5 — 회피 공격 설계/결정 문서 (RQ2: WAF 우회 취약성)

> 이 문서는 사람이 읽는 **설계·의사결정 기록**입니다.
> 실제 공격 수치는 (구현 후) `experiments/results/evasion_*.json` 과
> `docs/figures/attacks/` 에서 확인하세요.

RQ2: *"이미지 기반 탐지 모델은 실제 공격자가 WAF를 우회할 때 쓰는 회피 기법
(인코딩·주석 삽입·대소문자 변형 등)에 얼마나 취약한가?"*

핵심 관점: Phase 4(RQ1)의 clean 지표는 5개 모델 모두 매우 높았다(Macro-F1 0.95~0.996).
**RQ2는 "그 높은 점수가 실제 공격 앞에서 유지되는가"를 깨보는 단계**다. Phase 4 7절에서
실측한 "표면 토큰 shortcut" 편향이 여기서 정면으로 시험대에 오른다.

---

## 1. Phase 4가 넘긴 결정 — Phase 5 확정 내용

| 결정 | 확정 | 근거 |
|---|---|---|
| 공격 대상 트랙 | **`payload_4class_csicnorm` / raw / side=48** | Normal 을 CSIC 실트래픽으로 교체한 트랙(04문서 §7.1). benign-evasion(공격→Normal) 정의가 현실성을 가지려면 Normal 이 실제 HTTP 트래픽이어야 함. RQ1 도 이 트랙에서 재학습해 공격 대상 모델로 사용 |
| 공격 대상 모델 | **5종 전부**(제안 CNN + TF-IDF·charCNN·BiLSTM·RF) | 이미지 CNN만 공격하면 "이미지가 약한지 텍스트가 약한지" 비교 불가. RQ2의 진짜 질문은 **표현방식별 상대적 취약성** |
| 공격 대상 샘플 | **test 셋의 공격 3종만**(SQLi 8,592 / XSS 6,113 / CmdI 7,202) | Normal은 회피 대상이 아님(정상→정상은 공격이 아님) |
| 전처리 경로 | **raw(디코딩 미적용) 고정** | RQ1 주 실험이 raw. 아래 3절 참조 — 인코딩 공격의 성패가 여기서 갈림 |

---

## 2. 위협 모델 (Threat Model) — 명확히 못 박기

논문에서 "무슨 가정의 공격이냐"를 흐리면 심사에서 반드시 지적당한다. 두 축으로 분리한다.

### 2.1 지식 수준 (attacker knowledge)
- **Problem-space 공격 = 블랙박스 / 모델 무관(transfer)**
  공격자는 페이로드 문자열만 변형한다. 모델 내부(가중치·구조)를 몰라도 된다.
  → **똑같은 변형 페이로드 집합**을 5개 모델 각각에 통과시켜 ASR을 비교한다.
  이게 공정 비교의 핵심: 모델마다 다른 공격을 쓰면 취약성 비교가 왜곡된다.
- **Feature-space 공격 = 화이트박스**
  FGSM/PGD는 모델의 gradient가 필요하므로 **제안 CNN(미분 가능)에만** 깨끗하게 적용된다.
  TF-IDF/트리 계열은 gradient가 없어 대상에서 제외(한계로 서술).

### 2.2 "회피 성공"의 정의 (다중 클래스라 반드시 정의해야 함) ⚠️
이진 분류였다면 "공격→정상"이 자명하지만, 우리는 4-class다. 두 지표를 **분리 측정**한다.

1. **주 지표 — Benign-evasion (보안적으로 의미 있는 성공)**
   공격 페이로드(SQLi/XSS/CmdI)가 **`Normal`로 예측**되면 성공.
   = WAF가 공격을 정상 트래픽으로 흘려보냄 = 실제 침해로 이어지는 유일하게 위험한 실패.
2. **보조 지표 — Any-misclassification (분석용, 약한 성공)**
   `예측 ≠ 진짜 클래스`(예: SQLi→XSS)면 성공으로 카운트.
   → 탐지기의 "동요"는 보여주지만 WAF 우회는 아님. 해석 시 명확히 구분해 보고한다.

> **주 결론은 항상 지표 1(Benign-evasion)로 낸다.** 지표 2는 모델이 흔들리는 양상을 보는 보조 렌즈.

---

## 3. 왜 raw 트랙이어야 인코딩 공격이 "산다" — 전처리와의 상호작용

RQ1 주 모델은 `text_raw`(URL/HTML 디코딩을 **하지 않은** 원문 바이트)로 학습됐다.
따라서 공격자가 URL 인코딩(`'`→`%27`)을 하면 그 바이트가 **디코딩 없이 그대로 모델에 도달**한다
→ 인코딩 공격이 유효하다.

반대로 만약 파이프라인이 요청을 디코딩한 뒤 탐지했다면(`text_decoded` 트랙),
인코딩 공격은 탐지 전에 원상복구되어 **무력화**된다.

→ 이 대비 자체가 **Phase 6(방어) 후보이자 ablation 포인트**다(설계 3장 "디코딩 적용 vs 미적용").
Phase 5는 **raw 트랙(공격에 유리한 최악 조건)**을 주 실험으로 잡아 취약성 상한을 먼저 보고,
"디코딩 정규화가 방어책이 되는가"는 Phase 6에서 다룬다.

---

## 4. Problem-space 공격 카탈로그 (주 실험) — 의미 보존이 절대 조건

**철칙: 변형 후에도 페이로드가 여전히 "동작하는 공격"이어야 한다.**
동작하지 않는 페이로드로 회피에 성공해봐야 공격으로서 무의미하다. 우리는 라이브 타깃에
실행해 검증할 수 없으므로, **"의미를 보존한다고 규칙 수준에서 보장되는 변형"만** 사용한다
(임의 문자 삽입 같은 파괴적 변형은 금지). 이 제약과 한계는 논문에 명시한다(7절).

### 4.1 공통(문법 무관) 변형
| 기법 | 예시 | 의미보존 근거 |
|---|---|---|
| URL 인코딩 / 이중 인코딩 | `'` → `%27` → `%2527` | 서버단에서 디코딩되어 원 페이로드로 환원 |
| 대소문자 랜덤화 | `SELECT` → `sElEcT` | SQL 키워드·HTML 태그는 대소문자 무관 |
| 공백 치환 | 공백 → 탭/개행/`/**/`/`%09` | 파서가 동일 토큰 경계로 인식 |

### 4.2 SQLi 전용
| 기법 | 예시 |
|---|---|
| 인라인 주석 삽입 | `UNION SELECT` → `UNION/**/SELECT` |
| 논리적 동치 치환 | `OR 1=1` → `OR 2=2-1`, `OR 'a'='a'` |
| 버전 주석 | `/*!50000UNION*/` (MySQL 조건부 실행 주석) |

### 4.3 XSS 전용
| 기법 | 예시 |
|---|---|
| HTML 엔티티 인코딩 | `<` → `&lt;` / `&#60;` / `&#x3c;` |
| JS 문자열 난독화 | `alert(1)` → `String.fromCharCode(97,108,101,114,116)(1)` |
| 태그 대소문자·속성 변형 | `<script>` → `<ScRiPt>`, `onerror` 이벤트 치환 |

### 4.4 Command Injection 전용
| 기법 | 예시 |
|---|---|
| 셸 변수 삽입 | `cat` → `c$@at`, `c""at`, `c\at` |
| 구분자 치환 | `;` ↔ `|` ↔ `&&` ↔ `%0a`(개행) |
| IFS/공백 우회 | 공백 → `${IFS}` |

### 4.5 변형 예산(budget)과 탐색 전략 — 3단 구성
1. **단일 기법(isolation)**: 한 번에 한 기법만 적용 → **기법별 ASR** 산출.
   "어떤 회피가 어느 모델에 잘 통하는가"의 해상도를 얻는다.
2. **조합(stacked)**: 상위 기법 여러 개를 순차 적용(예산 k=1..5) → **예산-ASR 곡선**.
3. **탐색(GA, 선택/상한 측정)**: WAF-A-MoLE 방식의 유전 알고리즘으로 변형 조합을 탐색해
   **모델별 회피 성공 상한**을 잰다. 쿼리 예산(예: 개체 50 × 세대 20)을 고정해 공정 비교.
   → 비용이 크므로 대상 샘플을 **층화 표집(각 클래스 500건)**해 수행.

---

## 5. Feature-space 공격 (보조/비교) — 제안 CNN 한정

- **FGSM / PGD**로 48×48 이미지 픽셀을 직접 교란(ε 스윕). ART 라이브러리 사용.
- **결정적 한계(반드시 서술)**: 이렇게 만든 "적대적 이미지"는 임의의 실수 픽셀값을 가지므로
  **유효한 UTF-8 바이트 시퀀스로 역변환되지 않을 수 있다**. 즉 "실행 가능한 페이로드"가 아니다.
  → feature-space ASR은 **"이미지 표현이 이론적으로 얼마나 취약한가"의 상한**으로만 해석하고,
  실제 위협 지표는 4장의 problem-space ASR로 낸다. 이 구분이 이 연구의 방법론적 정직성.
- 비교 관점: 같은 ε에서 image CNN이 크게 무너지면 "이미지 표현의 매끄러운 결정경계" 특성을
  드러내는 근거가 된다(텍스트 모델은 이산 입력이라 동일 gradient 공격 불가 — 비대칭성 자체가 논점).

---

## 6. 공격 파이프라인 — 기존 코드 재사용이 원칙

**새 평가 로직을 만들지 않는다.** 공격은 "입력 페이로드를 바꾸는 것"일 뿐,
그 뒤 전처리→이미지화→예측→지표는 **Phase 3/4의 실제 함수를 그대로 통과**시킨다.
그래야 "같은 파이프라인, 입력만 오염"이라는 인과가 성립한다.

```
원본 공격 페이로드(text_raw, test 셋)
   → [src/attacks] 의미보존 변형 적용        ← Phase 5 신규
   → payload_to_image()  (side=48)          ← Phase 3 재사용, 그대로
   → 학습된 모델 5종 로드해 예측              ← Phase 4 checkpoint 재사용
   → src/eval/metrics.py + ASR 계산          ← Phase 4 모듈 재사용/확장
```

### 6.1 모듈 구성 (src/attacks/)
| 파일 | 책임 |
|---|---|
| `mutations.py` | 클래스별 **의미보존 변형 규칙**(4장). 순수 문자열 함수, 부작용 없음 |
| `problem_space.py` | 변형 적용(단일/조합) + GA 탐색 오케스트레이션 |
| `feature_space.py` | FGSM/PGD (제안 CNN 대상, ART 래핑) |
| `run_evasion.py` | 대상 모델 로드 → 공격 → ASR/지표 리포트 저장(CLI) |

### 6.2 산출물 규약 (Phase 4 명명 관습 계승)
- 지표: `experiments/results/evasion_{track}_{model}_{attack}.json`
  (`attack` = `single-urlenc`, `stacked-k3`, `ga`, `fgsm-eps0.05` 등)
- 그림: `docs/figures/attacks/` (모델×기법 ASR 히트맵, 예산-ASR 곡선, ε-ASR 곡선)
- 단위 테스트: `tests/test_attacks.py` — **변형의 멱등성/의미보존 스모크**
  (예: URL 디코딩하면 원문 복원되는지, 대소문자 변형이 키워드를 깨지 않는지)

---

## 7. Phase 4 편향(7절)과의 연결 — RQ2의 핵심 가설

Phase 4 7절 진단: 텍스트 모델의 높은 점수는 표면 구두점 토큰(`(`,`|`,`alert`…)에
의존한 **shortcut**일 가능성이 크다.

**→ RQ2 주 가설(H1)**: *표면 토큰에 의존하는 모델일수록 회피 공격에 더 취약할 것이다.*
- 인코딩·주석 삽입은 바로 그 토큰의 표면형을 흩뜨린다. 토큰 매칭에 기댄 TF-IDF/charCNN이
  **clean에선 이겼지만 회피에선 더 크게 무너지는** 그림이 나오면, RQ1의 순위가 **뒤집힌다**.
- 만약 제안 이미지 CNN이 (clean 점수는 낮았어도) 회피 저항성이 상대적으로 높다면,
  "clean 지표만으로 모델을 고르면 안 된다"는 이 논문의 핵심 메시지가 실측으로 완성된다.
- 물론 **반대 결과(이미지 CNN이 더 취약)도 유효한 기여**다. 어느 쪽이든 "표현 방식과
  강건성의 관계"라는 논문의 축을 데이터로 채운다. 결과를 미리 재단하지 않는다.

---

## 8. 실행 순서 (구현 후 재현)

```bash
# 0) 사전: Phase 4 학습 완료(experiments/checkpoints/*.pt 존재)

# 1) 단일 기법별 ASR (모든 모델 × 모든 기법)
python src/attacks/run_evasion.py --attack single --track payload_4class

# 2) 조합 예산 곡선 (k=1..5)
python src/attacks/run_evasion.py --attack stacked --budget 5

# 3) GA 상한 (표집 500/클래스, 대상 모델 지정)
python src/attacks/run_evasion.py --attack ga --model cnn --samples 500

# 4) feature-space (제안 CNN 한정, ε 스윕)
python src/attacks/run_evasion.py --attack fgsm --model cnn --eps 0.01,0.03,0.05

# 스모크(코드 점검용): --smoke --limit 200
```

---

## 9. 리스크 / 열린 결정 (다음 단계로 넘김)

- [ ] **의미보존 검증의 한계**: 라이브 타깃이 없어 "여전히 동작하는 공격"임을 실행으로
  증명하지 못한다. 규칙 수준 보장 + 소수 표본 수동 검토로 갈음하고 한계로 명시.
- [ ] **GA 쿼리 예산**을 얼마로 고정할지(공정성 vs 계산비용) — 500샘플×세대20이 현실적 후보.
- [x] Normal 편향(Phase 4 7절) → **해결**: Normal 을 CSIC 실트래픽으로 교체한
  `payload_4class_csicnorm` 트랙 확보(04문서 §7.1). 단, clean 점수는 여전히 포화(표면토큰 본질)
  → 회피가 쉽게 나오는 건 데이터가 아니라 과제 특성이며, 그 자체가 RQ2 의 논지.
- [ ] feature-space 결과를 논문 본문에 넣을지/부록으로 뺄지(invertibility 한계 때문).
- [x] 회피에 성공한 변형 페이로드 집합을 **adversarial training 학습 소스**로 그대로 넘기는
  인터페이스 확정 → **docs/10(Phase 11, RQ3)** 에서 확정. 변형본을 파일로 넘기지 않고
  `mutations.py` 를 학습 시점에 재호출하는 **치환식 온더플라이 증강**으로 정했다
  (docs/10 §7.2). 산출물 포맷을 맞출 필요 자체가 사라짐.
- [ ] **RQ3 의 주 지표 이동** — 본 문서 §13.2 의 "benign-evasion 5모델 전부 ~0" 실측 때문에
  원 RQ3 는 바닥 효과에 걸린다. 주 지표를 any-misclass 로 옮기는 근거·판정 기준은 docs/10 §1.
- [ ] (운영) 브랜치 정리 — 여전히 `phase1-data-acquisition`에 전 Phase가 쌓여 있음.

---

## 10. 요약 — 이 문서가 확정한 것

1. 공격 대상 = **payload_4class/raw/side=48**, **5개 모델 전부**, **공격 3종 test 샘플**.
2. "회피 성공"은 **Benign-evasion(→Normal 예측)**을 주 지표로, any-misclass를 보조로.
3. **Problem-space(의미보존 규칙)**를 주 실험(블랙박스·모델 무관), **feature-space(FGSM/PGD)**는
   제안 CNN 한정 보조 — invertibility 한계 명시.
4. 공격은 **Phase 3/4 실제 파이프라인을 그대로 통과**시키고, 지표 모듈을 재사용한다.
5. 핵심 가설: **표면 토큰 의존 모델이 더 취약** → RQ1 clean 순위가 뒤집히는지 검증.

---

## 11. 진행 현황 및 다음 작업 (2026-07-06 기준)

### ✅ 완료 (A) — RQ2 데이터·학습 준비
- `payload_4class_csicnorm` 트랙 확보(Normal=CSIC 실트래픽, 04문서 §7.1).
- 이미지 데이터셋 빌드 완료: `data/images/payload_4class_csicnorm_{split}_text_raw_48.npz`.
- `train.py` 에 **`--balance` 옵션 추가**(train 클래스 균형 언더샘플링, val/test 는 실분포 유지).
  균형화 산출물은 tag 에 `_bal` 접미사가 붙어 기존 결과와 안 섞임. (스모크로 두 경로 무결성 확인)

### ⬜ 내일 할 일 — B: RQ2 공격 코드 착수
- `src/attacks/mutations.py` 부터: 4장 카탈로그의 **의미보존 변형 규칙**(SQLi/XSS/CmdI 별)
  순수 문자열 함수로 구현 + `tests/test_attacks.py`(멱등성·의미보존 스모크).
- 이어서 `problem_space.py`(단일/조합/GA) → `run_evasion.py`(모델 로드·ASR 리포트).
- 대상 모델은 아래 C 로 재학습된 `*_bal` 체크포인트를 연결.

### ⬜ 내일 할 일 — C: 불균형 처리 방식 확정 + 5모델 재학습
- **결정 필요**: RQ2 공격 대상 모델을 (a) `--balance`(언더샘플링, 현재 구현) /
  (b) 오버샘플링 / (c) class weight 만 중 무엇으로 학습할지.
  → 권장: **(a) 언더샘플링**. Normal 이 실제 예측 후보가 되어야 benign-evasion 측정이
    유효(불균형이면 가짜 강건성). 단 데이터가 ~13k 로 줄어드는 트레이드오프 존재.
- 재학습 명령(GPU 권장, torch 계열):
  ```bash
  # 이미지 데이터셋은 이미 빌드됨. 아래로 5모델 재학습(_bal tag 로 저장)
  python src/models/baseline_tfidf.py --track payload_4class_csicnorm --clf logreg   # (CPU 가능)
  python src/models/baseline_tfidf.py --track payload_4class_csicnorm --clf rf
  python src/models/train.py --model cnn     --track payload_4class_csicnorm --balance
  python src/models/train.py --model charcnn --track payload_4class_csicnorm --balance
  python src/models/train.py --model bilstm  --track payload_4class_csicnorm --balance
  ```
  ※ baseline_tfidf 는 `class_weight="balanced"` 라 `--balance` 불필요(자체 보정).
- 재학습 후: RQ1 표를 csicnorm 기준으로 재작성할지(04문서 5절 대비) 판단.

---

## 12. 실측 결과 — TF-IDF 회피 공격 (2026-07-09)

> 산출물: `experiments/results/evasion_payload_4class_csicnorm_tfidf_{logreg,rf}_{single,stacked}.json`,
> 그림 `docs/figures/attacks/evasion_{single,stacked}_payload_4class_csicnorm_tfidf_{logreg,rf}.png`.
> 대상: `payload_4class_csicnorm` / raw / test 공격 21,907건(SQLi 8,592·XSS 6,113·CmdI 7,202).
> 공격 대상 TF-IDF(char_wb 2–4gram, max_features=20k)는 `run_evasion.py` 가 clean train 으로
> 즉석 학습(clean 정확도 logreg 0.9927). CNN·charCNN·BiLSTM 은 GPU 확보 후 동일 스크립트로 확장.

### 12.1 핵심 결과 — 주 지표(benign-evasion)는 전 구간 0%

| 지표 | logreg | rf |
|---|---|---|
| **주 — benign-evasion(공격→Normal)** | **0.0000** (단일·조합 k=1–5 전부) | **0.0000** (전부) |
| 보조 — any-misclass(clean) | 0.0075 | 0.0031 |
| 보조 — any-misclass(조합 k=5) | **0.5806** | **0.3421** |

- **주 지표 = 0.** 표면 변형(인코딩·주석·대소문자·구분자)은 공격을 단 한 건도 `Normal` 로
  밀어넣지 못했다. Normal 을 CSIC 실트래픽으로 교체(§1)한 트랙에서, 실제 HTTP 정상 트래픽과
  공격 페이로드는 char n-gram 공간에서 워낙 분리돼 있어 표면 변형으로는 그 경계를 못 넘는다.
  → **WAF 우회(유일하게 위험한 실패)라는 관점에서 TF-IDF 는 이 회피군에 강건**하다.
- **보조 지표는 폭증**(clean <1% → k=5 에서 logreg 58%·rf 34%). 즉 변형이 탐지기를 크게
  흔들긴 하나, 방향이 "공격→다른 공격 클래스"(대부분 XSS 로 붕괴)라 **탐지 자체는 유지**된다.
  주 지표만 보면 "무결"처럼 보이지만 보조 지표가 모델의 실제 동요를 드러낸다(§2.2 의 두 지표
  분리 측정이 여기서 결정적).

### 12.2 어떤 변형이 통하는가 (보조 지표 기준)

- 압도적으로 **URL 인코딩 계열**: `double_url_encode`(logreg Δ+0.62)·`url_encode`(Δ+0.59).
  raw 트랙(디코딩 미적용, §3)이라 `%XX` 바이트가 그대로 도달 → char n-gram 분포가 XSS 쪽으로
  이동. 반면 `space_to_tab`·클래스 전용 변형(주석/엔티티/IFS)은 거의 무효(Δ≈0).
- **logreg > rf**: 선형 모델이 인코딩 변형에 더 민감(any-misclass 58% vs 34%). 트리 앙상블이
  표면 n-gram 교란에 상대적으로 견고.

### 12.3 구현 메모 / 문서와의 차이

- `run_evasion.py` 는 §2.2 요구대로 **주·보조 두 지표를 함께** 기록하도록 확장했다(최초 구현은
  주 지표만 기록 → 주 지표가 전부 0 이라 "빈 결과"로 오독될 위험이 있어 보조 지표 추가).
- 그림(`evasion_stacked_*`)은 두 지표 곡선을 겹쳐 표시. 라벨은 폰트 문제로 ASCII.
- ~~**미측정(다음 단계)**: 이미지 CNN·charCNN·BiLSTM 대상 동일 실험(GPU 필요)~~ → **§13 에서 완료**.

---

## 13. 실측 결과 — 5모델 전면 회피 실험 (2026-07-11, GPU: RTX 4080 SUPER)

> §12 의 TF-IDF 2종에 더해 **이미지 CNN·charCNN·BiLSTM 3종을 같은 스크립트로** 실측해
> RQ2 의 핵심 질문("표현방식별 상대 취약성")을 완성했다. 5모델 **모두 동일 트랙·동일 변형·
> 동일 지표**(docs/05 §2.2)로 평가.
> 산출물: `experiments/results/evasion_payload_4class_csicnorm_{cnn,charcnn,bilstm,tfidf_logreg,tfidf_rf}_{single,stacked}.json`,
> 그림 `docs/figures/attacks/evasion_{single,stacked}_*.png` + **통합 비교 `evasion_compare_payload_4class_csicnorm.png`(헤드라인)**.
> 대상: `payload_4class_csicnorm` / raw / test 공격 21,907건. torch 3종은 `--balance` 재학습본(_bal)을 공격 대상으로 사용.

### 13.1 재현 절차 (이번 세션에서 실제 실행한 순서)
```bash
# 0) 데이터 재생성(gitignore 라 매번 재생성) — 인자 없이 전체(노트 유지)
python src/data/preprocess.py
python src/imaging/build_image_dataset.py --track payload_4class_csicnorm
# 1) torch 3모델 csicnorm 균형 재학습(체크포인트 _bal 저장)
python src/models/train.py --model cnn     --track payload_4class_csicnorm --balance
python src/models/train.py --model charcnn --track payload_4class_csicnorm --balance
python src/models/train.py --model bilstm  --track payload_4class_csicnorm --balance
# 2) 5모델 회피 실행(모델만 교체 — 같은 공격/파이프라인)
python src/attacks/run_evasion.py --model cnn     --track payload_4class_csicnorm
python src/attacks/run_evasion.py --model charcnn --track payload_4class_csicnorm
python src/attacks/run_evasion.py --model bilstm  --track payload_4class_csicnorm
python src/attacks/run_evasion.py --model tfidf_logreg --track payload_4class_csicnorm
python src/attacks/run_evasion.py --model tfidf_rf     --track payload_4class_csicnorm
# 3) 통합 비교 그림
python src/attacks/compare_evasion.py --track payload_4class_csicnorm
```

### 13.2 핵심 결과 — ① 주 지표(benign-evasion)는 5모델 전부 ~0

| 모델 | 입력 | clean macro-F1(참고) | benign-evasion (k=0→k=5) | any-misclass (k=5) |
|---|---|---|---|---|
| 제안 CNN | 48×48 이미지 | 0.967 | 0.0010 → 0.0022 | **0.594** |
| char-CNN | 바이트 시퀀스 | 0.995 | 0.0004 → 0.0001 | **0.224** |
| BiLSTM | 바이트 시퀀스 | 0.984 | 0.0008 → 0.0006 | 0.466 |
| TF-IDF+LogReg | 문자열 | 0.993 | 0.0000 → 0.0000 | 0.576 |
| TF-IDF+RF | 문자열 | 0.994 | 0.0000 → 0.0000 | 0.332 |

- **주 지표 = 사실상 0(모든 모델·전 예산).** 의미보존 표면 변형(인코딩·주석·대소문자·구분자)은
  **어떤 표현방식에서도** 공격을 `Normal` 로 밀어넣지 못한다. CSIC 실트래픽 Normal 과 공격
  페이로드가 이미지·시퀀스·n-gram **모든 표현 공간에서** 워낙 분리돼 있어, 표면 변형으로는
  그 경계를 못 넘는다. → **WAF 우회(유일하게 위험한 실패)라는 관점에서 5모델 전부 이 회피군에 강건.**

### 13.3 핵심 결과 — ② 보조 지표(any-misclass)로 본 표현방식별 '동요' 순위

표면 변형이 탐지기를 흔드는 정도(k=5)는 표현방식마다 크게 다르다:

**제안 CNN(0.594) ≳ TF-IDF LogReg(0.576) > BiLSTM(0.466) > TF-IDF RF(0.332) > char-CNN(0.224)**

- **가설 H1(§7)은 이 데이터에서 성립하지 않는다(오히려 반대).** H1 은 *"표면 토큰 의존
  모델(TF-IDF/charCNN)이 더 취약 → 이미지 CNN 이 상대적으로 강건할 수 있다"* 였다. 실측은
  **제안 이미지 CNN 이 any-misclass 로는 가장 크게 흔들린다**(0.594, LogReg 보다도 높음). 즉
  "이미지 표현이 회피에 더 강하다"는 **이 데이터에서 지지되지 않는다.**
- 단, **이 '동요'는 보안적 실패가 아니다**: 제안 CNN 도 benign-evasion 은 ~0 이라, 변형된 공격이
  Normal 이 아니라 **다른 공격 클래스로** 흩어질 뿐 탐지 자체는 유지된다(대부분 XSS 로 붕괴).
- **char-CNN 이 clean 최고이면서 회피 최저 동요**(0.995 / 0.224)로 이 회피군에는 가장 견고.
  학습형 바이트 임베딩이 TF-IDF 의 생(raw) char n-gram 표면 매칭보다 인코딩 교란에 덜 민감했다.
- 변형별로는 **URL 인코딩 계열이 압도적**(raw 트랙이라 `%XX` 바이트가 그대로 도달). 클래스 전용
  변형(주석/엔티티/IFS)은 대부분 무효 — §12 의 TF-IDF 관찰과 일치하며 표현방식 무관하게 재현됐다.

### 13.4 RQ2 에 대한 정직한 중간 결론

- **RQ1 clean 순위는 benign-evasion 관점에서 뒤집히지 않았다** — 모두 안전(0). 그러나 이는
  모델 강건성보다 **과제/데이터 특성**(Normal=실트래픽이 전 표현공간에서 잘 분리됨)을 더 반영한다.
- **표면 변형만으로는 이 트랙에서 WAF 우회가 안 된다**가 강건한 실측 사실. 따라서 "정말 뚫리는가"는
  **Normal 을 모방하도록 방향을 잡는 더 강한 공격**이 필요 → 다음 단계: (a) GA 탐색(§4.5-③,
  benign 방향 목적함수), (b) feature-space FGSM/PGD(§5, 제안 CNN 한정, invertibility 한계 명시).
- 논문 서술 축: "표면 변형에 대한 benign-evasion 강건성은 표현방식 무관하게 확보되나,
  any-misclass 동요는 표현방식별로 최대 2.6배 차이(0.224~0.594) — 제안 이미지 CNN 이 오히려 최상위."

### 13.5 구현 메모
- `run_evasion.py` 를 **모델 무관 `predict(list[str])→인덱스` 콜러블**로 추상화해 확장(TF-IDF 즉석
  학습 / torch 체크포인트 로드 두 경로를 한 지점에서만 분기). 변형 텍스트는 **파일 재빌드 없이**
  그 자리에서 `payload_to_image`(CNN) 또는 `encode_byte_matrix`(charCNN/BiLSTM)로 변환해 예측
  → docs/05 §6 "같은 파이프라인, 입력만 오염" 원칙 유지.
- **BiLSTM OOM 주의**: 긴 시퀀스(max_len=2304)라 예측 배치를 크게 잡으면(1024) CUDA OOM(30GiB+).
  → bilstm 만 배치 128 로 낮춤(conv 계열은 1024 유지). 코드에 근거 주석 명시.
- `compare_evasion.py` 신규: 5모델 stacked JSON → 2패널(주/보조) 통합 그림. 라벨은 폰트 문제로 ASCII.
- **미측정(다음 단계)**: GA 탐색(benign 방향), feature-space FGSM/PGD → RQ2 상한 측정 후 Phase 6(RQ3 방어)로.
