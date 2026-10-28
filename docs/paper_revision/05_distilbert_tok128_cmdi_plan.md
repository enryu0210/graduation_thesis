# 05. DistilBERT 128 토큰판(원 설정) 측정 계획 · "공격 종류 구분" 서사 정리 (2026-10-28)

> 상위 문서: `docs/14_cascade_tradeoff_gate_roadmap.md` §8.34~§8.36 · 판정 기준 `docs/paper_revision/03_prereg_v2.md` §E12
> 목적 (1) WAMM'25 원 설정(128 토큰)으로 DistilBERT 를 다시 재구현해, 512 토큰 판본과 함께 보고한다.
> 목적 (2) 기존 실험 전반의 약한 고리가 "공격 종류 구분(특히 CmdI)"이었고 DistilBERT 가 SQLi→CmdI 혼동을 크게 줄였다는 내용을 논문에 넣을 수 있는 강도로 정리한다.
> ⚠️ `04_next_actions_plan.md`(미추적)는 51.6배·"Q4 답 예" 등 정정 전 수치가 남아 있다 — 이 문서가 DL-A 관련 후속 계획을 대신한다.

---

## 0. 한눈에

| 항목 | 내용 |
|---|---|
| 새 학습 | DistilBERT 128 토큰 · seed 42·43 각 5-fold (`--exclude-test --save-probs`, 나머지 설정은 E10·E11 그대로) |
| 새 지연 측정 | 같은 실행 안에서 512·128 토큰을 번갈아 잰다 (CPU 스레드 1·배치 1) |
| 학습 없는 분석 | 128 vs 512 오답 비교 · **캐스케이드가 DistilBERT 의 CmdI 이득을 실제로 물려받는지** |
| 판정 | `03_prereg_v2.md` §E12 — **학습 전에 커밋** |
| 주 비교 대상 | **512 토큰 판본이 그대로 주 기준**이다. 128 판본은 "원 설정 재현" 보조 결과다 |

---

## 1. 왜 128 토큰판을 재는가

1. **공정성 질문을 막는다.** 지금 재구현(E10·E11)은 512 토큰·Adam 이라 원 논문(128 토큰·AdamW)과 다르다. 심사에서 "원 논문 설정이 아니다"라는 지적이 나오면, 원 설정 판본도 쟀다고 답할 수 있어야 한다.
2. **CmdI 이득과 직접 이어진다.** 이 풀에서 128 토큰이면 전체 6.9% 가 잘리는데, **CmdI 26.5% · CodeInjection 24.7%** 로 주입 클래스에 몰린다(사전 문서 E10, 2026-10-25 셈). DistilBERT 의 이득이 CmdI 구분(§3)이라면, 128 토큰은 바로 그 이득을 깎을 수 있다 → "긴 문맥이 CmdI 구분에 필요한가"를 같은 실험으로 본다.
3. **지연 차이는 작을 것이다(가설).** 지연 측정은 패딩 없이 실제 길이로 토큰화한다(`dl_a.py` 의 `tok(text, truncation=True, ...)`, 배치 1). 토큰 수 p50 54 · p99 189 라 128 에서 잘리는 6.9% 만 짧아진다. 학습 시간은 패딩 길이가 1/4 이라 크게 준다.

### 1.1 ⚠️ 이 실험으로 하면 안 되는 것

- **"128 기준이면 캐스케이드가 이긴다"로 갈아타지 않는다.** 128 판본은 비교 상대를 약하게 만드는 쪽이다. 약한 기준선을 골라 이긴 것으로 보이면 체리피킹이다. E11(512, seed 43 재현 실패)은 그대로 보고하고, 128 결과는 **나란히** 적는다.
- **AdamW 까지 바꾸지 않는다.** 이번에는 토큰 길이 하나만 바꾼다(변수 하나씩). 옵티마이저 차이는 한계로 계속 병기한다.
- 128 결과로 E10·E11 판정을 고치지 않는다.

---

## 2. 실행 계획

### 2.1 코드 변경 지점 (착수 시)

| 파일 | 바꿀 것 | 이유 |
|---|---|---|
| `src/models/text_models.py` | `DISTILBERT_MAX_TOKENS` 는 기본값(512)으로 두고, 인코딩 함수가 이미 받는 `max_len` 을 호출부에서 넘기게 한다 | 기본값을 바꾸면 기존 512 산출물과 섞인다 |
| `src/eval/cross_validate.py` | `--max-tokens` 인자(기본 512). 토큰 모델일 때만 쓰고, 다른 모델에 주면 에러 | 바이트 모델의 `--max-len` 과 헷갈리지 않게 |
| `src/models/tagging.py` (`build_tag`) | 512 가 아닐 때만 `_tok{N}` 접미사 → `srbh_4class_distilbert_raw_lr2e-05_tok128_sealed[_s43]` | ⚠️ tag 규칙: 하이퍼파라미터를 반영하지 않으면 512 산출물을 덮어쓴다(커밋 5eede2f 사고). 기본값이면 생략해 기존 파일명과 호환 |
| `src/eval/dl_a.py` | `--max-tokens` 인자. `--latency` 는 **512·128 을 같은 실행에서 번갈아** 재서 한 JSON 에 함께 저장 · `--analyze` 는 tag 로 입력 분기 | 지연 배수는 같은 실행 값끼리만 비교한다(실행 간 4~6% 표류) |
| `src/eval/dl_a_errors.py` | (a) 128 판본 비교 열 · (b) **캐스케이드(B0 운영점) 예측의 혼동 행렬** · (c) 잘린 행만 따로 본 정답률 | §3.3 의 미측정 항목 |
| `tests/` | `_tok128` tag 생성 · 512 일 때 접미사 없음을 고정하는 테스트 1개 | 덮어쓰기 사고를 테스트로 막는다 |

- ⚠️ `distilbert` 는 `cross_validate.TOKEN_MODELS` 에만 있고 `train.TEXT_MODELS` 에는 일부러 없다 → `train.py` 는 건드리지 않는다. 단 tag 규칙은 `train.py`·`cross_validate.py` 가 공유하므로 `build_tag` 변경 뒤 `train.py` 경로의 tag 가 그대로인지 테스트로 확인한다.

### 2.2 실행 순서 (데스크톱 — GPU 필요)

```bash
# 0. 사전 점검: 원격 동기화 · GPU · 살아 있는 파이썬 프로세스
git fetch && git log --oneline HEAD..origin/main
PYTHONIOENCODING=utf-8 python -c "import torch; print(torch.cuda.is_available())"
powershell -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\""

# 1. 사전 판정 기준(03_prereg_v2.md §E12) 커밋 — 학습보다 먼저

# 2. 학습 (GPU, 순차 실행 — 동시에 돌리지 말 것)
PYTHONIOENCODING=utf-8 python -u src/eval/cross_validate.py --model distilbert --track srbh_4class \
  --exclude-test --save-probs --epochs 3 --batch-size 16 --lr 2e-5 --max-tokens 128
PYTHONIOENCODING=utf-8 python -u src/eval/cross_validate.py --model distilbert --track srbh_4class \
  --exclude-test --save-probs --epochs 3 --batch-size 16 --lr 2e-5 --max-tokens 128 --seed 43

# 3. 지연 (CPU — 학습·pytest 가 끝난 조용한 기기에서, 512·128 번갈아)
PYTHONIOENCODING=utf-8 python -u src/eval/dl_a.py --latency --max-tokens 512,128

# 4. 분석 (CPU, 수 초)
PYTHONIOENCODING=utf-8 python src/eval/dl_a.py --analyze --max-tokens 128
PYTHONIOENCODING=utf-8 python src/eval/dl_a.py --analyze --max-tokens 128 --seed 43
PYTHONIOENCODING=utf-8 python src/eval/dl_a_errors.py --seeds 42 43

# 5. 테스트
PYTHONIOENCODING=utf-8 python -m pytest tests/ -q
```

- 예상 소요: 512 판본이 fold 당 약 21~23분이었다. 128 은 그보다 짧을 것으로 본다(미측정). 두 seed 를 합쳐 넉넉히 4시간을 잡는다.
- 완료 감지는 산출 JSON 으로 한다: `until [ -f experiments/results/cv_srbh_4class_distilbert_raw_lr2e-05_tok128_sealed.json ]; do sleep 20; done`
- 결과 기록 위치: `docs/14` §8.37(새 절). 숫자는 `results_manifest.md` 에도 출처를 적는다.

---

## 3. "공격 종류 구분" 서사 — 논문에 넣을 내용과 강도

### 3.1 기존 실험의 공통 약한 고리 = CmdI (공격 "여부"가 아니라 "종류")

모델·입력이 달라도 클래스별 F1 에서 가장 낮은 것은 계속 CommandInjection 이었다.

| 실험 | 모델 | CmdI F1 | 다른 클래스 | 출처 |
|---|---|---:|---|---|
| E2 단계1 (train→val) | TF-IDF char 1–3 + LogReg (F2) | **0.785** | CodeInj 0.993 · SQLi 0.990 · Normal 0.985 | docs/13 §E2 단계1 표 C |
| E2 단계2 (5-fold) | RGB CNN, F2 | **0.757** (F3 0.852) | — | docs/13 §E2 단계2 표 |
| §8.7 (5-fold, class weight 1차) | RGB CNN 1차 | **0.756** | CodeInj 0.981 · Normal 0.889 · SQLi 0.952 | docs/14 §8.7 |
| 같은 분석의 2차 | char-CNN | **0.968** | 0.991~0.998 | docs/14 §8.7 |
| E10·E11 (5-fold, sealed) | char-CNN 단독 | **0.9682 / 0.9618** (seed 42/43) | 0.1%p 안팎 차이 | docs/14 §8.36 |

- 이 약한 고리는 게이트에도 그대로 드러났다. T1 운영점(1차 RGB CNN `_bal`, 넘김 50%)의 클래스별 넘김률이 **CmdI 83.7%** 로 가장 높았고(CodeInj 13.7%, docs/14 §8.3), 이것이 클래스별 τ(G2)를 시도한 근거였다(§8.5).
- Macro-F1 은 소수 클래스에 민감하다. CmdI 오답 1건을 고칠 때의 Macro-F1 이득이 SQLi·Normal 의 약 6배다(docs/14 §8.7). 그래서 **모델 간 Macro-F1 차이의 대부분이 CmdI 에서 나온다.**

### 3.2 DistilBERT 가 줄인 것: SQLi → CmdI 혼동

5 fold 의 test 예측을 합쳐 char-CNN 단독(a)과 DistilBERT 단독(b)을 같은 요청끼리 비교했다(`dl_a_errors.py`, docs/14 §8.36).

| 오답 종류 (5 fold 합산) | seed 42: char-CNN → DistilBERT | seed 43 |
|---|---|---|
| **SQLi 를 CmdI 로 착각** | **192 → 99 (−48%)** | **252 → 2 (−99%)** |
| 공격 종류 혼동 전체 (차단은 됨) | 266 → 167 (−37%) | 359 → 59 (−84%) |
| CmdI F1 | 0.9682 → 0.9784 | 0.9618 → 0.9903 |
| 차단 실패 (공격 → Normal) | 791 → 736 (−7%) | 839 → 777 (−7%) |
| 정상 오탐 (Normal → 공격) | 206 → 245 (+19%) | 140 → 118 (−16%) |

### 3.3 쓸 수 있는 문장과 쓰면 안 되는 문장

✅ 쓸 수 있다
- "기존 실험에서 모델·입력과 무관하게 가장 약한 클래스는 CommandInjection 이었다(TF-IDF 0.785, RGB CNN 0.757, char-CNN 0.962~0.968). 즉 어려움은 공격 여부보다 **공격 종류 구분**에 있었다."
- "DistilBERT 는 char-CNN 이 SQLi 를 CmdI 로 착각하던 오답을 seed 42 에서 절반(192→99건), seed 43 에서 거의 전부(252→2건) 없앴고, CmdI F1 을 0.978·0.990 으로 올렸다."
- "반면 차단 실패는 두 seed 모두 약 7% 줄었을 뿐이다. DistilBERT 의 이득은 탐지 여부가 아니라 공격 종류 구분에 있다."

❌ 쓰면 안 된다 (이유)
- "DistilBERT 로 SQLi–CmdI 혼동 문제를 **해결했다**" (단서 없이) — seed 42 에서는 절반이 남았다. seed 43 의 −99% 만 보면 과장이다. "크게 줄였다 / seed 43 에서는 거의 사라졌다"로 쓴다.
- "**캐스케이드가** 혼동을 해결했다" — 위 수치는 **DistilBERT 단독** 값이다. 캐스케이드는 요청의 약 7% 만 DistilBERT 로 넘기므로, 이 이득을 얼마나 물려받는지는 **미측정**이다(§2.1 `dl_a_errors.py` (b) 로 잰다). 캐스케이드 F1 이 char-CNN 보다 +0.28·+0.53%p(비유의)만 올랐다는 점을 보면 상당 부분을 놓칠 가능성이 있다.
- "DistilBERT 가 공격을 더 잘 막는다" — 차단 실패 감소는 7% 이고, 정상 오탐은 seed 에 따라 방향이 바뀐다.
- 위 분석은 **E11 판정 뒤에 고른 사후 분석**이다(docs/14 §8.36). 본문에서는 "사후 분석"이라고 밝히고, E11 판정(seed 43 재현 실패)을 뒤집거나 보강하는 데 쓰지 않는다.

### 3.4 왜 DistilBERT 가 이 혼동을 줄였나 — 가설 (미검증)

- char-CNN 의 합성곱 창은 k3·k5·k7 바이트라 짧은 국소 패턴(`;`, `|`, `' or` 등)에 반응한다. SQLi 와 CmdI 는 따옴표·세미콜론·공백 우회 같은 국소 기호를 공유한다. DistilBERT 는 서브워드 토큰과 전 구간 어텐션으로 `select`·`union` 같은 키워드와 `cat`·`wget` 같은 명령어를 **요청 전체 문맥에서** 구분할 수 있다.
- 이 가설이 맞다면 문맥을 자르는 128 토큰 판본에서 CmdI 이득이 줄어야 한다 → §E12-2 가 이를 본다. 라벨 쪽 원인도 배제되지 않았다(SR-BH Normal 의 스캐너 탐침 오라벨, docs/14 §8.10). CmdI 는 템플릿 공유 비율이 14.7% 로 다른 클래스(60~71%)보다 훨씬 낮아(docs/14 §8.27), 외워서 맞히기 어려운 클래스라는 점도 함께 적는다.
- 논문에는 "가설"로만 쓰고, 어텐션 시각화 같은 근거 없이 원인으로 단정하지 않는다.

---

## 4. 결과 해석 미리 정해 두기 (어느 쪽이 나와도 쓸 문장)

| 결과 | 쓸 문장 |
|---|---|
| 128 이 512 보다 낮고 CmdI 하락이 가장 크다 | "원 설정(128 토큰)은 주입 클래스의 약 25% 를 잘라 CmdI 구분 이득을 깎는다. 길이 제한이 없는 문자 모델과 긴 문맥 트랜스포머의 조합이 이 데이터에 맞다." 단, 512 를 주 기준으로 쓴 E11 결론은 그대로 둔다 |
| 128 과 512 의 차이가 확인되지 않는다 | "토큰 길이는 이 데이터에서 DistilBERT 성능을 바꾸지 않았다. 원 논문 대비 수치 차이(0.8547 vs 0.99대)는 토큰 길이가 아니라 클래스 구성(13종 vs 4분류)과 데이터 정제 차이로 설명된다" |
| 캐스케이드가 CmdI 이득을 대부분 물려받는다 | "캐스케이드는 넘긴 7% 안에서 DistilBERT 의 공격 종류 구분 이득을 가져온다" — 단 사후 분석 표기 |
| 캐스케이드가 CmdI 이득을 거의 못 물려받는다 | "DistilBERT 가 고친 오답 상당수를 char-CNN 이 확신하며 틀려 넘김이 일어나지 않는다" → 클래스별 τ(G2)를 DL-A 에 거는 사전 등록 확인(seed 44·45)의 근거로만 쓴다 |

---

## 5. 체크리스트

1. [ ] `03_prereg_v2.md` §E12 검토 → **커밋·푸시 (학습 전)**
2. [ ] §2.1 코드 변경 + 테스트 (`pytest tests/ -q` 통과)
3. [ ] seed 42 → seed 43 학습 (순차)
4. [ ] 지연 512·128 같은 실행에서 측정
5. [ ] `dl_a.py --analyze`·`dl_a_errors.py` 실행
6. [ ] `docs/14` §8.37 기록 · `results_manifest.md` · `CLAUDE.md` 요약 갱신
7. [ ] 초안(`draft_kiisc_submission.html`)에 §3.3 의 ✅ 문장만 반영. 제목도 2026-10-27 판으로 교체
