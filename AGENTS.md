# AGENTS.md — Codex 작업 지침 (웹 공격 페이로드 이미지화 CNN 탐지 / 졸업논문)

> 이 파일은 **Codex 가 자동으로 읽는 유일한 프로젝트 컨텍스트**다.
> 구조를 바꾸는 커밋에서는 이 파일도 같이 갱신할 것 — 낡으면 Codex 가 조용히 틀린 가정 위에서 코딩한다.
> 사람용 정본은 `CLAUDE.md`(코드 관례·함정)와 `docs/CURRENT_BASELINE.md`(연구 현행 기준).

---

## 0. 이 저장소가 무엇인가

학부 졸업논문. 웹 공격 페이로드(SQLi/XSS/Command Injection)를 **48×48 RGB 바이트 이미지**로
변환해 탐지하고, 그 위에서 회피 공격·방어·비용 기반 공격까지 다룬다.

**제안 모델은 캐스케이드**다 — 1차 RGB CNN 이 전량을 보고, 확신도가 임계값 τ 미만인 소수만
2차 char-CNN 으로 넘긴다(추가 학습 없음).

⚠️ **이 저장소의 산출물은 논문에 실린다.** 지표 하나가 틀리면 논문이 틀린다.
"아마 맞을 것"으로 넘어가지 말고, 안 돌려봤으면 안 돌려봤다고 쓸 것.

---

## 1. 역할 경계 (반드시 지킬 것)

| 하는 것 | 하지 않는 것 |
|---|---|
| `src/`, `tests/` 의 코드 작성·수정 | **`docs/` 수정** — 문서는 Claude 소관. `docs/handoff/RESULT.md` 만 예외 |
| 검증 명령 실행과 출력 첨부 | **`git commit` / `git push`** — 사용자가 직접 지시할 때만 |
| `docs/handoff/RESULT.md` 작성 | **요청 범위 밖 인접 코드 정리·리팩터링** |
| 요청된 파일만 변경 | `README.md`, `CLAUDE.md`, `AGENTS.md` 수정 |

⚠️ **요청하지 않은 개선을 끼워 넣지 말 것.** 눈에 거슬리는 코드를 발견하면 고치지 말고
`RESULT.md` 의 "발견했지만 건드리지 않은 것" 절에 적어라. 판단은 사람이 한다.

---

## 2. 검증 명령 (전문 그대로 복사해 쓸 것)

```bash
# 필수 — 모든 코드 변경 후. 2026-08-25 기준 168 passed, 약 5초. GPU 불필요.
PYTHONIOENCODING=utf-8 python -m pytest tests/ -q
```

```bash
# 특정 영역만 빠르게 볼 때
PYTHONIOENCODING=utf-8 python -m pytest tests/test_cascade.py -q
```

- **`PYTHONIOENCODING=utf-8` 를 빼지 말 것.** Windows 한글 콘솔(cp949)에서 파이썬의
  비-ASCII 출력이 `UnicodeEncodeError` 로 죽는다. 코드 문제가 아니라 콘솔 문제다.
- 테스트가 **168개보다 줄었으면** 뭔가 수집되지 않은 것이다. 통과 개수를 RESULT 에 적어라.
- 기존에 실패하던 테스트는 없다. 실패가 보이면 **네 변경 때문**이라고 가정하고 조사할 것
  ("원래 깨져 있던 것" 으로 넘기지 말 것).

---

## 3. 실행 환경 — 먼저 어느 머신인지 확인하라

이 프로젝트는 **노트북과 데스크톱 양쪽**에서 작업한다. 능력이 다르다.

```bash
PYTHONIOENCODING=utf-8 python -c "import torch; print(torch.cuda.is_available())"
```

| 결과 | 머신 | 가능한 것 |
|---|---|---|
| `True` | 데스크톱 (RTX 4080 SUPER 17GB, torch cu124, RAM 34GB) | 전면 학습 가능 |
| `False` | 노트북 (Intel Arc iGPU, torch CPU 빌드) | **학습 불가.** 테스트·정적 검증만 |

- 노트북에서 학습을 시도하지 말 것. 몇 시간 걸리다 실패한다.
  학습이 필요한 작업인데 CUDA 가 없으면 **RESULT.md 에 "이 머신에서 검증 불가"라고 적고 멈춰라.**
- `nvidia-smi` 는 PATH 에 없다. 위 파이썬 한 줄을 쓸 것.

### GPU 는 1대뿐 — 학습은 순차 실행
- 학습 작업을 **동시에 두 개 띄우지 말 것.** 재현성이 깨진다.
- 학습 중 다른 검증이 필요하면 `CUDA_VISIBLE_DEVICES=` 로 CPU 를 강제한다.
- ⚠️ **백그라운드 파이썬이 "killed" 로 보고돼도 자식 프로세스는 살아 있을 수 있다**(셸 래퍼만 죽음).
  재실행 전 반드시 생존 확인:
  ```powershell
  Get-CimInstance Win32_Process -Filter "Name='python.exe'"
  ```
- 백그라운드 파이썬은 stdout 버퍼링으로 로그가 안 보인다 → `python -u` 를 쓰거나
  산출 JSON 파일 존재 여부로 완료를 판정한다.

---

## 4. 절대 하지 말 것 (사고가 실제로 났던 것들)

### 4.1 `train.py` 를 `--smoke` 없이 로컬 검증용으로 돌리지 말 것
`docs/figures/models/cm_*.png` 는 **git 추적 대상**이다. `--smoke` 없이 돌리면 논문에 들어갈
그림을 덮어쓴다. 코드가 도는지만 보려면 반드시:
```bash
PYTHONIOENCODING=utf-8 python src/models/train.py --smoke ...
```

### 4.2 산출물 tag 에 축을 빠뜨리지 말 것
tag 는 **실험을 가르는 하이퍼파라미터를 전부 반영**해야 한다. 안 그러면 서로 다른 실험이
같은 파일명으로 저장돼 **조용히 덮어쓴다**(커밋 5eede2f 사고: RGB 조합이 전부 `_rgb` 로 저장됨).

단일 진실 소스는 `src/models/tagging.py` 의 `build_tag`. 규칙을 바꾸면
`train.py` 와 `cross_validate.py` **2곳**을 동시에 갱신한다.

⚠️ **"막아둔 조합"이 tag 버그를 가린다.** 어떤 조합이 `parser.error` 로 막혀 있으면 그 경로의
tag 결함이 드러나지 않는다. **차단을 푸는 변경을 할 때는 저장 경로의 tag 부터 점검할 것.**

### 4.3 matplotlib 라벨에 한글을 쓰지 말 것
DejaVu Sans 에 한글이 없다 → □ 로 깨지고 경고가 쏟아진다.
**그림의 라벨·범례·제목은 ASCII 만.** 한글은 콘솔 출력과 JSON 에만.

### 4.4 `.gitignore` 에 인라인 주석을 쓰지 말 것
`.gitignore` 는 **줄 전체가 패턴**이다. `experiments/checkpoints/  # 주석` 은 규칙이 무효가 되어
실제로 `.pt` 파일이 노출돼 있었다(2026-07 수정). 주석은 별도 줄에 쓰고, 규칙 추가 후 반드시:
```bash
git check-ignore -v <경로>
```

### 4.5 `liteLLM` 라이브러리를 쓰지 말 것 (보안 문제)

---

## 5. 동기화 지점 체크리스트

한 곳만 고치면 조용히 깨지는 자리들이다. 해당하는 작업이면 **전부** 고쳤는지 대조할 것.

**새 CSV 트랙 추가 → 5곳**
1. `src/data/preprocess.py` (TRACKS / REQUIRED_FILES)
2. `src/imaging/build_image_dataset.py`
3. `src/models/baseline_tfidf.py`
4. `src/eval/diagnose_payload_bias.py`
5. `src/models/train.py`

> 비-CSV 트랙(예: `ustc_flow_binary`)은 preprocess/build_image_dataset 경로를 안 탄다
> → `train.py` **한 곳만** 갱신. 위 "5곳"은 CSV 트랙 한정이다.

**새 이미지 모델 추가 → `IMAGE_MODELS` 2곳**
`src/models/train.py`, `src/eval/cross_validate.py` (트랙의 "5곳"과는 별개)

**새 RGB 채널 인코더 추가 → 2곳**
`src/imaging/build_image_dataset.py` 와 `src/models/data_image.py` 의 `_ENCODER_ABBR`
(저장 파일명과 로드 파일명이 일치해야 한다)

**tag 규칙 변경 → 2곳**
`src/models/train.py`, `src/eval/cross_validate.py`

---

## 6. 데이터 — 없으면 재생성한다

`data/processed`, `data/images` 는 `.gitignore` 대상(대용량)이라 클론 직후엔 없다.

```bash
PYTHONIOENCODING=utf-8 python src/data/preprocess.py                      # seed=42 결정론, CSV 3분할
PYTHONIOENCODING=utf-8 python src/imaging/build_image_dataset.py --track payload_4class
```

⚠️ `preprocess.py <단일트랙>` 은 `docs/03_preprocessing_notes.md` 를 **그 트랙만으로 덮어쓴다.**
노트를 유지하려면 **인자 없이 전체 실행**할 것.

⚠️ 새 `data/raw/<dataset>/` 는 자동 무시되지 않는다 → `.gitignore` 에 수동 추가.
대용량 커밋 사고를 막기 위한 것이고, 재현은 다운로드 스크립트로 보장한다.

---

## 7. 코드 스타일

- **주석과 문자열은 한국어.** 코드가 '무엇'을 하는지보다 **'왜' 이렇게 썼는지** 의도를 적을 것.
- 변수명은 직관적으로, 로직은 단순하게. "읽기 쉬운 코드"가 최우선이다.
- 파일이 길어지면 기능별로 분리한다.
- 예외 상황을 항상 고려한다 (특히 파일 부재·CUDA 부재·빈 배치).
- 모든 모델 평가는 `src/eval/metrics.py` 를 거친다. **지표를 직접 계산하지 말 것** —
  계산 방식 차이가 논문 수치를 어긋나게 만든다.

### 용어 (섞으면 실제로 사고가 난다)
- **`hybrid`** = 모델 이름 (CNN stem + Transformer, docs/08 ViT 계열)
- **`cascade`** = 배치 구조 (1차 CNN → 2차 char-CNN)
- 이 둘은 **다른 것**이다. 이름을 섞지 말 것.
- **`제안 CNN` 이라는 말은 쓰지 말 것** — (ㄱ)제안 모델과 (ㄴ)캐스케이드 1차 부품 두 뜻으로
  읽혀 실제로 혼동이 났다. `RGB CNN`(48×48×3 단일) / `gray CNN`(1채널) / `얕은 CNN`(아키텍처) 으로 쓸 것.

---

## 8. RESULT.md 작성 규칙

작업이 끝나면 `docs/handoff/RESULT.md` 에 보고한다. **이 파일은 의심하며 읽힌다.**
가장 비싼 실패는 **검증하지 않은 것을 검증했다고 보고하는 것**이다.

반드시 포함할 것:

1. **변경한 파일 목록** — 경로와 한 줄 요약
2. **실제로 실행한 명령과 그 출력** — 요약만 적지 말 것. 안 붙어 있으면 안 돌린 것으로 간주된다
3. **수용 기준 항목별 대조** — TASK.md 의 기준을 하나씩 O/X 로. 뭉뚱그린 "모두 충족" 금지
4. **미검증 절** — 못 돌린 것을 여기에 정직하게 적는다. **비워두려고 억지로 채우지 말 것.**
   "이 머신에 GPU 가 없어 학습 검증 불가" 는 완벽하게 받아들여지는 답이다
5. **발견했지만 건드리지 않은 것** — 범위 밖 문제를 여기에 적는다

---

## 9. 직접 지시 모드 (사용자가 Codex 를 직접 돌릴 때)

사용자가 **터미널에서 Codex 를 직접 실행 중**이라면 위 `TASK.md`/`RESULT.md` 사이클은
적용되지 않는다. 그 장치는 Claude 와 Codex 가 서로 말을 걸 수 없어서 생긴 우편함이고,
사람이 앞에 앉아 있으면 그 전제가 없다.

- **사용자의 말이 곧 지시서**다. TASK.md 를 요구하지 말 것.
- **보고는 대화로** 한다. RESULT.md 를 쓸 필요 없다.
- 커밋은 사용자가 지시할 때.
- 단, **§1 역할 경계와 §4 절대 하지 말 것은 그대로 유효하다.**
  (`docs/` 수정 금지, `--smoke` 없는 train.py 금지, 무단 커밋 금지)

절차가 사용자의 직접 작업을 막으면 그건 규약이 아니라 고장이다.

---

## 10. git 관련

- 커밋 대상 브랜치는 **`main`**.
  ⚠️ `phase1-data-acquisition` 은 **삭제된 옛 이름**이다(main 으로 통합됨). 여기에 커밋하지 말 것.
- 커밋·푸시는 **사용자가 명시적으로 지시할 때만.**
- 노트북/데스크톱 양쪽에서 작업하므로 **착수 전 `git fetch`** 를 권한다
  (원격에 Phase 가 쌓여 있으면 문서 번호·모델 이름이 이미 선점돼 있다).
