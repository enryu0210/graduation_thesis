# TASK — <한 줄 제목>

> 작성: Claude / 수행: Codex / 날짜: YYYY-MM-DD
> ⚠️ 이 파일은 **휘발성**이다. 다음 작업에서 덮어쓰인다.
> 남길 가치가 있는 경위는 `docs/0N_*.md` 나 커밋 메시지로 옮긴다.

## 1. 배경 — 왜 이 작업이 필요한가
<지금 무엇이 잘못됐거나 부족한지. Codex 가 맥락 없이 판단하지 않도록 2~4줄.>

## 2. 할 것
<구체적으로. "개선" 같은 말 대신 어떤 파일에서 무엇이 어떻게 되어야 하는지.>

- [ ] `src/...` — <무엇을>
- [ ] `tests/...` — <어떤 테스트를 추가/수정>

## 3. 하지 말 것 (경계)
> ⚠️ **이 절을 비우지 말 것.** 없으면 Codex 가 요청과 무관한 인접 코드를 정리하려 든다.

- `docs/` 는 건드리지 말 것 (이 RESULT.md 만 예외)
- `git commit` / `git push` 하지 말 것
- <이번 작업에서 특별히 손대면 안 되는 파일·함수>
- 요청하지 않은 리팩터링·포맷팅·주석 정리 금지
  (거슬리는 게 보이면 고치지 말고 RESULT 의 "건드리지 않은 것" 절에 적을 것)

## 4. 검증 명령 (전문 그대로 실행)

```bash
# 필수 — 2026-08-25 기준 168 passed, 약 5초
PYTHONIOENCODING=utf-8 python -m pytest tests/ -q
```

```bash
# <이번 작업에 필요한 추가 검증. 없으면 이 블록을 지울 것>
```

> ⚠️ `PYTHONIOENCODING=utf-8` 를 빼면 한글 콘솔에서 UnicodeEncodeError 로 죽는다.
> ⚠️ `train.py` 로 코드 동작만 확인할 때는 **반드시 `--smoke`** (추적 대상 그림 덮어쓰기 방지).
> ⚠️ 학습이 필요한데 `torch.cuda.is_available()` 이 False 면 이 머신은 노트북이다.
>    시도하지 말고 "이 머신에서 검증 불가"로 보고할 것.

## 5. 수용 기준 (O/X 로 판정 가능하게)
> ⚠️ "잘 동작한다" 같은 문장 금지. 각 항목이 참/거짓으로 갈려야 한다.

- [ ] `pytest tests/ -q` 가 **168개 이상** 통과하고 실패 0
- [ ] <예: `build_tag(...)` 가 `--lr 3e-4` 일 때 `_lr0.0003` 을 포함한다>
- [ ] <예: 기존 산출물 파일명이 바뀌지 않는다 (하위 호환)>

## 6. 동기화 지점 (해당하면 체크, 아니면 이 절 삭제)
> AGENTS.md §5 참조 — 한 곳만 고치면 조용히 깨진다.

- [ ] 새 CSV 트랙 → 5곳 (preprocess / build_image_dataset / baseline_tfidf / diagnose_payload_bias / train)
- [ ] 새 이미지 모델 → `IMAGE_MODELS` 2곳 (train / cross_validate)
- [ ] 새 채널 인코더 → `_ENCODER_ABBR` 2곳 (build_image_dataset / data_image)
- [ ] tag 규칙 변경 → 2곳 (train / cross_validate)

## 7. 참고 파일
- `AGENTS.md` — 역할 경계, 함정, 검증 명령 (Codex 가 자동으로 읽음)
- <이 작업과 직접 관련된 기존 코드·문서 경로>
