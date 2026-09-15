# docs/handoff — Claude ↔ Codex 우편함

Claude(계획·문서)와 Codex(구현·검증)가 서로 말을 걸 수 없어서 만든 **파일 우편함**이다.
두 도구가 이 폴더를 통해 지시와 보고를 주고받는다.

## 무엇이 추적되고 무엇이 안 되나

| 파일 | git | 성격 |
|---|---|---|
| `README.md` | ✅ 추적 | 이 문서 |
| `TASK.template.md` | ✅ 추적 | 지시서 원본 |
| `TASK.md` | ❌ 무시 | **휘발성** — 다음 작업에서 덮어쓰임 |
| `RESULT.md` | ❌ 무시 | **휘발성** — 다음 작업에서 덮어쓰임 |

`.gitignore` 규칙은 `docs/handoff/*` + 위 두 파일 예외다.
⚠️ `docs/handoff/`(디렉터리 자체)로 막으면 git 이 폴더 안으로 들어가지 않아 예외가 **동작하지 않는다.**
반드시 `/*` 형태를 유지할 것.

> **남길 가치가 있는 경위는 `docs/0N_*.md` 나 커밋 메시지로 옮긴다.**
> 여기 있는 채로 두면 다음 작업에서 사라진다.

## 사이클

```
1. Claude 가 TASK.template.md 를 복사해 TASK.md 작성
2. Codex 호출 (아래 두 방식 중 하나)
3. Claude 가 RESULT.md 를 "의심하며" 읽는다
4. 미흡 → 후속 TASK.md 로 재호출 / 충분 → docs/0N_*.md 로 정리
```

### 호출 방법 A — 래퍼 (권장)

```powershell
powershell -File scripts/codex/run_codex.ps1
powershell -File scripts/codex/run_codex.ps1 -ReadOnly    # 검증만 재실행 (코드 수정 불가)
```

래퍼가 해주는 것: `PYTHONIOENCODING=utf-8` 주입, 경로 기본값, RESULT.md 회수,
GPU 유무 안내, 사전 점검(TASK.md 존재·git 상태).

### 호출 방법 B — 직접 (래퍼가 안 될 때)

```bash
# 저장소 루트는 기기마다 다르다 — 경로를 박지 말고 그때그때 구한다
PYTHONIOENCODING=utf-8 codex exec -C "$(git rev-parse --show-toplevel)" -s workspace-write -o docs/handoff/LAST_MESSAGE.md - < docs/handoff/TASK.md
PYTHONIOENCODING=utf-8 codex exec -C "$(git rev-parse --show-toplevel)" -s read-only      -o docs/handoff/RESULT.md - < docs/handoff/TASK.md
```

⚠️ **래퍼가 즉시 `error: unexpected argument '...' found` 로 끝나면 방법 B 를 쓴다**(2026-09-14 실제).
래퍼는 지시서를 명령줄 인자로 넘기는데, Windows PowerShell 5.1 이 지시서 안의 큰따옴표(예: `python -c "..."`)에서 인자를 쪼갠다.
stdin 으로 넘기는 방법 B 는 이 문제가 없다.
⚠️ `requires a newer version of Codex` → CLI 업그레이드(`npm i -g @openai/codex`). `out of credits` → 충전 전까지 Codex 불가
(작업 도중 끊기면 반쯤 된 변경이 작업 트리에 남는다 — RESULT.md 가 없으므로 diff 를 직접 검토할 것).

### ⚠️⚠️ 쓰기 모드에서 `-o` 를 `RESULT.md` 로 주지 말 것 (2026-09-15 실제 사고)

위 방법 B 의 첫 줄이 `-o` 를 **`LAST_MESSAGE.md`** 로 두는 데는 이유가 있다. 그대로 쓸 것.

`-o` 는 **Codex 의 최종 채팅 메시지**를 그 파일에 쓴다. 그런데 workspace-write 모드에서는
Codex 가 **지시서를 따라 자기 손으로 `RESULT.md` 를 이미 작성**한다. 여기에 `-o RESULT.md` 를
주면 **잘 쓴 보고서가 짧은 요약 메시지로 덮어써진다.**

실제로 이렇게 됐다 — 683바이트짜리 RESULT.md 가 남았고, 그 안에는
*"검증 명령·출력과 수용 기준별 판정은 RESULT.md 에 기록했습니다"* 라는 **자기 자신을 가리키는
순환 참조**만 있었다. 검증 증거가 통째로 사라진 것이다.

- **쓰기 모드**: `-o docs/handoff/LAST_MESSAGE.md` (RESULT.md 는 Codex 가 직접 쓴다)
- **read-only 모드**: `-o docs/handoff/RESULT.md` 로 받아도 된다 — 파일을 못 쓰니 최종 메시지가 유일한 산출이다

⚠️ 증거가 사라졌다고 결과를 믿지 말 것. 그때는 **검토자가 직접 검증 명령을 돌린다**
(CPU 작업이면 가능하다). 2026-09-15 에는 그렇게 복구했고, 결과는 docs/13 §1.12 에 남겼다.

## RESULT.md 를 읽을 때 의심할 것

가장 비싼 실패는 **검증하지 않은 것을 검증했다고 보고하는 것**이다.

- 실제 명령과 **출력**이 붙어 있는가 (요약만이면 안 돌린 것으로 간주)
- "미검증" 절이 정말 비어 있는가, 아니면 **안 채운** 것인가
- skip 을 실패로, 또는 대량 실패를 "원래 깨져 있던 것"으로 오독하지 않았는가
- 수용 기준을 **항목별로** 대조했는가 (뭉뚱그린 "모두 충족" 금지)
- 통과 테스트 개수가 적혀 있는가 (168개보다 적으면 수집이 안 된 것)
- GPU 가 필요한 작업인데 노트북에서 "검증 완료"라고 쓰지 않았는가

의심스러우면 `-ReadOnly` 로 다시 불러 **검증만** 재실행시킨다 (코드 수정 불가라 안전).
