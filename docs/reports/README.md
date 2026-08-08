# docs/reports — 외부 제출용 보고서

교수님 제출·미팅 배포처럼 **저장소 밖으로 나가는 문서**를 둔다.
설계·의사결정 기록(`docs/0N_*.md`)과 달리 **읽는 사람이 저장소를 모른다고 가정**하고 쓴다.

## 규칙

- **HTML 이 원본, PDF·DOCX 는 산출물**이다. 내용 수정은 `.html` 을 고치고 둘 다 다시 굽는다.
  - ⚠️ 단 **DOCX 에서 직접 손본 경우는 예외**다. 그 순간 원본이 갈라지므로, 고친 내용을
    HTML 에 되돌려 넣거나 "이번 판은 DOCX 가 최신"임을 아래 목록 표에 적어 둔다.
    적어두지 않으면 다음에 HTML 에서 다시 구울 때 **손본 것이 조용히 사라진다.**
- 수치는 반드시 `docs/0N_*.md` 나 `experiments/results/*.json` 에 근거를 둔다. 새 수치를 여기서 만들지 않는다.
- 그림은 `../figures/` 를 **상대 경로**로 참조한다(절대 경로 금지 — 기기마다 드라이브 문자가 다름).
- PDF 도 커밋한다. 기기 간 이동 수단이 커밋뿐이라, 커밋하지 않으면 다른 기기에서 다시 구워야 한다.

## PDF 재생성

pandoc·wkhtmltopdf 는 이 프로젝트에 설치돼 있지 않다. **Chrome 헤드리스**로 굽는다.
Chrome 은 상대 경로 출력에서 액세스 거부가 나므로 **입출력 모두 절대 경로**로 넘긴다.

```bash
# Bash 도구 기준. 경로는 그때그때 구한다(하드코딩 금지)
WIN=$(cd "$(git rev-parse --show-toplevel)/docs/reports" && pwd -W)
"/c/Program Files/Google/Chrome/Application/chrome.exe" \
    --headless --disable-gpu --no-sandbox --no-pdf-header-footer \
    --print-to-pdf="$WIN/progress_2026-08-08.pdf" \
    "file:///$WIN/progress_2026-08-08.html"
```

⚠️ Chrome 실행 파일 위치는 기기마다 다를 수 있다. 없으면 Edge(`msedge.exe`)도 같은 옵션으로 동작한다.

## DOCX 재생성 (직접 고칠 때)

pandoc 이 없으므로 `html_to_docx.py`(python-docx + bs4, 둘 다 설치돼 있음)로 굽는다.

```bash
python docs/reports/html_to_docx.py docs/reports/progress_2026-08-08.html
# → 같은 이름의 .docx
```

⚠️ 이 변환기는 **범용이 아니다.** 보고서가 실제로 쓰는 태그·클래스만 안다
(`h2/h3 · p · ul/ol · table · figure · div.lead/.warn/.note/.flow/.qa · .pagebreak`).
HTML 에 새 구조를 추가하면 `html_to_docx.py` 의 dispatch 에도 함께 넣어야 조용히 누락되지 않는다.
검증은 표 개수·그림 개수를 HTML 과 대조하는 것이 가장 빠르다.

## 목록

| 파일 | 내용 | 기준일 | 최신 원본 |
|---|---|---|---|
| `progress_2026-08-08.{html,pdf,docx}` | 캐스케이드 전환 이후(Phase 10~12) 진행 보고. 근거: docs/09·10·11·12 | 2026-08-08 | **DOCX·PDF** |

> ⚠️ `progress_2026-08-08` 은 **HTML 과 DOCX·PDF 가 갈라져 있다**(위 예외 규칙 적용).
> DOCX·PDF 를 직접 손본 판이 최신이고, 그 수정은 HTML 에 **반영돼 있지 않다.**
> 갈라진 내용(직전 커밋본 → 현재본):
> - 문체를 서술형("~했습니다") → **개조식**("~확정.")으로 축약. 본문 글자수 −17%
> - **§8 진행 현황과 남은 계획 · §9 여쭙고 싶은 것 · 부록 B 재현성 사고** 세 절이 빠짐(16쪽 → 15쪽)
> - 표 17개 → 14개
>
> → HTML 에서 다시 구우면 위 수정이 **전부 되돌아간다.** 다시 구울 일이 있으면 HTML 을 먼저 맞출 것.
