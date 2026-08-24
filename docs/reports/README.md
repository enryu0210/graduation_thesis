# docs/reports — 외부 제출용 보고서

교수님 제출·미팅 배포처럼 **저장소 밖으로 나가는 문서**를 둔다.
설계·의사결정 기록(`docs/0N_*.md`)과 달리 **읽는 사람이 저장소를 모른다고 가정**하고 쓴다.

## 규칙

- **HTML 이 원본, PDF·DOCX 는 산출물**이다. 내용 수정은 `.html` 을 고치고 둘 다 다시 굽는다.
  - ⚠️ **예외 하나**: `current_baseline_*.pdf` 는 **`docs/CURRENT_BASELINE.md` 가 원본**이다.
    저장소 안에서도 읽히는 참조 문서라 Markdown 을 원본으로 두는 쪽이 유지비가 싸다.
    굽는 방법은 아래 "Markdown 원본 문서의 PDF 재생성" 절.
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

## Markdown 원본 문서의 PDF 재생성 (`docs/CURRENT_BASELINE.md`)

`pandoc` 이 없으므로 **python-markdown → HTML → 위 Chrome 헤드리스** 2단계로 굽는다.
변환기는 저장소에 두지 않는다(일회성 빌드 도구) — 아래 스니펫을 임시 디렉터리에 붙여 쓴다.

```bash
ROOT=$(git rev-parse --show-toplevel); TMP=$(mktemp -d)
cat > "$TMP/style.css" <<'CSS'
@page { size: A4; margin: 16mm 14mm 18mm 14mm; }
body { font-family: "Malgun Gothic","Segoe UI",sans-serif; font-size: 9.6pt; line-height: 1.55; }
h1 { font-size: 17pt; color:#1f4e79; border-bottom: 2.5px solid #1f4e79; }
h2 { font-size: 13pt; color:#1f4e79; border-bottom: 1px solid #c8d4e0; page-break-after: avoid; }
h3 { font-size: 11pt; color:#24405c; page-break-after: avoid; }
table { border-collapse: collapse; width: 100%; font-size: 8.4pt; }
th, td { border: 1px solid #c2ccd6; padding: 3.5pt 5pt; vertical-align: top; }
th { background: #e6edf4; }
tr { page-break-inside: avoid; }
code { font-family: Consolas, monospace; font-size: 8.6pt; background:#eef2f6; }
blockquote { background:#f5f8fb; border-left: 3px solid #7f9ab5; padding: 6pt 10pt; }
pre { background:#f4f6f8; border-left: 3px solid #7f9ab5; padding: 7pt 9pt; page-break-inside: avoid; }
CSS
python - "$ROOT/docs/CURRENT_BASELINE.md" "$TMP" <<'PY'
import sys, pathlib, markdown
src, tmp = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
body = markdown.markdown(src.read_text(encoding="utf-8"),
                         extensions=["tables", "fenced_code", "sane_lists", "attr_list"])
css = (tmp / "style.css").read_text(encoding="utf-8")
(tmp / "out.html").write_text(
    f'<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">'
    f'<title>현행 기준 정리</title><style>{css}</style></head><body>{body}</body></html>',
    encoding="utf-8")
print("표", body.count("<table>"), "개 / h2", body.count("<h2>"), "개")
PY
# 2단계: Chrome 헤드리스 (입출력 모두 절대 경로)
WIN=$(cd "$ROOT/docs/reports" && pwd -W); TMPWIN=$(cd "$TMP" && pwd -W)
"/c/Program Files/Google/Chrome/Application/chrome.exe" \
    --headless --disable-gpu --no-sandbox --no-pdf-header-footer \
    --print-to-pdf="$WIN/current_baseline_2026-08-24.pdf" "file:///$TMPWIN/out.html"
```

검증은 다른 문서와 같다 — **표 개수를 원본과 대조**한다(2026-08-24 기준 표 22개 · h2 11개 · 13쪽).

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

## 함께 두는 인용 논문 PDF (2026-08-12 추가)

보고서가 인용하는 논문 중 **교수님께 함께 드린 5편**은 이 폴더에 사본을 두고 **커밋한다.**

`2312.13041v1.pdf` · `2512.19203v2.pdf` · `deepsloth_2010.02432.pdf` ·
`feature_squeezing_ndss2018.pdf` · `s42400-023-00170-z.pdf`

- 원본은 `docs/thetics/` 에 있지만 그쪽은 **`.gitignore` 대상**이라 기기를 옮기면 따라오지 않는다.
  제출 묶음은 다시 만들 일이 잦으므로 사본을 추적한다.
- **서지정보는 여기 적지 않는다** — 논문 목록의 단일 진실 소스는 `docs/12 §1.3` 이다.
- ⚠️ 압축본(`.zip`)은 커밋하지 않는다(`.gitignore`). 위 PDF 들의 중복이라 히스토리만 불린다.
  제출용 묶음이 필요하면 그때 다시 압축한다.

## 목록

| 파일 | 내용 | 기준일 | 최신 원본 |
|---|---|---|---|
| `progress_2026-08-08.{html,pdf,docx}` | 캐스케이드 전환 이후(Phase 10~12) 진행 보고. 근거: docs/09·10·11·12 | 2026-08-08 | **DOCX·PDF** |
| `current_baseline_2026-08-24.pdf` | **현행 기준 정리 — RQ · 평가지표 · 데이터셋.** 1기(gray CNN)·2기(MCC)·3기(Phase 13 개편)를 시기별로 갈라, 지금 무엇이 유효하고 무엇이 폐기됐는지를 한 문서로 통합. 근거: 마스터 + docs/07·11·12·13 | 2026-08-24 | **`docs/CURRENT_BASELINE.md`** (Markdown 원본 — 위 예외) |
| `thesis_onepass_2026-08-12.{html,pdf}` | **논문 반영용 ONE FLOW 정리.** 캐스케이드 전환 이후 전 과정을 **논문 서술 순서**(RQ1→RQ5→RQ2→RQ3→RQ6, RQ4 별도)로 한 흐름에 엮고, 목차 매핑·그림 인벤토리·기여 문구/금지어까지 붙였다. 근거: docs/09·10·11·12 + 마스터 | 2026-08-12 | **HTML** |

> ⚠️ `progress_2026-08-08` 은 **Phase 12 종결 전**(M1·M2·M4 실측 이전) 판이다.
> Phase 12 결과(docs/11 §13·§14·§15 와 종결 요약 §16)를 담은 것은
> **`thesis_onepass_2026-08-12`** 이다. 두 보고서는 대상이 다르다 —
> 앞의 것은 **미팅용 진행 보고**, 뒤의 것은 **논문 작성용 재료 정리**다.
>
> `thesis_onepass_2026-08-12` 는 **HTML 이 원본**이다(위 기본 규칙 그대로).
> DOCX 는 굽지 않았다 — 필요하면 `html_to_docx.py` 로 굽되, 이 문서가 쓰는
> **`pre`(4곳)와 `h4`(1곳)는 변환기 dispatch 에 없다**(2026-08-12 코드 확인:
> 처리 태그는 h2·h3·table·ul/ol·figure·p·div뿐). 먼저 추가하지 않으면
> 그 블록들이 **조용히 누락**된다.

> ⚠️ `progress_2026-08-08` 은 **HTML 과 DOCX·PDF 가 갈라져 있다**(위 예외 규칙 적용).
> DOCX·PDF 를 직접 손본 판이 최신이고, 그 수정은 HTML 에 **반영돼 있지 않다.**
> 갈라진 내용(직전 커밋본 → 현재본):
> - 문체를 서술형("~했습니다") → **개조식**("~확정.")으로 축약. 본문 글자수 −17%
> - **§8 진행 현황과 남은 계획 · §9 여쭙고 싶은 것 · 부록 B 재현성 사고** 세 절이 빠짐(16쪽 → 15쪽)
> - 표 17개 → 14개
>
> → HTML 에서 다시 구우면 위 수정이 **전부 되돌아간다.** 다시 구울 일이 있으면 HTML 을 먼저 맞출 것.
