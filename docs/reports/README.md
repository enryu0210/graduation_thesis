# docs/reports — 외부 제출용 보고서

> ℹ️ 이 문서가 가리키는 `docs/01`~`11`·`docs/archive`·옛 그림·옛 보고서는 2026-11-04 정리로 작업 트리에서 지웠다 → 복구 방법은 `docs/DELETED_DOCS.md`.

교수님 제출·미팅 배포처럼 **저장소 밖으로 나가는 문서**를 둔다.
설계·의사결정 기록(`docs/0N_*.md`)과 달리 **읽는 사람이 저장소를 모른다고 가정**하고 쓴다.

## 폴더 구조 (2026-10-04 정리)

| 폴더 | 무엇을 두나 | 파일 이름 규칙 |
|---|---|---|
| `progress/` | 교수님 제출·미팅용 **진행 보고**(시점별 스냅숏) | `progress/progress_YYYY-MM-DD.{html,pdf,docx}` |
| `papers/` | 보고서와 함께 교수님께 드린 **인용 논문 사본** | 원 파일명 그대로 |
| (이 폴더 바로 아래) | 이 README 와 변환 도구 `html_to_docx.py` 만 | — |

- 새 보고서는 위 표의 해당 폴더에 바로 만든다. 폴더가 한 단계 깊어졌으므로 HTML 의 그림 경로는 **`../../figures/`** 다.
- 다른 문서에서 보고서를 가리킬 때는 폴더까지 적는다(예: `docs/reports/progress/progress_2026-10-31.pdf`).

## 규칙

- **HTML 이 원본, PDF·DOCX 는 산출물**이다. 내용 수정은 `.html` 을 고치고 둘 다 다시 굽는다.
  - ⚠️ **예외 하나**: `current_baseline_*.pdf` 는 **`docs/CURRENT_BASELINE.md` 가 원본**이다.
    저장소 안에서도 읽히는 참조 문서라 Markdown 을 원본으로 두는 쪽이 유지비가 싸다.
    굽는 방법은 아래 "Markdown 원본 문서의 PDF 재생성" 절.
  - ⚠️ 단 **DOCX 에서 직접 손본 경우는 예외**다. 그 순간 원본이 갈라지므로, 고친 내용을
    HTML 에 되돌려 넣거나 "이번 판은 DOCX 가 최신"임을 아래 목록 표에 적어 둔다.
    적어두지 않으면 다음에 HTML 에서 다시 구울 때 **손본 것이 조용히 사라진다.**
- 수치는 반드시 `docs/0N_*.md` 나 `experiments/results/*.json` 에 근거를 둔다. 새 수치를 여기서 만들지 않는다.
- 그림은 `../../figures/` 를 **상대 경로**로 참조한다(절대 경로 금지 — 기기마다 드라이브 문자가 다름).
- PDF 도 커밋한다. 기기 간 이동 수단이 커밋뿐이라, 커밋하지 않으면 다른 기기에서 다시 구워야 한다.

## PDF 재생성

pandoc·wkhtmltopdf 는 이 프로젝트에 설치돼 있지 않다. **Chrome 헤드리스**로 굽는다.
Chrome 은 상대 경로 출력에서 액세스 거부가 나므로 **입출력 모두 절대 경로**로 넘긴다.

```bash
# Bash 도구 기준. 경로는 그때그때 구한다(하드코딩 금지)
WIN=$(cd "$(git rev-parse --show-toplevel)/docs/reports/progress" && pwd -W)
"/c/Program Files/Google/Chrome/Application/chrome.exe" \
    --headless --disable-gpu --no-sandbox --no-pdf-header-footer \
    --print-to-pdf="$WIN/progress_2026-10-31.pdf" \
    "file:///$WIN/progress_2026-10-31.html"
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
mkdir -p "$ROOT/docs/reports/summary"  # 2026-11-04 폴더 삭제, 필요할 때만 다시 만든다
WIN=$(cd "$ROOT/docs/reports/summary" && pwd -W); TMPWIN=$(cd "$TMP" && pwd -W)
"/c/Program Files/Google/Chrome/Application/chrome.exe" \
    --headless --disable-gpu --no-sandbox --no-pdf-header-footer \
    --print-to-pdf="$WIN/current_baseline_2026-08-24.pdf" "file:///$TMPWIN/out.html"
```

검증은 다른 문서와 같다 — **표 개수를 원본과 대조**한다(2026-08-24 기준 표 **25개** · h2 11개 · **15쪽**).

> ⚠️ **`CURRENT_BASELINE.md` 는 내부 참조본이다.** 2026-08-24 에 외부 독자용 용어 해설을 넣었다가
> 도로 걷어냈다 — 이 문서는 `docs/NN`·`Phase N`·`F5` 같은 상호참조가 본체라서, 그걸 풀어 쓰면
> 내부 문서로서 못 쓰게 된다. **교수님 제출본은 `progress_2026-08-24.html` 로 분리**했다.

## DOCX 재생성 (직접 고칠 때)

pandoc 이 없으므로 `html_to_docx.py`(python-docx + bs4, 둘 다 설치돼 있음)로 굽는다.

```bash
python docs/reports/html_to_docx.py docs/reports/progress/progress_2026-10-31.html
# → 같은 이름의 .docx
```

⚠️ 이 변환기는 **범용이 아니다.** 보고서가 실제로 쓰는 태그·클래스만 안다
(`h2/h3 · p · ul/ol · table · figure · div.lead/.warn/.note/.flow/.qa · .pagebreak`).
HTML 에 새 구조를 추가하면 `html_to_docx.py` 의 dispatch 에도 함께 넣어야 조용히 누락되지 않는다.
검증은 표 개수·그림 개수를 HTML 과 대조하는 것이 가장 빠르다.

## 함께 두는 인용 논문 PDF (2026-08-12 추가)

보고서가 인용하는 논문 중 **초안이 인용하는 2편**은 `papers/` 에 사본을 두고 **커밋한다.**

`2312.13041v1.pdf` · `deepsloth_2010.02432.pdf`

- 원본은 `docs/thetics/` 에 있지만 그쪽은 **`.gitignore` 대상**이라 기기를 옮기면 따라오지 않는다.
  제출 묶음은 다시 만들 일이 잦으므로 사본을 추적한다.
- **서지정보는 여기 적지 않는다** — 논문 목록의 단일 진실 소스는 `docs/12 §1.3` 이다.
- ⚠️ 압축본(`.zip`)은 커밋하지 않는다(`.gitignore` 의 `docs/reports/**/*.zip` — 하위 폴더까지 막는다). 위 PDF 들의 중복이라 히스토리만 불린다.
  제출용 묶음이 필요하면 그때 다시 압축한다.

## 목록

| 파일 | 내용 | 기준일 | 최신 원본 |
|---|---|---|---|
| `progress/progress_2026-10-25.{html,pdf}` ⭐ | **교수님 제출용 진행 보고(10-14 판 후속).** 딥러닝 후속 후보 B(작은 문자 모델 → 문자 단위 CNN)·C(문자 단위 CNN 내부 조기 종료)의 학습 없는 관문(합성곱이 시간의 91%), 학습 전에 고정한 판정 기준, 본 측정 결과(**후보 C 는 단독 모델보다 위 3.53배·F1 −0.10%p, 후보 B 는 위가 아님**), 후보 C 의 이득이 넘김이 아니라 보조 출력 자체에서 나왔다는 fold 별 해석, 같은 날 후속 측정(폭 3 단독 학습은 F1 −1.36%p 로 위가 아님·단 학습 불안정·p 0.34, 씨앗 43 에서 후보 C 3.62배·+0.33%p 로 2/2, 유사 연구 그래프에 후보 C 점 추가 — 개선 주장 아님), 다음 계획(폭 3 학습 안정화 대조·후보 A 관문). 질문 문장은 2026-10-31 제거(질문은 사용자가 직접). 10-14 판과 같은 서술형·내부 약칭 금지 규칙. 근거: docs/14 §8.31~§8.33 · `docs/paper_revision/03_prereg_v2.md` §E7·§E8·§E9 | 2026-10-25 | **HTML** (표 5 · h2 8 · 그림 1 · 5쪽) |
| `progress/progress_2026-10-27.{html,pdf}` ⭐ | **교수님 제출용 진행 보고(10-25 판 후속).** 후보 A(문자 단위 CNN → DistilBERT)와 같은 데이터·같은 fold·같은 CPU 로 재구현한 유사 연구 DistilBERT 비교: 학습 전 판정 기준, **씨앗 42 는 위(12.3배·F1 −0.03%p)**, 속도 배수 평균 방식 정정(fold 산술평균 51.6배 → 평균 지연 기준 12.3배, 예시 포함), **씨앗 43 확인은 재현 실패(10.6배지만 F1 −0.32%p > 0.3%p, p 0.038)** → "위"는 씨앗 42 한정, 해석(속도는 문자 단위 CNN 몫·캐스케이드의 정확도 이득 비유의·클래스별 임계값은 병기만), 유사 연구 그래프, 다음 계획. 질문 절(3건)은 2026-10-31 제거(질문은 사용자가 직접) → h2 9. 10-25 판과 같은 서술형·내부 약칭 금지 규칙. 근거: docs/14 §8.34·§8.35 · `docs/paper_revision/03_prereg_v2.md` §E10·§E11 | 2026-10-27 | **HTML** (표 3 · h2 9 · 그림 1 · 4쪽) |
| `progress/progress_2026-10-31.{html,pdf}` ⭐ | **교수님 제출용 진행 보고(10-27 판 후속).** 범위를 딥러닝 구성으로 정리, DistilBERT 이득은 공격 종류 구분(사후), **128 토큰 원 설정 재구현은 512 보다 −1.96·−1.70%p**(문자 단위 CNN 단독보다도 낮음), 작은 출구 앞단 캐스케이드는 후보 A 보다 위(2/2, ×2.40·×2.27 — DistilBERT 보다 위는 아님), 느린 fold 진단(사후), **클래스별 임계값은 새 씨앗 44·45 에서 재현 안 됨**, URL 인코딩 변형 공격의 비용 6.4배·앞단 공격 판정률 99.4→86.7%, 하지 않기로 한 것 3건, 논문 초안 요약. 질문 절 없음(사용자가 직접). 근거: docs/14 §8.36~§8.41 · `docs/paper_revision/03_prereg_v2.md` §E12~§E15 | 2026-10-31 | **HTML** (표 6 · h2 12 · 그림 0 · 5쪽) |
| `progress/progress_2026-11-08.{html,pdf}` ⭐ | **교수님 피드백 반영 논문 초안 개편 및 수치 정합화 보고(10-31 판 후속).** char-CNN 단독 대비 F1 개선(+0.28·+0.53%p, CmdI F1 +0.94·+1.74%p) + DistilBERT 단독 대비 12.3배(10.6배) 가속 교환 부각(Table 2 개편), SQLi–CmdI 오분류 개선(56~74% 회수)과 CmdI 정밀도·F1 향상을 통한 보안 관제 유효 경고 확보(혼동행렬 방향성 점검 및 Table 3 캐스케이드 열 추가), 초록 [문제 → 해결 → 한계] 3단 축약, RQ1(RGB-CNN) 및 내부 코드명 전면 서술어화, Fig. 1 범례 축소 및 DistilBERT 대비 가속 프런티어 재제작, 원고 내 수치 불일치 8건 전수 교차검증·정정. 근거: `docs/paper_write/draft_kiisc_submission.md` · `src/eval/literature_compare.py` (커밋 `4ce36df`) | 2026-11-08 | **HTML** (표 4 · h2 8 · 그림 1) |
