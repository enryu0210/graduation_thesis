"""draft_kiisc_submission.md → 투고양식 모양의 확인용 PDF.

최종 제출은 HWP 라서 이 PDF 는 "양식대로 놓으면 어떻게 보이나"를 확인하는 용도다.
양식 수치(190×260mm, 여백 20mm, 2단·단 간격 10mm, 본문 9pt·줄간격 150%, 캡션 8pt)를 CSS 로 옮겼다.
신명 계열 글꼴은 이 기기에 없어 바탕(명조)·맑은 고딕·Times New Roman 으로 대신한다.

실행: python docs/paper_write/build_pdf.py   (Chrome 필요, 경로가 다르면 CHROME 수정)
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SRC = HERE / "draft_kiisc_submission.md"
OUT = HERE / "draft_kiisc_submission.pdf"
CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")

CSS = """
@page { size: 190mm 260mm; margin: 20mm; @bottom-center { content: counter(page); } }
body { font-family: 'Times New Roman', Batang, serif; font-size: 9pt; line-height: 150%;
       letter-spacing: -0.03em; margin: 0; color: #000; word-break: keep-all; }
.front { text-align: center; }
.front h1 { font-family: Batang, serif; font-weight: bold; font-size: 18pt; line-height: 130%;
            letter-spacing: -0.1em; margin: 0 0 12pt; }
.front h2 { font-family: 'Times New Roman', serif; font-size: 14pt; font-weight: normal;
            line-height: 130%; margin: 0 0 8pt; }
.front h3 { font-family: 'Malgun Gothic', sans-serif; font-size: 9pt; margin: 12pt 0 8pt; }
.front p { text-align: justify; font-size: 8.5pt; text-indent: 10pt; margin: 0; }
.front p.kw { text-indent: 0; margin-top: 6pt; }
.body { column-count: 2; column-gap: 10mm; margin-top: 14pt; text-align: justify; }
.body h2 { font-family: 'Malgun Gothic', sans-serif; font-size: 10pt; margin: 12pt 0 6pt;
           break-after: avoid; }
.body h3 { font-family: 'Malgun Gothic', sans-serif; font-size: 9pt; margin: 9pt 0 4pt;
           break-after: avoid; }
.body p { margin: 0; text-indent: 10pt; }
.body ul { margin: 2pt 0; padding-left: 14pt; }
.body .eq { display: flex; justify-content: space-between; text-indent: 10pt; margin: 4pt 0; }
.body .cap { font-family: Arial, 'Malgun Gothic', sans-serif; font-size: 8pt; line-height: 130%;
             text-indent: 0; margin: 5mm 0 2pt; break-after: avoid; }
.body .note { font-size: 7.5pt; line-height: 130%; text-indent: 0; margin: 2pt 0 5mm; }
.tbl { break-inside: avoid; }
.wide { column-span: all; }
table { border-collapse: collapse; width: 100%; font-size: 8pt; line-height: 130%; }
th, td { border-top: 0.5pt solid #000; border-bottom: 0.5pt solid #000; padding: 1.5pt 3pt;
         text-align: left; vertical-align: top; }
th { border-top: 1pt solid #000; font-weight: bold; }
figure { margin: 5mm 0; break-inside: avoid; }
figure img { width: 100%; }
figcaption { font-family: Arial, sans-serif; font-size: 8pt; line-height: 130%; text-align: left; }
.ref p { text-indent: -16pt; padding-left: 16pt; font-size: 8.5pt; line-height: 135%; }
mark { background: #fff3a0; }
"""


def md(text):
    return markdown.markdown(text, extensions=["tables"])


def render_front(front):
    """제목·요약 부분은 양식이 1단·가운데 정렬이라 markdown 을 거치지 않고 직접 짠다."""
    titles = re.findall(r"^## (.+)$", front, re.M)
    if len(titles) != 2:
        sys.exit(f"제목 줄(## …)이 둘이어야 하는데 {len(titles)}개다 — 초안 머리 구조를 확인")
    parts = re.split(r"^### (.+)$", front, flags=re.M)  # [앞, 제목1, 본문1, 제목2, 본문2]
    html = [f"<h1>{titles[0]}</h1>", f"<h2>{titles[1]}</h2>"]
    for head, body in zip(parts[1::2], parts[2::2]):
        html.append(f"<h3>{head}</h3>")
        for para in [p.strip() for p in body.strip().split("\n\n") if p.strip()]:
            cls = ' class="kw"' if para.startswith("**Keywords") else ""
            html.append(f"<p{cls}>{md(para)[3:-4]}</p>")
    return '<div class="front">' + "\n".join(html) + "</div>"


def figure_from_placeholder(match):
    """【그림: … `경로.png` … "Fig. N. 캡션"】 → 실제 그림. 경로 없는 자리는 표시만 남긴다."""
    inner = match.group(1)
    path = re.search(r"<code>([^<]+\.png)</code>", inner)
    cap = re.search(r"&quot;(Fig\. \d+\..*?)&quot;|\"(Fig\. \d+\..*?)\"", inner)
    if not path or not (REPO / path.group(1)).exists():
        return f"<p><mark>【{inner}】</mark></p>"
    caption = (cap.group(1) or cap.group(2)) if cap else "Fig. ?"
    src = (REPO / path.group(1)).as_uri()
    return f'<figure class="wide"><img src="{src}"><figcaption>{caption}</figcaption></figure>'


def render_body(body):
    html = md(body)
    # 그림 자리 → 그림 (한 문단 전체가 【그림: …】 인 경우)
    html = re.sub(r"<p>【그림: (.*?)】</p>", figure_from_placeholder, html, flags=re.S)
    # 나머지 【TODO】·【확인】 은 확인용이니 눈에 띄게
    html = re.sub(r"【(.*?)】", r"<mark>【\1】</mark>", html, flags=re.S)
    # 식: "… &nbsp;&nbsp;(n)" → 식 번호 오른쪽 정렬
    html = re.sub(r"<p>(.*?)(?:\s*&nbsp;)+\s*\((\d+)\)</p>",
                  r'<p class="eq"><span>\1</span><span>(\2)</span></p>', html)
    # 표 캡션(표 위)과 표 아래 주석. 열이 5개 이상인 표는 한 단에 안 들어가 두 단에 걸친다.
    html = re.sub(r"<p>(<strong>Table \d+\.</strong>.*?)</p>\s*<table>",
                  r'<p class="cap">\1</p>\n<table>', html, flags=re.S)
    # 표 바로 뒤 문단 중 한글이 없는 것만 주석으로 본다(표 내용·주석은 영문 규칙이라 본문과 갈린다)
    html = re.sub(r"</table>\s*<p>([^<가-힣][^가-힣]*?)</p>", r'</table>\n<p class="note">\1</p>', html)

    def widen(m):
        cols = m.group(0).split("</tr>")[0].count("<th")
        cls = "tbl wide" if cols >= 5 else "tbl"  # 캡션·표·주석이 쪽/단 경계에서 갈라지지 않게 묶는다
        return f'<div class="{cls}">{m.group(0)}</div>'

    html = re.sub(r'<p class="cap">.*?</table>(?:\s*<p class="note">.*?</p>)?', widen, html, flags=re.S)
    # 초안의 "r_N", "τ_{ĉ₁(x)}", "argmax_k" 같은 아래첨자 표기 → <sub>. 경로(03_prereg_v2)는 앞이 글자라 안 걸린다
    html = re.sub(r"(?<![\w/])(argmax|max|Σ|[A-Za-zτρπŜ])_(\{[^}]*\}|[A-Za-zπ]+|\d)",
                  lambda m: f"{m.group(1)}<sub>{m.group(2).strip('{}')}</sub>", html)
    # 참고문헌은 내어쓰기
    html = re.sub(r"(<h2>References</h2>)(.*)$", r'\1<div class="ref">\2</div>', html, flags=re.S)
    return '<div class="body">' + html + "</div>"


def main():
    text = SRC.read_text(encoding="utf-8")
    # 맨 위 작업 메모(blockquote)는 논문이 아니므로 버린다: --- 로 나뉜 [메모, 제목·요약, 본문]
    chunks = re.split(r"^---\s*$", text, flags=re.M)
    if len(chunks) < 3:
        sys.exit("--- 구분선이 두 개 이상 있어야 한다(메모 / 제목·요약 / 본문)")
    front, body = chunks[1], "---".join(chunks[2:])
    page = (f"<!doctype html><html lang='ko'><head><meta charset='utf-8'><style>{CSS}</style></head>"
            f"<body>{render_front(front)}{render_body(body)}</body></html>")

    with tempfile.TemporaryDirectory() as tmp:
        html_path = Path(tmp) / "draft.html"
        html_path.write_text(page, encoding="utf-8")
        if not CHROME.exists():
            sys.exit(f"Chrome 이 없다: {CHROME}")
        subprocess.run([str(CHROME), "--headless", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={OUT}", html_path.as_uri()], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
