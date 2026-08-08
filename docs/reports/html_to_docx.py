"""보고서 HTML → Word(.docx) 변환기.

왜 만들었나
-----------
`docs/reports/README.md` 규칙대로 **HTML 이 단일 원본**이고 PDF·DOCX 는 산출물이다.
그런데 이 환경에는 pandoc 이 없다(설치 확인 2026-08-08). 대신 python-docx·bs4 가
이미 있으므로, 보고서가 실제로 쓰는 태그·클래스만 다루는 작은 변환기를 직접 둔다.

범용 HTML 변환기가 아니다. `progress_*.html` 이 쓰는 아래 구조만 안다:
  h2/h3 · p · ul/ol · table(+caption, tr.hl, td.num) · figure>img+figcaption
  · div.lead/.warn/.note (강조 박스) · div.flow>div.box|div.arrow (구조 도식)
  · div.qa (질문 블록) · class="pagebreak" (쪽 나눔)
새 구조를 HTML 에 추가하면 여기 dispatch 에도 함께 넣어야 한다.

사용법
------
    python docs/reports/html_to_docx.py docs/reports/progress_2026-08-08.html
    # → 같은 폴더에 progress_2026-08-08.docx
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag
from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

# ── 색·글꼴 상수 (HTML 쪽 CSS 와 같은 값을 쓴다) ────────────────────────────
NAVY = RGBColor(0x1F, 0x3A, 0x5F)
GREEN = RGBColor(0x1A, 0x6B, 0x3C)
RED = RGBColor(0xA8, 0x32, 0x32)
GRAY = RGBColor(0x77, 0x77, 0x77)
DARK = RGBColor(0x1A, 0x1A, 0x1A)

SHADE_HEADER = "EAEFF5"   # 표 머리행
SHADE_HL = "FFF6E0"       # 강조행(tr.hl)
SHADE_LEAD = "F4F7FB"     # 파란 강조 박스
SHADE_WARN = "FDF4F4"     # 붉은 주의 박스
SHADE_FLOW = "F7F9FC"     # 구조 도식 상자

KOR_FONT = "맑은 고딕"
MONO_FONT = "Consolas"

BODY_PT = 10
TABLE_PT = 8.5


# ── 저수준 헬퍼 ────────────────────────────────────────────────────────────
def set_font(font_obj, name: str) -> None:
    """python-docx 는 동아시아 글꼴을 따로 지정해야 한글에 실제로 적용된다."""
    font_obj.name = name
    rpr = font_obj.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rfonts.set(qn(attr), name)


def shade(element, hex_color: str) -> None:
    """셀·문단 배경색. python-docx 에 API 가 없어 XML 을 직접 넣는다."""
    pr = element._tc.get_or_add_tcPr() if hasattr(element, "_tc") else element._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), hex_color)
    pr.append(shd)


def left_bar(paragraph, hex_color: str) -> None:
    """강조 박스의 왼쪽 세로선. CSS border-left 를 문단 테두리로 흉내낸다."""
    ppr = paragraph._p.get_or_add_pPr()
    borders = OxmlElement("w:pBdr")
    bar = OxmlElement("w:left")
    bar.set(qn("w:val"), "single")
    bar.set(qn("w:sz"), "18")       # 1/8 pt 단위
    bar.set(qn("w:space"), "6")
    bar.set(qn("w:color"), hex_color)
    borders.append(bar)
    ppr.append(borders)


def classes(node: Tag) -> list[str]:
    return node.get("class") or []


# ── 인라인 서식 ────────────────────────────────────────────────────────────
def add_inline(paragraph, node, *, bold=False, italic=False, mono=False, color=None, size=None):
    """자식 노드를 재귀로 훑어 b/i/code/span 색상을 run 서식으로 옮긴다."""
    for child in node.children:
        if isinstance(child, NavigableString):
            text = re.sub(r"\s+", " ", str(child))
            if not text.strip() and not paragraph.runs:
                continue  # 태그 사이 들여쓰기 공백은 버린다
            run = paragraph.add_run(text)
            run.bold, run.italic = bold, italic
            set_font(run.font, MONO_FONT if mono else KOR_FONT)
            run.font.size = Pt(size if size else BODY_PT)
            run.font.color.rgb = color if color else DARK
            continue

        if child.name == "br":
            paragraph.add_run().add_break()
            continue

        cls = classes(child)
        nxt_color = color
        nxt_bold = bold or child.name in ("b", "strong")
        if "pos" in cls:
            nxt_color, nxt_bold = GREEN, True
        elif "neg" in cls:
            nxt_color, nxt_bold = RED, True
        elif "muted" in cls:
            nxt_color = GRAY

        add_inline(
            paragraph, child,
            bold=nxt_bold,
            italic=italic or child.name in ("i", "em"),
            mono=mono or child.name == "code",
            color=nxt_color,
            size=size,
        )


def para(doc, node, *, size=None, space_after=6, indent=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.space_before = Pt(0)
    if indent:
        p.paragraph_format.left_indent = Cm(indent)
    add_inline(p, node, size=size)
    return p


# ── 블록별 변환 ────────────────────────────────────────────────────────────
def add_callout(doc, node, shade_hex: str, bar_hex: str):
    """div.lead / div.warn — 배경색 + 왼쪽 세로선을 가진 강조 박스."""
    # <ol>/<ul> 을 품은 경우가 있어 문단 단위로 쪼개 각각 배경을 입힌다.
    blocks = [c for c in node.children if isinstance(c, Tag)] or [node]
    inline_only = all(c.name not in ("ol", "ul", "table") for c in blocks)

    if inline_only:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(8)
        p.paragraph_format.left_indent = Cm(0.2)
        add_inline(p, node, size=BODY_PT - 0.3)
        shade(p, shade_hex)
        left_bar(p, bar_hex)
        return

    for child in blocks:
        if child.name in ("ol", "ul"):
            for idx, li in enumerate(child.find_all("li", recursive=False), 1):
                p = doc.add_paragraph()
                p.paragraph_format.left_indent = Cm(0.8)
                p.paragraph_format.space_after = Pt(3)
                marker = f"{idx}. " if child.name == "ol" else "· "
                run = p.add_run(marker)
                set_font(run.font, KOR_FONT)
                run.font.size = Pt(BODY_PT - 0.3)
                add_inline(p, li, size=BODY_PT - 0.3)
                shade(p, shade_hex)
                left_bar(p, bar_hex)
        else:
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.2)
            p.paragraph_format.space_after = Pt(3)
            add_inline(p, child, size=BODY_PT - 0.3)
            shade(p, shade_hex)
            left_bar(p, bar_hex)


def add_list(doc, node):
    ordered = node.name == "ol"
    for idx, li in enumerate(node.find_all("li", recursive=False), 1):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.7)
        p.paragraph_format.space_after = Pt(3)
        run = p.add_run(f"{idx}. " if ordered else "• ")
        set_font(run.font, KOR_FONT)
        run.font.size = Pt(BODY_PT)
        add_inline(p, li)


def add_table(doc, node):
    head_rows = node.select("thead tr")
    body_rows = node.select("tbody tr")
    rows = head_rows + body_rows
    if not rows:
        return
    ncols = max(len(r.find_all(["th", "td"])) for r in rows)

    table = doc.add_table(rows=len(rows), cols=ncols)
    table.style = "Table Grid"
    table.autofit = True

    for r_idx, tr in enumerate(rows):
        is_header = tr in head_rows
        is_hl = "hl" in classes(tr)
        for c_idx, cell_node in enumerate(tr.find_all(["th", "td"])):
            if c_idx >= ncols:
                break
            cell = table.cell(r_idx, c_idx)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.TOP
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.space_before = Pt(1)
            if "num" in classes(cell_node):
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            add_inline(p, cell_node, bold=is_header, size=TABLE_PT)
            if is_header:
                shade(cell, SHADE_HEADER)
            elif is_hl:
                shade(cell, SHADE_HL)

    caption = node.find("caption")
    if caption:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(10)
        add_inline(p, caption, italic=True, color=GRAY, size=TABLE_PT - 0.3)
    else:
        doc.add_paragraph().paragraph_format.space_after = Pt(4)


def add_flow(doc, node):
    """div.flow — 캐스케이드 구조 도식. 1행짜리 표로 옮긴다(테두리 없음)."""
    cells_src = [c for c in node.children if isinstance(c, Tag)]
    table = doc.add_table(rows=1, cols=len(cells_src))
    table.style = "Table Grid"
    for i, src in enumerate(cells_src):
        cell = table.cell(0, i)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        p = cell.paragraphs[0]
        is_arrow = "arrow" in classes(src)
        if is_arrow:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run("→")
            set_font(run.font, KOR_FONT)
            run.font.size = Pt(14)
            run.font.color.rgb = NAVY
            run.bold = True
        else:
            add_inline(p, src, size=TABLE_PT)
            shade(cell, SHADE_FLOW)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)


def add_figure(doc, node, base_dir: Path):
    img = node.find("img")
    if img and img.get("src"):
        path = (base_dir / img["src"]).resolve()
        if path.exists():
            doc.add_picture(str(path), width=Cm(17.0))
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        else:
            para(doc, BeautifulSoup(f"<p>[그림 없음: {img['src']}]</p>", "html.parser").p)
    cap = node.find("figcaption")
    if cap:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(10)
        add_inline(p, cap, color=GRAY, size=TABLE_PT)


def add_qa(doc, node):
    """div.qa — 'Q1  제목' 한 줄 + 본문."""
    qid = node.find("span", class_="qid")
    qtitle = node.find("span", class_="qtitle")
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(3)
    if qid:
        run = p.add_run(f"[{qid.get_text(strip=True)}] ")
        set_font(run.font, KOR_FONT)
        run.font.size = Pt(BODY_PT)
        run.bold = True
        run.font.color.rgb = NAVY
    if qtitle:
        add_inline(p, qtitle, bold=True)
    for body in node.find_all("p", recursive=False):
        para(doc, body, indent=0.3, space_after=4)


# ── 문서 조립 ──────────────────────────────────────────────────────────────
def build(html_path: Path, docx_path: Path) -> None:
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "html.parser")
    base_dir = html_path.parent
    doc = Document()

    section = doc.sections[0]
    section.page_width, section.page_height = Cm(21.0), Cm(29.7)   # A4
    section.top_margin = section.bottom_margin = Cm(2.0)
    section.left_margin = section.right_margin = Cm(1.9)

    normal = doc.styles["Normal"]
    set_font(normal.font, KOR_FONT)
    normal.font.size = Pt(BODY_PT)
    normal.paragraph_format.line_spacing = 1.25
    for style_name, size in (("Heading 1", 13.5), ("Heading 2", 11.5)):
        st = doc.styles[style_name]
        set_font(st.font, KOR_FONT)
        st.font.size = Pt(size)
        st.font.bold = True
        st.font.color.rgb = NAVY

    # 표지
    cover = soup.find("div", class_="cover")
    if cover:
        for cls, size, color, bold in (
            ("kicker", 9, GRAY, False),
            (None, 16, NAVY, True),      # h1
            ("sub", 10, DARK, False),
            ("meta", 8.5, GRAY, False),
        ):
            node = cover.find("h1") if cls is None else cover.find("div", class_=cls)
            if not node:
                continue
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(4)
            add_inline(p, node, bold=bold, color=color, size=size)
        doc.add_paragraph().paragraph_format.space_after = Pt(6)

    body = soup.find("body")
    for node in body.children:
        if not isinstance(node, Tag) or node is cover:
            continue
        cls = classes(node)

        if "pagebreak" in cls:
            doc.add_page_break()

        if node.name == "h2":
            h = doc.add_paragraph(style="Heading 1")
            h.paragraph_format.space_before = Pt(14)
            h.paragraph_format.space_after = Pt(6)
            add_inline(h, node, bold=True, color=NAVY, size=13.5)
        elif node.name == "h3":
            h = doc.add_paragraph(style="Heading 2")
            h.paragraph_format.space_before = Pt(10)
            h.paragraph_format.space_after = Pt(4)
            add_inline(h, node, bold=True, color=NAVY, size=11.5)
        elif node.name == "table":
            add_table(doc, node)
        elif node.name in ("ul", "ol"):
            add_list(doc, node)
        elif node.name == "figure":
            add_figure(doc, node, base_dir)
        elif node.name == "p":
            is_note = "note" in cls
            para(doc, node,
                 size=(BODY_PT - 1.2) if is_note else None,
                 space_after=8 if is_note else 6)
        elif node.name == "div":
            if "lead" in cls:
                add_callout(doc, node, SHADE_LEAD, "1F3A5F")
            elif "warn" in cls:
                add_callout(doc, node, SHADE_WARN, "A83232")
            elif "flow" in cls:
                add_flow(doc, node)
            elif "qa" in cls:
                add_qa(doc, node)

    doc.save(str(docx_path))


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    html_path = Path(sys.argv[1]).resolve()
    if not html_path.exists():
        print(f"입력 HTML 을 찾을 수 없습니다: {html_path}")
        return 1
    docx_path = html_path.with_suffix(".docx")
    build(html_path, docx_path)
    print(f"생성 완료: {docx_path} ({docx_path.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
