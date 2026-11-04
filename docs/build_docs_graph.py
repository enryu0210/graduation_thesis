"""docs/ 와 CLAUDE.md 의 연결 관계를 graphify 그래프로 만든다.

왜 직접 추출하는가: `graphify extract` 는 문서 본문을 외부 LLM 으로 보낸다.
미발표 논문을 내보내지 않으려고 제목·명시적 참조·실험 코드만 정규식으로 뽑는다.
그래서 "뜻이 비슷한 문서" 연결은 없고, 문서에 실제로 적힌 연결만 있다.

사용법 (저장소 루트에서):
    python docs/build_docs_graph.py
    graphify cluster-only . --no-label
    graphify query "E10"                    # 코드와 그 주변 절
    graphify explain "DL-A"                 # 하이픈이 든 이름은 query 가 쪼개므로 explain 으로
    graphify path "RQ5" "E15" --undirected  # 간선이 절→개념 방향이라 --undirected 가 필요하다
"""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "graphify-out" / "graph.json"
MASTER = "docs/web_attack_image_cnn_thesis_design.md"
# docs/01~11 은 2026-11-04 삭제됐다. 참조를 버리지 않고 한 노드로 모아 "어디서 옛 문서를 가리키는지" 찾게 한다.
DELETED_NUMBERS = {f"{n:02d}" for n in range(1, 12)}

# 대문자 코드는 F1 점수·파일명과 섞이지 않게 앞뒤 경계를 강제한다. F1~F4 는 지표와 입력 조합이 겹쳐 뺐다.
CONCEPT = re.compile(r"(?<![\w.\-/])(RQ[1-7]|E\d{1,2}|DL-[ABC]|G[1-6]|T\d{1,2}|C[1-9]|B0|H-[A-Za-z0-9-]+)(?![\w])")
KEYWORDS = ("DistilBERT", "char-CNN", "RGB CNN", "SR-BH", "slowdown", "HARKing", "실사용자")
DOC_REF = re.compile(r"docs/(\d{2})(?:_[\w.\-]*)?((?:\s*§\s*\d+(?:\.\d+)*(?:\s*[~·,]\s*§?\s*\d+(?:\.\d+)*)*)?)")
MASTER_REF = re.compile(r"마스터((?:\s*§\s*\d+(?:\.\d+)*)?)")
LOCAL_REF = re.compile(r"§\s*(\d+(?:\.\d+)*)")
FILE_REF = re.compile(r"[\w#().\-]+\.(?:md|py|png|pdf|csv|hwp)\b")
HEADING = re.compile(r"^(#{1,4})\s+(.+?)\s*$")
SECTION_NUMBER = re.compile(r"^[^\d]{0,6}(\d+(?:\.\d+)*)")


def html_to_markdown(text):
    """보고서 HTML 을 제목·본문 줄로 바꿔 Markdown 과 같은 경로로 읽는다."""
    text = re.sub(r"<(style|script)\b.*?</\1>", "", text, flags=re.S)
    text = re.sub(r"<h([1-3])[^>]*>", lambda m: "\n" + "#" * int(m.group(1)) + " ", text)
    text = re.sub(r"</h[1-3]>", "\n", text)
    return re.sub(r"<[^>]+>", "", text)


def split_sections(text):
    """(제목, 줄 번호, 본문) 목록을 돌려준다. 코드 블록 안의 # 은 제목이 아니다."""
    sections = [("", 1, [])]
    in_fence = False
    for number, line in enumerate(text.splitlines(), start=1):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        match = None if in_fence else HEADING.match(line)
        if match:
            sections.append((match.group(2).strip("# *"), number, []))
        else:
            sections[-1][2].append(line)
    return [(title, line, "\n".join(body)) for title, line, body in sections]


def expand_section_numbers(spec):
    """'§8.31~§8.33·§8.40' → ['8.31', '8.32', '8.33', '8.40']. 범위는 같은 절 안에서만 펼친다."""
    numbers = []
    for part in re.split(r"[·,]", spec):
        ends = re.findall(r"\d+(?:\.\d+)*", part)
        if len(ends) == 2 and "." in ends[0] and ends[0].split(".")[0] == ends[1].split(".")[0]:
            major, low = ends[0].split(".")[:2]
            high = ends[1].split(".")[1]
            numbers += [f"{major}.{n}" for n in range(int(low), int(high) + 1)]
        else:
            numbers += ends
    return numbers


class DocGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.section_ids = {}  # (파일, 절 번호) → 노드 id
        self.repo_files = {}  # 파일 이름 → 저장소 경로

    def add_node(self, node_id, label, file_type, source_file, location=None):
        self.nodes.setdefault(node_id, {
            "id": node_id, "label": label, "file_type": file_type,
            "source_file": source_file, "source_location": location,
        })
        return node_id

    def add_edge(self, source, target, relation, source_file):
        if source != target:
            self.edges.setdefault((source, target, relation), {
                "source": source, "target": target, "relation": relation,
                "confidence": "EXTRACTED", "source_file": source_file, "weight": 1.0,
            })

    def file_node(self, rel):
        kind = {".py": "code", ".png": "image", ".pdf": "paper"}.get(Path(rel).suffix, "document")
        return self.add_node("file_" + re.sub(r"\W", "_", rel), rel, kind, rel, "L1")

    def deleted_node(self, number, source_file):
        label = f"docs/{number} (삭제됨, DELETED_DOCS.md)"
        return self.add_node(f"deleted_doc_{number}", label, "document", source_file)

    def read_documents(self):
        paths = sorted((ROOT / "docs").rglob("*.md")) + sorted((ROOT / "docs").rglob("*.html"))
        paths.append(ROOT / "CLAUDE.md")
        documents = []
        for path in paths:
            rel = path.relative_to(ROOT).as_posix()
            if rel.startswith("docs/thetics/"):
                continue  # gitignore 대상 원문 PDF 폴더. 다른 기기에 없어 그래프가 어긋난다.
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as error:
                print(f"[건너뜀] {rel}: {error}", file=sys.stderr)
                continue
            if path.suffix == ".html":
                text = html_to_markdown(text)
            documents.append((rel, split_sections(text)))
        return documents

    def link_section(self, rel, section_id, body):
        """본문의 참조를 간선으로 만든다. docs/NN § 를 먼저 지워야 남은 § 가 같은 문서 참조로 읽힌다."""
        def to_doc(match):
            self.link_target(f"docs/{match.group(1)}", match.group(1), match.group(2), section_id, rel)
            return " "

        def to_master(match):
            self.link_numbered(MASTER, match.group(1), section_id, rel)
            return " "

        body = DOC_REF.sub(to_doc, body)
        body = MASTER_REF.sub(to_master, body)
        for number in LOCAL_REF.findall(body):
            target = self.section_ids.get((rel, number))
            if target:
                self.add_edge(section_id, target, "references", rel)
        for name in FILE_REF.findall(body):
            self.link_file(name, section_id, rel)
        for code in set(CONCEPT.findall(body)):
            self.add_edge(section_id, self.add_node("concept_" + code, code, "concept", rel), "mentions", rel)
        for word in KEYWORDS:
            if word in body:
                concept = self.add_node("concept_" + re.sub(r"\W", "_", word), word, "concept", rel)
                self.add_edge(section_id, concept, "mentions", rel)

    def link_target(self, _prefix, number, spec, section_id, rel):
        if number in DELETED_NUMBERS:
            self.add_edge(section_id, self.deleted_node(number, rel), "references_deleted", rel)
            return
        target_rel = next((path for path in self.repo_files.values()
                           if path.startswith(f"docs/{number}_") and path.endswith(".md")), None)
        if target_rel:
            self.link_numbered(target_rel, spec, section_id, rel)

    def link_numbered(self, target_rel, spec, section_id, rel):
        numbers = expand_section_numbers(spec)
        targets = [self.section_ids[(target_rel, n)] for n in numbers if (target_rel, n) in self.section_ids]
        for target in targets or [self.file_node(target_rel)]:
            self.add_edge(section_id, target, "references", rel)

    def link_file(self, name, section_id, rel):
        path = self.repo_files.get(name)
        prefix = re.match(r"(\d{2})_", name)
        if path:
            self.add_edge(section_id, self.file_node(path), "references", rel)
        elif name.endswith(".md") and prefix and prefix.group(1) in DELETED_NUMBERS:
            self.add_edge(section_id, self.deleted_node(prefix.group(1), rel), "references_deleted", rel)

    def build(self):
        tracked = subprocess.run(["git", "-c", "core.quotepath=off", "ls-files", "--cached", "--others",
                                  "--exclude-standard"], cwd=ROOT, capture_output=True, text=True,
                                 encoding="utf-8", check=True).stdout.splitlines()
        self.repo_files = {Path(p).name: p for p in tracked if (ROOT / p).exists()}
        documents = self.read_documents()
        # 앞 문서가 뒤 문서의 절을 가리킬 수 있으므로 절 노드를 전부 만든 뒤 참조를 잇는다.
        plan = []
        for rel, sections in documents:
            file_id = self.file_node(rel)
            for title, line, body in sections:
                section_id = file_id
                if title:
                    label = f"{Path(rel).name} > {title[:90]}"
                    section_id = self.add_node(f"{file_id}_L{line}", label, "document", rel, f"L{line}")
                    self.add_edge(file_id, section_id, "contains", rel)
                    number = SECTION_NUMBER.match(title)
                    if number:
                        self.section_ids.setdefault((rel, number.group(1)), section_id)
                plan.append((rel, section_id, (title + "\n" + body) if title else body))
        for rel, section_id, body in plan:
            self.link_section(rel, section_id, body)
        return {"directed": False, "multigraph": False, "graph": {},
                "nodes": list(self.nodes.values()), "links": list(self.edges.values())}


def self_check():
    """파싱이 깨지면 그래프가 조용히 비므로, 쓰기 전에 핵심 규칙을 확인한다."""
    assert expand_section_numbers("§8.31~§8.33·§8.40") == ["8.31", "8.32", "8.33", "8.40"]
    assert [s[0] for s in split_sections("# 제목\n```\n# 코드\n```\n## 8.1 절")] == ["", "제목", "8.1 절"]
    assert CONCEPT.findall("E10 과 RQ5, F1 0.99, x.E1, 1E5") == ["E10", "RQ5"]
    assert html_to_markdown("<style>h1{}</style><h2 id='a'>제목</h2><p>본문</p>").split() == ["##", "제목", "본문"]


def main():
    self_check()
    graph = DocGraph().build()
    node_ids = {node["id"] for node in graph["nodes"]}
    dangling = [edge for edge in graph["links"] if edge["target"] not in node_ids or edge["source"] not in node_ids]
    if dangling:
        raise SystemExit(f"끊어진 간선 {len(dangling)}건: {dangling[:3]}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(graph, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"노드 {len(graph['nodes'])}개, 간선 {len(graph['links'])}개 → {OUTPUT.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
