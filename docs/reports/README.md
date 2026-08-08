# docs/reports — 외부 제출용 보고서

교수님 제출·미팅 배포처럼 **저장소 밖으로 나가는 문서**를 둔다.
설계·의사결정 기록(`docs/0N_*.md`)과 달리 **읽는 사람이 저장소를 모른다고 가정**하고 쓴다.

## 규칙

- **HTML 이 원본, PDF 는 산출물**이다. 내용 수정은 반드시 `.html` 을 고치고 PDF 를 다시 굽는다.
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

## 목록

| 파일 | 내용 | 기준일 |
|---|---|---|
| `progress_2026-08-08.{html,pdf}` | 캐스케이드 전환 이후(Phase 10~12) 진행 보고 + 여쭐 것 8건. 근거: docs/09·10·11·12 | 2026-08-08 |
