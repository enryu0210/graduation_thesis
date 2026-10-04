# 논문 초안 (정보보호학회논문지 투고)

- `draft_kiisc_submission.html` — 초안 원본. 양식은 `#(투고양식)submission_form_2026.hwp`(심사용, 저자·소속 없음)를 따랐다.
  `draft_kiisc_submission.pdf` 는 이 HTML 을 Chrome 으로 인쇄한 확인용 사본이다.
- 최종 제출본은 HWP 양식 파일에 본문을 옮겨 넣어 만든다(학회 접수는 HWP). 수식은 HWP 수식 편집기로 다시 입력한다.
- 그림은 `../figures/` 를 참조한다 — HTML 을 다른 위치로 옮기면 그림이 깨진다.
- 게재 확정 후에는 `(게재양식)` 파일로 옮기고 저자·소속·사사를 추가한다.

## PDF 재생성

```bash
WIN=$(cd "$(git rev-parse --show-toplevel)/docs/paper_write" && pwd -W)
"/c/Program Files/Google/Chrome/Application/chrome.exe" --headless --disable-gpu --no-sandbox --no-pdf-header-footer \
  --print-to-pdf="$WIN/draft_kiisc_submission.pdf" "file:///$WIN/draft_kiisc_submission.html"
```

Chrome 설치 경로는 기기마다 다를 수 있다.

## 제출 전 확인 목록

- [ ] **참고문헌 서지 확인** — 아래는 docs/12 의 메모로 채운 것이라 원문과 대조하지 않았다.
  - [4] Eroğlu Demirkan & Aydos (Appl. Sci. 2025) — 저자 이니셜, 권(호)
  - [9] Sevri & Karacan (KSII TIIS 2022) — 쪽 범위 632-657
  - [10] Kutlimuratov et al. (Computers 2026) — 전체 저자, 호
  - [24] Lucz & Forstner (Data 2025) — 저자 이니셜, 정확한 제목
  - [20] Reddi et al., [23] Ovadia et al. — "et al." 대신 전체 저자를 요구하는지 학회 규정 확인
- [ ] Fig. 2 — 두 수평 기준선(RGB CNN·char-CNN)이 같은 색이다. 그림을 다시 그리거나 캡션의 (lower)/(upper) 구분을 유지한다.
- [ ] 5.5절·Table 6 의 운영점 비교는 **사후 탐색**이다(docs/14 §8.6). 심사 과정에서 "개선"으로 승격하지 않는다.
- [ ] 5.3절 헤드라인(1.22배)은 고정 분할 `_bal` 체크포인트, 5.5절(1.78배)은 CV class weight 모델 — 출처를 섞지 않는다.
- [ ] 분량: 학회 투고 규정(쪽수)을 HWP 로 옮긴 뒤 확인한다.
