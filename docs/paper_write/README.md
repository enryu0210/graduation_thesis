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
  - [25]~[29] (거리 기반 점수 Lee·Ren·Sun·Jiang·Jaeger) — docs/12 §1.8 메모로 채움, PDF 미수집. 학회명·연도·쪽 확인
  - [20] Reddi et al., [23] Ovadia et al. — "et al." 대신 전체 저자를 요구하는지 학회 규정 확인
- [ ] Fig. 2 — 두 수평 기준선(RGB CNN·char-CNN)이 같은 색이다. 그림을 다시 그리거나 캡션의 (lower)/(upper) 구분을 유지한다.
- [ ] 5.5절·Table 6 의 운영점 비교와 Table 7 아래쪽(class weight 1차 위 게이트)은 **사후·탐색**이다(docs/14 §8.6·§8.12). 심사 과정에서 "개선"으로 승격하지 않는다.
- [ ] 출처를 섞지 않는다: Table 2·3·7 은 고정 분할(단일 seed), Table 4·5·6 은 CV(class weight 1차). 언더샘플링 1차(`_bal`)와 class weight 1차(`_s2bal` 구성)는 표 안에서 열로 갈라 둔다.
- [ ] Table 3 은 두 구성에 **같은 지연 측정값**(T2 CPU·1 0.350/1.114ms)을 써서 class weight 헤드라인이 1.63·1.42배다. 같은 실행의 도구 출력은 1.67·1.45배(docs/14 §8.11) — 캡션에 병기했고 판정(H-T4a 채택, 2배 미만이라 "조건부")은 어느 쪽이든 같다.
- [ ] 5.7절 라벨 감사는 LLM 1인 판정(`docs/audit/srbh_tail_label_audit.csv`)이다. 전문가 2인 재판정 전에는 "라벨 잡음이 주원인"으로 쓰지 않는다.
- [ ] 분량: 학회 투고 규정(쪽수)을 HWP 로 옮긴 뒤 확인한다.
- [ ] 5.10절 T3 의 TF-IDF "대안 병기"는 F1 0.94 이하 구간 한정이고, TF-IDF 단독이 저정확도 구간을 지배한다는 관찰은 **탐색**이다(docs/14 §8.18). TF-IDF 1차 캐스케이드는 측정하지 않았다 — 심사에서 물으면 후속 과제로 답한다.
