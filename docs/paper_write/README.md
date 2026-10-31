# 논문 초안 (정보보호학회논문지 투고)

- `draft_kiisc_submission.md` — 초안 원본(2026-10-31 딥러닝 캐스케이드 집중판, RGB 이미지 실험 제거 및 `char-CNN`·`DistilBERT` 캐스케이드 중심으로 개고, 표 3개·참고문헌 21개). 양식은 `#(투고양식)submission_form_2026.hwp`(심사용, 저자·소속 없음)의 순서·번호·캡션 규칙을 따른다.
  옛 HTML·PDF 판(RGB 캐스케이드 제안 시기)은 삭제했다 — 필요하면 `git log -- docs/paper_write/draft_kiisc_submission.html` 로 되살린다.
- `draft_kiisc_submission.pdf` — 위 초안을 양식 모양(190×260mm, 여백 20mm, 2단·단 간격 10mm, 본문 9pt)으로 찍은 **확인용 사본**.
  초안을 고친 뒤 `python docs/paper_write/build_pdf.py` 로 다시 만든다(Chrome 필요). 신명 글꼴 대신 바탕·맑은 고딕·Times New Roman 을 쓴다.
- 최종 제출본은 HWP 양식 파일에 본문을 옮겨 넣어 만든다(학회 접수는 HWP, 2단 190×260mm). 수식은 HWP 수식 편집기로 다시 입력한다.
- 그림은 `../figures/` 의 PNG 를 HWP 에 직접 넣는다(`build_pdf.py` 는 `【그림: ...】` 구문을 읽어 PDF 에 자동 삽입).
- 게재 확정 후에는 `(게재양식)` 파일로 옮기고 저자·소속·사사를 추가한다.

## 제출 전 확인 목록

- [ ] **교수 피드백 반영** — 받은 뒤 항목으로 풀어 적는다.
- [x] **E12(DistilBERT 128 토큰판) · E13(G2 새 seed 44·45) · E14(k3 출구→DistilBERT) · E15(DL-A slowdown 공격면) · A2(느린 폴드 진단)** — `RESULTS_FOR_WRITING.md` 기준 반영 완료.
- [ ] **새 참고문헌 서지 확인** — [2] Zhang et al.(char-CNN, NIPS 2015 쪽 649-657), [3] Sanh et al.(DistilBERT) 원문 대조.
- [x] **참고문헌 번호** — 첫 인용 순서([1]~[21])로 다시 매겼다. 문단을 옮기면 다시 매긴다.
- [x] **DL 실험의 CV 풀** — IV절의 "학습+검증 226,048행"이 DL-A·B·C 모두 동일함을 `03_prereg_v2.md` §E10 및 `cross_validate.py` 로 확인 완료.
- [x] **Table 1** — `char-CNN` 141,252 · `char-CNN-tiny` 14,692 · `char-CNN-k3` 41,668 · `char-CNN-EE` 141,768 · `DistilBERT` 66.96M 파라미터 및 학습 방식(class-weighted CE) 반영 완료.
- [x] **서론의 사전 고정 커밋 해시** — 실제 커밋(`2a456d7`~`d9d3678`) 반영 완료.
- [x] 주장 강도: "개선"·"해결" 무제한 표현 금지 준수. DL-A 는 seed 42 한정(seed 43 미재현 병기), G2 는 E13 미재현 반영, DL-C 는 "2차 모델 압축"으로, 5.3절 CmdI 오탐 분석은 **사후 분석**이며 "공격을 더 잘 막는다"가 아니라 "SQLi→CmdI 공격 종류 오탐을 줄인다"로 기술.
- [x] 70.9% 라벨 감사는 결과 절에서 빼고 VI절 한계에서만 LLM 1인 판정 기준 "가능성" 수준으로 기술(T8 생략 반영).
- [x] RGB 바이트 이미지 실험(폐기된 초기 가설) 완전 삭제 및 `char-CNN`·`DistilBERT` 딥러닝 캐스케이드 집중 구조(안 A)로 개고 완료.
- [x] 초록·서론·결론의 숫자와 V·VI절 수치를 `RESULTS_FOR_WRITING.md` 및 `docs/paper_revision/results_manifest.md` 에 동기화 완료.
