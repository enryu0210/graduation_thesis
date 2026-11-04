# 삭제한 문서 기록 (2026-11-04)

DistilBERT 구성으로 논문 범위를 좁힌 뒤, 투고 초안에 쓰지 않는 옛 문서·그림·보고서를 작업 트리에서 지웠다.
다른 문서가 아래 파일을 가리키면 이 문서로 되살려 읽는다.

## 되살리는 법

삭제 직전 기준 커밋은 **`6780c81`** 이다.

```bash
git checkout 6780c81 -- docs/09_cascade_design.md    # 파일 하나
git checkout 6780c81 -- docs/figures/models          # 폴더 하나
git ls-tree -r --name-only 6780c81 docs/             # 당시 파일 전체 목록
```

⚠️ 이 삭제가 커밋되기 전까지는 `git status` 에도 삭제 목록(`D`)이 그대로 보인다.

## 삭제 목록 (추적 파일 231개)

| 분류 | 대상 | 삭제 이유 |
|---|---|---|
| 옛 Phase 문서 | `docs/01_data_acquisition_plan.md`, `03_preprocessing_imaging_design.md`, `03_preprocessing_notes.md`, `04_models_rq1_design.md`, `05_evasion_attacks_rq2_design.md`, `06_encryption_boundary_rq4_design.md`, `07_related_work_metrics_rgb_design.md`, `08_vit_design.md`, `09_cascade_design.md`, `10_adversarial_defense_rq3_design.md`, `11_cascade_extension_design.md`, `EDA_notes.md`, `paper_revision_plan.md` | 1기·2기 기록. RGB·ViT·RQ2/3/4/6 시기 문서 |
| 지난 기록 | `docs/archive/` 전체 | 지난 시점 기록 |
| 논문 수정 경과 | `docs/paper_revision/01_why_results_low.md`, `02_revision_status.md`, `04_status_2026-10-27.md` | `docs/paper_write/SUBMISSION_ROADMAP.md` 로 대체 |
| 그림 | `docs/figures/{attacks,imaging,models,eda,rq4}/` | 초안 미사용 |
| 그림(캐스케이드 일부) | `docs/figures/cascade/` 의 `tradeoff_budget_*`, `gate_frontier_b0_g2_*`, `stage1_alternatives_frontier_*`, `stage2_candidates_*` | RGB 1차 시기 |
| 보고서 | `docs/reports/progress/progress_2026-{08-08,08-24,09-14,09-15,10-04,10-14}.*`, `docs/reports/summary/` 전체, `docs/reports/pareto_b0_g2.png` | DistilBERT 이전 |
| 인용 논문 사본 | `docs/reports/papers/feature_squeezing_ndss2018.pdf`, `2512.19203v2.pdf`, `s42400-023-00170-z.pdf` | 초안 미인용. 원본 사본은 `docs/thetics/` 에 있다 |

## 복구할 수 없는 삭제

git 에 올라간 적 없는 파일이라 되살릴 수 없다.

- `docs/paper_revision/04_next_actions_plan.md` : 51.6배 등 정정 전 수치가 남은 옛 계획. `05_distilbert_tok128_cmdi_plan.md` 가 대체한다.
- `docs/handoff/` : 2026-09-15 Codex 위임 휘발성 파일 3개. 2026-10-27 이후 쓰지 않는 절차다.

## 남긴 것

- `docs/thetics/` : gitignore 대상이라 지우면 복구할 수 없고, 서지 확인에 필요하다.
- `docs/DATA_MANIFEST.md` : `src/data/validate.py` 가 읽는다.
- `docs/audit/srbh_tail_label_audit.csv` : 초안 한계 절의 근거다.

## 알아둘 점

- `src/`·`tests/` 주석의 `docs/04`~`docs/11` 참조는 고치지 않았다. 이 문서로 되살려 읽는다.
- `python src/data/preprocess.py`, `eda.py`, `train.py`(`--smoke` 없이)는 지운 노트나 그림을 다시 만든다. 다시 생긴 파일은 커밋하지 않는다.
