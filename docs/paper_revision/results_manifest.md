# 원고 수치 출처표 (results manifest)

> 대상 원고: `docs/paper_write/draft_kiisc_submission.md`
> 작성: 2026-10-11 · 판정 규칙: `docs/paper_revision/03_prereg_v2.md`
>
> **이 표가 하는 일**: 초록·서론·결론에 나오는 숫자마다 "어느 결과 파일의 어느 키에서 왔고, 어느 기록 절에 풀이가 있는지"를 한 줄로 잇는다.
> 예: 초록의 "0.27배"가 맞는지 확인하려면 아래 2절의 첫 줄을 보고 `latency_fair_srbh_4class.json` 의
> `cpu[0].judgment.speedup_formula.fair_e2e_real_p50["pi_0.033"]` 을 읽으면 0.2684 가 나온다.

## 0. 읽는 법과 주의

- 결과 파일은 전부 `experiments/results/` 아래에 있고 **`.gitignore` 대상이라 저장소에 없다.** 다른 기기에서는 3절의 명령으로 다시 만들어야 한다.
  재생성한 값은 cuDNN 비결정성과 지연 측정의 실행 간 차이 때문에 소수 셋째 자리 근처에서 달라질 수 있다(지연은 실행 간 4~6%).
- "키" 열에 경로가 적힌 줄은 2026-10-11 에 이 파일들을 직접 열어 값을 대조한 것이다. "—" 인 줄은 기록 절의 표를 출처로 삼았고 키 경로는 대조하지 않았다.
- 원고의 숫자는 반올림 값이다. 반올림 전 값은 "값" 열에 적었다.
- 새 수치를 원고에 넣을 때는 이 표에 먼저 한 줄을 추가한다. 표에 없는 숫자는 출처를 확인하지 못한 숫자로 본다.

## 1. 이전 데이터셋과 기존 측정 방식(채움·순전파만)

| 원고의 수치 | 값 | 결과 파일 | 키 | 기록 |
|---|---|---|---|---|
| 이전 데이터셋 5.07배 | 5.0729 | `payload_4class_csicnorm_cascade_raw_rgb_b0.1_bal.json` | `latency.speedup_vs_stage2_only` | docs/09 §9.2 |
| SR-BH 1.22배 (언더샘플링 1차, 시험 비율) | 1.2236 | `tradeoff_srbh_4class_cnn_raw_rgb_bal_len2304_nall_s42_dcuda-cpu_b1-32-128-1024.json` | `headline.speedup` | docs/14 §8.3 |
| 0.99배 (같은 구성, 공격 3.3%) | 0.9878 | 같은 파일 | `headline.reweighted[]` 중 `prevalence` 0.033 인 행의 `speedup` | docs/14 §8.3 |
| 1.63배 · 1.42배 (클래스 가중치 1차, Table 3) | 1.63 · 1.42 | 위 두 `tradeoff_*` 파일 | — (두 구성에 같은 지연 측정값을 넣어 계산한 값) | docs/14 §8.11 |
| 1.67배 · 1.45배 (Table 3 각주) | 1.6714 · 1.4477 | `tradeoff_srbh_4class_cnn_raw_rgb_s2bal_len2304_nall_s42_dcuda-cpu_b1-32-128-1024.json` | `headline.speedup` · `headline.reweighted[]` | docs/14 §8.11 |

⚠️ **5.07배의 파일명에 주의.** 접미사 없는 `..._raw_rgb_bal.json` 은 교사 일치 규칙으로 고른 운영점(넘김 27.7%, 2.52배)이다.
5.07배는 넘김 예산 10% 규칙의 `_b0.1` 파일이다(넘김 7.99%).

⚠️ **1.42 와 1.45 는 둘 다 맞다.** 1.42 는 두 구성을 같은 지연 측정값으로 맞춰 비교한 값(Table 3 본문), 1.45 는 클래스 가중치 구성을 따로 돌린 실행의 값(각주)이다.
원고 5.14절의 "0.99배 → 1.42배, +43%" 는 앞의 것을 쓴다. docs/14 §8.11·§8.28 은 뒤의 것(1.448)으로 적혀 있다.

## 2. 공정 지연 기준(전처리 포함·실제 길이)과 1차 대안

| 원고의 수치 | 값 | 결과 파일 | 키 | 기록 |
|---|---|---|---|---|
| 1차 1.311 ms | 1.311 | `latency_fair_srbh_4class.json` | `cpu[0].components.stage1_e2e.p50` (`threads` = 1) | docs/14 §8.21 |
| └ 이미지 변환 0.407 ms · 1차 추론 0.879 ms | 0.4067 · 0.879 | 같은 파일 | `cpu[0].components.prep_img.p50` · `fwd1.p50` | docs/14 §8.21 |
| 2차 0.393 ms | 0.3931 | 같은 파일 | `cpu[0].components.stage2_real_e2e.p50` | docs/14 §8.21 |
| 캐스케이드 0.27배 | 0.2684 | 같은 파일 | `cpu[0].judgment.speedup_formula.fair_e2e_real_p50["pi_0.033"]` | docs/14 §8.21 |
| 넘김이 0 이어도 0.30배가 한계 | 0.2998 | 같은 파일 | `cpu[0].judgment.upper_bound_real` | docs/14 §8.21 |
| 철회 판정 | true | 같은 파일 | `cpu[0].judgment.e1a_withdraw` | docs/14 §8.21 |
| 운영점 τ · r_A · r_N · E_π | 0.9562 · 0.2658 · 0.3950 · 0.3908 | 같은 파일 | `tau` · `r_A` · `r_N` · `e_pi` | docs/14 §8.21 |
| 1차원 합성곱 1차 +3.9%p | 0.9341 대 0.8949 | `cv_srbh_4class_cnn_1d_raw_rgb_sealed.json` · `cv_srbh_4class_cnn_raw_rgb_sealed.json` | `summary.macro_f1` | docs/14 §8.25 |
| └ Holm p = 0.009 | 0.009 | — | — (fold 별 Macro-F1 로 Nadeau–Bengio 보정 t 를 계산하고 세 후보에 Holm 을 적용. `cv_compare.py` 는 보정 전 t 라 쓰지 않았다) | docs/14 §8.25 |
| └ 1차 지연 1.205 · 1.234 ms (두 실행) | 1.205 · 1.234 | `latency_stage1_variants_srbh_4class.json` | `stage1_e2e_ms.cnn_1d` (실행마다 덮어쓴다 — 첫 실행 값은 docs/14 에만 남아 있다) | docs/14 §8.25 |
| └ 배수 0.28배 | 0.2809 | `linear_stage1_srbh_4class.json` | `stage1.cnn_1d.val_mix.b0.speedup` | docs/14 §8.25 |
| └ 넘김 43.7% → 29.8% | 0.4369 → 0.2984 | 같은 파일 | `stage1.rgb_cnn.val_mix.b0.e_pi` · `stage1.cnn_1d.val_mix.b0.e_pi` | docs/14 §8.25 |
| TF-IDF 1차 1.24배 | 1.2443 | 같은 파일 | `stage1.tfidflr.val_mix.b0.speedup` | docs/14 §8.26 |
| └ 실사용자 정상 요청 0.86배 | 0.8578 | 같은 파일 | `stage1.tfidflr.real_user.b0.speedup` | docs/14 §8.26 |
| └ 이미지 1차의 4.7배 | 1.2443 / 0.2652 = 4.69 | 같은 파일 | `stage1.tfidflr.vs_rgb_cnn_val_mix` (`speedup_gain` 3.69 = +369%) | docs/14 §8.26 |
| └ TF-IDF 1차 지연 0.295 ms | 0.2953 | 같은 파일 · `linear_stage1_latency_srbh_4class.json` | `stage1.tfidflr.ms1` · `tfidflr.p50` | docs/14 §8.26 |
| Q 통계량 0.954 대 0.996 | 0.9542 · 0.9964 | `linear_stage1_srbh_4class.json` | `stage1.rgb_cnn.q_with_stage2` · `stage1.tfidflr.q_with_stage2` | docs/14 §8.26 |
| "필요조건이 아니다" 판정 | true | 같은 파일 | `stage1.tfidflr.vs_rgb_cnn_val_mix.heterogeneity_not_necessary` | docs/14 §8.26 |
| TF-IDF 1차 + 클래스별 임계값 1.31배 (탐색) | 1.3061 | 같은 파일 | `stage1.tfidflr.real_user.g2.speedup` | docs/14 §8.26 |

## 3. 넘김 판단

| 원고의 수치 | 값 | 결과 파일 | 키 | 기록 |
|---|---|---|---|---|
| 확신도 게이트가 이상적 라우터의 10.5배 | 10.53 | `gate_cv_cvprobs_srbh_4class_cnn_raw_rgb_sealed_vs_cvprobs_srbh_4class_charcnn_raw_sealed.json` | — | docs/14 §8.4 |
| 거리 점수는 넘김 46~128% 증가 | 46~128% | `gate_g6_cv_cvprobs_srbh_4class_cnn_raw_rgb_sealed_embtr.json` | — (`summary.gates`) | docs/14 §8.15 |
| 클래스별 임계값, 세 번의 확인 모두 경계 위 | 3/3 | `gate_confirm_srbh_4class_s43.json` · `_s44` · `_s45` | `primary.val_mix.passed` · `primary.real_user.passed` | docs/14 §8.23·§8.24 |
| └ p = 0.17~0.41 | 0.17~0.41 | 같은 세 파일 | `primary.{val_mix,real_user}.corrected_p` (seed 43 · 검증 분포 = 0.318) | docs/14 §8.24 |
| 실사용자 넘김률이 검증보다 약 25%p 높음 | +24.9%p | `gate_g4_e1a.json` | — (`gates.b0`) | docs/14 §8.15 |
| 1차 학습 방식의 교차검증 재평가 +27% (p = 0.11) | 1.054 → 1.343, 0.109 | `linear_stage1_srbh_4class.json` | `t11_stage1_training.by_latency["CPU b1, forward only, padded 2304 (old T2)"].val_mix` | docs/14 §8.28 |
| └ 실사용자 +4.9% (p = 0.70) | 0.967 → 1.014 | 같은 파일 | 같은 키의 `.real_user` | docs/14 §8.28 |
| └ 언더샘플링 1차 단독 0.8178 | 0.8178 | 같은 파일 · `cv_srbh_4class_cnn_raw_rgb_sealed_bal.json` | `t11_stage1_training.solo_macro_f1_undersampled` | docs/14 §8.28 |
| 확신 오답의 70.9% 가 라벨 불일치 (대조 31.8%) | 78/110 (wrong 64 + doubtful 14) · 대조 35/110 | `docs/audit/srbh_tail_label_audit.csv` (**추적 파일**) | `group` 별 `judgment` 집계 | docs/14 §8.10 |
| seed 42~45 단독 Macro-F1 (5.12절) | 0.8949 · 0.9229 · 0.9054 · 0.9080 | `cv_srbh_4class_cnn_raw_rgb_sealed.json` · `_s43` · `_s44` · `_s45` | `summary.macro_f1` | docs/14 §8.24 |
| 템플릿 묶음 분할 −0.06%p · +0.10%p (5.7절) | — | `cv_srbh_4class_{cnn_raw_rgb,charcnn_raw}_sealed_gtpl.json` | `summary.macro_f1` | docs/14 §8.27 |

## 3b. 딥러닝 캐스케이드(DL-B · DL-C · DL-A)와 CmdI 오답 분해 (5.5절 · 5.6절)

| 원고의 수치 | 값 | 결과 파일 | 키 | 기록 |
|---|---|---|---|---|
| char-CNN 합성곱 비중 91.0% · 고정비 8.5% | 0.3271 / 0.3516 · 0.0307 / 0.3609 | `latency_charcnn_breakdown_srbh_4class.json` | `components` (`conv3`·`conv5`·`conv7`, `enc`·`floor`) | docs/14 §8.31 |
| DL-B (`charcnn_tiny` → `char-CNN`) 합산 2.13배 (산술 2.39배), F1 0.9839 (−0.45%p) | 2.13 · 2.39 · 0.9839 | `dl_cascade_srbh_4class.json` | `dl_b.real_user.b0` | docs/14 §8.32 · 04_status §1.2 |
| DL-C (`charcnn_ee`) seed 42 합산 3.52배 (산술 3.53배), F1 0.9874 (−0.10%p) | 3.52 · 3.53 · 0.9874 | `dl_cascade_srbh_4class.json` | `dl_c.real_user.b0` | docs/14 §8.32 · 04_status §1.2 |
| DL-C seed 43 확인 3.62배, F1 0.9895 (+0.33%p) | 3.617 · 0.9895 | `dl_cascade_srbh_4class_s43.json` | `dl_c.real_user.b0` | docs/14 §8.33 |
| 대조 `charcnn_k3` 단독 3.62배, F1 0.9748 (−1.36%p) | 3.617 · 0.9748 | `dl_cascade_srbh_4class.json` · `cv_srbh_4class_charcnn_k3_raw_sealed.json` | `summary.macro_f1` | docs/14 §8.33 |
| DL-A 지연: char-CNN 0.563 ms · DistilBERT 45.15 ms (80.2배) | 0.563 · 45.15 · 80.19 | `dl_a_latency_srbh_4class.json` | `latency.ms.t2` · `latency.ms.t_R` | docs/14 §8.34 |
| DistilBERT 단독 F1: seed 42 0.9914 · seed 43 0.9947 | 0.9914 · 0.9947 | `dl_a_srbh_4class.json` · `dl_a_srbh_4class_s43.json` | `distilbert_macro_f1` | docs/14 §8.34·§8.35 |
| DL-A (`char-CNN` → `DistilBERT`, B0) seed 42: 합산 12.3배 (산술 51.6배), F1 0.9911 (−0.03%p), r_N 7.1% · r_A 0.7% | 12.32 · 51.6 · 0.9911 · 0.071 · 0.007 | `dl_a_srbh_4class.json` | `e10_3.real_user.pooled_speedup.b0` · `e10_3.real_user.b0` | docs/14 §8.34·§8.35 |
| DL-A (B0) seed 43: 합산 10.6배 (산술 38.3배), F1 0.9914 (−0.32%p, 보정 p = 0.038) | 10.56 · 38.3 · 0.9914 · −0.32 | `dl_a_srbh_4class_s43.json` | `e10_3.real_user.e11_vs_r` | docs/14 §8.35 |
| DL-A + 클래스별 τ (G2, 병기): seed 42 합산 51.1배 (F1 0.9920) · seed 43 합산 42.5배 (F1 0.9920, −0.27%p) | 51.1 · 42.5 · 0.9920 | `dl_a_srbh_4class{,_s43}.json` | `e10_3.real_user.pooled_speedup.g2` · `e10_3.real_user.g2` | docs/14 §8.34·§8.35 |
| 오답 분해(5.6절): 차단 실패 791→736 (s42) · 839→777 (s43), 공격 종류 혼동 266→167 (s42) · 359→59 (s43) | — | `dl_a_errors_srbh_4class.json` | `s42.{charcnn,distilbert}` · `s43.{charcnn,distilbert}` | docs/14 §8.36 |
| └ SQLi→CmdI 오탐: seed 42 192→99 (−48.4%) · seed 43 252→2 (−99.2%), CmdI F1 0.9682→0.9784 · 0.9618→0.9903 | — | 같은 파일 | `sqli_as_cmdi` · `class_f1.CommandInjection` | docs/14 §8.36 |
| └ DistilBERT 교정 오답의 char-CNN 확신도 중앙값 0.775 · 0.766, ≥0.99 비율 15.1% · 10.7% (전체 97.0%) | 0.775 · 0.766 · 0.151 · 0.107 · 0.970 | 같은 파일 | `fixed_charcnn_conf_median` · `fixed_charcnn_conf_ge_confident` · `all_charcnn_conf_ge_confident` | docs/14 §8.36 |

## 3c. 2026-10-31 후속 (E12 · E13 · E14 · E15 · A2)

지연은 E12 실행(`dl_a_latency_srbh_4class_tok512-128.json`, t2 0.604 · t_front 0.159 · t_R 512 46.10 · 128 46.08 ms) 값만 쓴다.

| 원고의 수치 | 값 | 결과 파일 | 키 | 기록 |
|---|---|---|---|---|
| E12 지연: DistilBERT 512 46.10 ms · 128 46.08 ms | 46.10 · 46.08 | `dl_a_latency_srbh_4class_tok512-128.json` | `latency.ms.t_R` · `latency.ms.t_R_tok128` | docs/14 §8.37 |
| E12-1 128 vs 512: seed 42 0.9718 vs 0.9914 (−1.96%p, p 0.006) · seed 43 0.9777 vs 0.9947 (−1.70%p) | — | `dl_a_srbh_4class_tok128{,_s43}.json` | `distilbert_macro_f1` · `e12_1` | docs/14 §8.37 |
| E12-2 CmdI F1 512→128 · SQLi→CmdI · 잘린 행 정답률 | 0.9784→0.9038 · 99→847 (s42) | `dl_a_errors_srbh_4class_tok128.json` | `s42.by_tokens.{512,128}` · `s43.by_tokens` | docs/14 §8.37 |
| E12-3 char-CNN → DistilBERT-128 vs R128: 합산 67.5배 · +1.66%p (s42), 76.3배 · +0.84%p (s43) | 67.5 · 76.3 | `dl_a_srbh_4class_tok128{,_s43}.json` | `e10_3.real_user.e11_vs_r` | docs/14 §8.37 |
| └ 병기: 512 캐스케이드 vs R128 12.2배 · +1.93%p | 12.22 · 1.93 | 같은 파일 | `bystander_cascade_full_vs_r` | docs/14 §8.37 |
| 캐스케이드(B0) SQLi→CmdI: 192→123→99 (s42) · 252→113→2 (s43) | — | `dl_a_errors_srbh_4class_tok128.json` | `s42.by_tokens.512.cascade_b0` | docs/14 §8.37 |
| E14 k3 출구 → DistilBERT: 합산 29.3배 vs DL-A 12.2배 (×2.40), F1 −0.28%p (s42) · 23.8 vs 10.5 (×2.27), +0.05%p (s43) | — | `dl_a_e14_srbh_4class{,_s43}.json` | `vs_dl_a_b0` · `e14.pooled_speedup` · `dl_a.pooled_speedup` | docs/14 §8.38 |
| A2 fold 4: 예산 3.0% · τ 0.985 · 실사용자 r_N 33.6% [25.4, 43.0] | — | `dl_a_fold_diag_srbh_4class.json` | `folds[3]` | docs/14 §8.39 |
| E13 G2 vs B0 합산 배수: s44 76.0→59.6배(−21.6%, p 0.34) · s45 42.4→54.4배(+28.4%, p 0.54) | — | `dl_a_e13_srbh_4class_s44.json` · `_s45.json` | `pooled_gain` · `operating.pooled_speedup` · `e_pi_diff_corrected` | docs/14 §8.40 |
| E15 r_A 배율 평균 B0 11.0 · G2 2.78 · G2/B0 0.25 | 0.252 | `dl_a_slowdown_srbh_4class.json` | `mean_inflation` · `E15.g2_over_b0` | docs/14 §8.41 |
| E15 url_encode: B0 r_A 0.27→8.76% · 지연 0.73→4.64ms · 1차 공격 판정률 99.4→86.7% | — | 같은 파일 | `per_mutation.url_encode` | docs/14 §8.41 |

## 4. 다시 만드는 순서

데이터 트랙과 층 CSV 가 먼저 있어야 한다(`CLAUDE.md` "데이터 파이프라인"의 `srbh_4class` 순서). 그다음은 아래 순서다. GPU 는 1대이므로 학습은 하나씩 돌린다.

```bash
# (1) 교차검증 확률 — 1차·2차, seed 42 (게이트 비교와 2절의 기준선)
python src/eval/cross_validate.py --track srbh_4class --model cnn     --channels rgb --exclude-test --save-probs
python src/eval/cross_validate.py --track srbh_4class --model charcnn                --exclude-test --save-probs

# (2) 공정 지연 기준선 → latency_fair_srbh_4class.json
python src/eval/latency_fair.py --gpu
python src/eval/latency_fair.py --stage1-variants   # latency_stage1_variants_*.json

# (3) 선형 1차와 1차 대안 → linear_stage1_*.json, 그림 stage1_alternatives_frontier_*.png
python src/eval/linear_stage1.py --fit
python src/eval/linear_stage1.py --latency
python src/eval/linear_stage1.py --analyze

# (4) 유사 연구 비교 그림 → docs/figures/cascade/literature_comparison_srbh_4class.png
python src/eval/literature_compare.py
# 딥러닝 모델만 남긴 판 → docs/figures/cascade/literature_comparison_dl_only_srbh_4class.png
python src/eval/literature_compare.py --dl-only
```

딥러닝 캐스케이드(E8~E16, 2026-10-25~31)는 다음 순서다. DistilBERT 는 fold 당 약 23분(GPU)이다.

```bash
# (5) 딥러닝 모델 교차검증 확률 — seed 42 기준, 확인 seed 는 --seed 43|44|45 를 붙여 반복
DB="--model distilbert --track srbh_4class --exclude-test --save-probs --epochs 3 --batch-size 16 --lr 2e-5"
python src/eval/cross_validate.py --model charcnn_ee   --track srbh_4class --exclude-test --save-probs   # DL-C (+ _aux)
python src/eval/cross_validate.py --model charcnn_tiny --track srbh_4class --exclude-test --save-probs   # DL-B
python src/eval/cross_validate.py --model charcnn_k3   --track srbh_4class --exclude-test --save-probs   # E9 대조
python src/eval/cross_validate.py --model charcnn_k3   --track srbh_4class --exclude-test --save-probs --patience 10  # E16(2026-10-31 미실시)
python src/eval/cross_validate.py $DB                     # DL-A 2차 (E10)
python src/eval/cross_validate.py $DB --max-tokens 128    # E12 원 설정
python src/eval/cross_validate.py --model charcnn --track srbh_4class --exclude-test --save-probs --save-models  # E15 가중치(_ckpt)

# (6) 지연(조용한 기기, 다른 작업 없이) → dl_cascade_latency_* · dl_a_latency_*
python src/eval/dl_cascade.py --latency
python src/eval/dl_a.py --latency                          # E10
python src/eval/dl_a.py --latency --max-tokens 512,128     # E12 (E12~E15 가 쓰는 실행)
python src/eval/dl_a.py --latency --threads 4              # B3(2026-10-31 미실시)

# (7) 판정 — 학습 없음
python src/eval/dl_cascade.py --analyze [--seed 43]        # DL-B·DL-C
python src/eval/dl_a.py --analyze [--seed 43] [--max-tokens 128]   # DL-A · E11 · E12
python src/eval/dl_a_followup.py --e14 [--seed 43]          # E14
python src/eval/dl_a_followup.py --fold-diag                # A2
python src/eval/dl_a_followup.py --e13 --seed 44            # E13 (45 도)
python src/eval/dl_a_followup.py --e16                      # E16(미실시 — 산출물 없음)
python src/eval/dl_a_slowdown.py                            # E15 (--limit 50 은 저장 없는 스모크)
```

- 한글 콘솔에서는 각 명령 앞에 `PYTHONIOENCODING=utf-8` 을 붙인다.
- (1)의 seed 43~45·1차 재설계 후보·템플릿 묶음·언더샘플링 판본은 같은 명령에 `--seed 43`, `--model cnn_1d|cnn_crop8|cnn_gmp`, `--group-template`, `--balance` 를 바꿔 넣어 돌린다(2026-10-11 의 `--help` 기준 — 코드가 바뀌었으면 `--help` 를 먼저 볼 것).
- `linear_stage1.py --analyze` 는 (1)·(2)와 선형 1차의 교차검증 확률이 모두 있어야 돈다.

## 5. 측정 환경 (이 기기 한정 — 다른 기기에서는 다시 확인)

아래는 **2026-10-11 에 데스크톱 한 대에서 잰 값**이다. 다른 기기의 지연은 절대값이 다르므로 이 표의 ms 를 그대로 인용하지 않는다.

| 항목 | 값 | 확인 방법 |
|---|---|---|
| 측정 일시 · 커밋 | 2026-10-11T12:56:07 · `2a456d7` | `latency_fair_srbh_4class.json` 의 `env.date` · `env.git_hash` |
| CPU | AMD Ryzen 7 9800X3D (8코어 · 논리 16) | `env.processor` · `env.cpu_count`, PowerShell `Get-CimInstance Win32_Processor` |
| 메모리 | 31.6 GB | `env.ram_gb` |
| GPU | NVIDIA GeForce RTX 4080 SUPER | `env.gpu`, `python -c "import torch; print(torch.cuda.get_device_name(0))"` |
| OS | Windows 11 (10.0.26300) | `env.platform` |
| Python · torch | 3.12.3 · 2.6.0+cu124 | `python -V` · `env.torch` |
| scikit-learn · numpy · scipy | 1.5.1 · 1.26.4 · 1.13.1 | `python -c "import sklearn, numpy, scipy; print(sklearn.__version__, numpy.__version__, scipy.__version__)"` |
| 측정 설정 | 요청 2,000건 × 30회 반복, 스레드 1·4·8, 대표값은 스레드 1 의 p50 | `n` · `repeats` · `env.args.threads` |
| 측정 표본의 바이트 길이 | p50 97 · p95 233 · p99 349 | `sample_byte_length` |
| `xgboost` | 설치돼 있지 않음 | `python -c "import xgboost"` — 유사 연구의 XGBoost 를 재구현하지 않은 이유(원고 5.13절) |

## 6. 이 표가 보증하지 않는 것

- **참고문헌 [34]**(Kuncheva & Whitaker, *Machine Learning* 51(2), 2003)의 권·호·쪽은 원문을 열어 대조하지 않았다 → `docs/paper_write/README.md` 의 제출 전 확인 목록에 있다.
- **유사 연구의 수치**(원고 Fig. 5 의 (a) 패널)는 각 논문이 보고한 값을 옮긴 것이고, 데이터·장치·지연 정의가 서로 다르다. 같은 조건의 비교는 (b)·(c) 패널뿐이다. 출처와 조건은 docs/14 §8.26 과 `src/eval/literature_compare.py` 의 표에 있다.
- **1차원 합성곱 1차와 선형 1차는 seed 42 한 번**의 교차검증이다. seed 간 1차 Macro-F1 의 표준편차가 1.15%p 이므로(5.12절) 2%p 안팎의 차이는 이 표로 확정되지 않는다.
- **2인 라벨 감사**는 하지 않았다. 70.9% 는 대형 언어모델 한 판정자의 값이다.
