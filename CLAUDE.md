# 프로젝트: 웹 공격 페이로드 이미지화 CNN 탐지 (졸업논문)

## 워크플로우
- Phase 단위 진행. 각 Phase 의 결정/실측/리스크는 `docs/0N_*_design.md`(설계·의사결정 기록)에 남긴다.
- 실측 지표 → `experiments/results/*.json`(⚠️ 전체 .gitignore, 재생성으로 확보 — 커밋 대상 아님), 그림 → `docs/figures/`(추적·커밋 대상).
- 커밋/푸시 대상 브랜치는 **`main`**. ⚠️ 2026-08-24 이전 문서·커밋에 나오는 `phase1-data-acquisition` 은 **삭제된 옛 이름**이다(전 Phase 가 그 브랜치에 쌓여 있다가 main 으로 fast-forward 통합됨).
- git 커밋 메시지(한글 여러 줄)는 파일로 써서 `git commit -F <file>` 사용. Bash 도구에서 PowerShell here-string(`@'...'@`)은 메시지가 깨짐.
- ⚠️ 노트북·데스크톱 양쪽에서 작업 → **착수 전 `git fetch` 필수**. 원격에 Phase 가 쌓여 있으면 문서 번호·모델 이름이 이미 선점됨(실제 사고: `docs/08` 과 모델명 `hybrid` 충돌 → push 거부 후 리베이스·개명 재작업).
- 커밋 전 `python -m pytest tests/ -q` 실행(~3초, GPU 불필요).
- 모든 작업(코드 포함)은 Claude 가 직접 한다 — 외부 코딩 도구 위임 없음(2026-10-27, 옛 `AGENTS.md`·`docs/handoff/` 삭제).
- ⚠️ `liteLLM` 라이브러리 사용 금지(보안 문제).

## 논문 축 (RQ · 주장 강도)
- **마스터 설계 문서는 `docs/web_attack_image_cnn_thesis_design.md`** — RQ 표·기여 문구·목차의 단일 진실 소스. Phase 문서와 어긋나면 마스터를 고친다.
- 📌 **"지금 무엇이 유효한가"는 `docs/CURRENT_BASELINE.md`** — 1기(gray CNN)·2기(MCC)·3기(Phase 13 개편)를 시기별로 갈라 RQ·지표·데이터셋을 통합 정리한 참조본(§0.2 에 단일 진실 소스 지도). 새 사실은 여기서 만들지 않고 Phase 문서에 먼저 남긴다.
- ⚠️ **RQ 번호는 재배열 금지.** 번호가 파일명에 박혀 있다(`05_..._rq2`, `06_..._rq4`, `10_..._rq3`). 새 질문은 **뒤 번호로 추가**(현행 RQ1~RQ6). 논문 서술 순서만 마스터 §1.3 을 따른다.
- 📌 **논문 제목(2026-10-04)**: *표현 이질적 캐스케이드 웹 공격 탐지: 넘김 판단의 한계와 개선, 배포 조건별 속도–정확도 교환의 재평가*. ⚠️ **RQ4 는 논문 제외**(번호는 결번 보존 — 재사용 금지). 흐름 트랙 `ustc_flow_binary` 는 RQ4b 용이라 현재 논문에 쓰이지 않는다.
- 📌 **3기 연구 초점(2026-09-15 사용자 결정) = RQ5 의 두 축**: ① 캐스케이드 속도·정확도 교환 ② 2차 넘김 판단(게이트) 개선.
  **새 실험·주제를 제안할 때는 이 두 축으로 잡는다.** 실험 목록(T0~T4·G1~G5)·우선순위·판정 기준 초안은 `docs/14_cascade_tradeoff_gate_roadmap.md`. RQ 번호·문구는 불변.
- ⚠️ **RQ1 의 답은 "아니오"다**(이미지 단독은 텍스트 모델에 유의하게 열세, F1). 원 문구와 음성 결과를 **그대로 보고**하고 RQ5(캐스케이드)가 거기서 파생됐음을 명시할 것 — 조용히 비용 질문으로 바꾸면 HARKing 이 된다.
- ⚠️ **주장 강도는 마스터 §11 표를 따른다.** 선행 대조로 이미 하향된 것:
  - 비용 기반 공격(F7) → "새 공격면 **발견**" ❌ / "DeepSloth(ICLR'21) slowdown 공격의 **problem-space 도메인 이전**" ✅
  - 표현 불일치(M1) → "새 발상" ❌ / "Feature Squeezing(NDSS'18) 대비 **추가 비용 정확히 0**" ✅
  - 속도 배수는 **2차 기준 모델을 명시**해 병기(우리 5.07배 vs Tasdemir 20배는 기준이 BERT라 직접 비교 불가).
  - ⚠️ **C1 은 "조건부 이득"으로 하향됐다**(2026-10-04, docs/14 §8.3): 5.07배는 2기 트랙 값이고, SR-BH 에서는 넘김 50% 운영점 · 헤드라인(CPU·배치1) 1.22배 · 배포 공격 비율에서 0.99배(`_bal` 1차). 1차를 class weight 로 바꾸면 1.67배·1.45배(docs/14 §8.11, 아래 확인 실험 종합). 5.07배를 3기 수치 없이 쓰지 말 것.
  - 게이트 신호 교체(msp_T·margin·entropy)는 과잉 넘김을 4~5% 밖에 못 줄인다(G1, H-E5-1 기각, docs/14 §8.4) → 넘김 개선은 G2·G3·T4 로.
  - 📌 **확인 실험 종합(docs/14 §8.13)**: 배포 속도를 올린 것은 **1차 class weight 학습**(H-T4a: 공격 3.3% 0.99→1.45배). 권장 구성 = 1차 `srbh_4class_cnn_raw_rgb.pt`(비균형) + 2차 `_bal` char-CNN(`cascade_tradeoff.py --balance --stage1-no-balance`, tag `_s2bal`).
    학습형 라우터 G3(`gate_g3.py`·`gate_g3_cv.py`)는 CV 배율을 못 줄였다(H-G3F-cv 기각). ⚠️ **게이트는 test 비율 넘김률만 보면 오판한다** — 정상 넘김(r_N)을 늘려 배포 배수가 떨어진 게이트가 기준을 통과했다(H-G2c·H-G3F on `_bal`). 게이트 비교 시 r_N·π 재가중 배수를 반드시 병기.
  - 📌 **2026-10-07 실험 종합(docs/14 §8.14~§8.18)**: 거리 기반 넘김 점수 G6(maha·rmaha·knn·trust, `gate_g6*.py`)은 msp 보다 46~128% 더 넘김 → 기각, "점수는 병목이 아니다"(교수 지적의 직접 답). 1차 지표는 이제 **E_π = 0.033·r_A + 0.967·r_N**.
    실사용자 r_N 은 val 보다 +24.9pp(H-G4 채택) → **실사용자 보정 S(π=3.3%) B0 1.01배·G2 1.28배를 병기**. 1차 폭 ×0.5·×2(H-T4b) 기각. G5 변형 부풀리기: G2/B0 1.16 → 대가 조항 미발동, 단 URL 인코딩이 넘김 ×1.8.
    T3: 2차는 char-CNN 유지. ⚠️ **TF-IDF 단독(0.29ms, F1 0.938)이 1차 RGB CNN 단독·F1 0.94 이하 캐스케이드보다 빠르다** → 캐스케이드 주장은 F1 > 0.94 구간 한정. TF-IDF 1차 캐스케이드는 미측정(후속 후보).
  - 📌 **교수 기준의 "개선"(2026-10-07, docs/14 §8.20)**: 가장 좋은 유사 연구 대비 **정확도(Y)–속도(X) 그래프에서 위**면 "개선"이라 쓸 수 있다(곡선 전체 비교).
    `gate_frontier.py` 탐색에서 G2 가 B0(표준 MaxProb 넘김)보다 속도 1.25배 이상 구간에서 대부분 fold 위 — 단 G2 발견 데이터라 **새 seed 확인 전 "개선" 확정 금지**. GPU·128 지연은 이상치(재측정 전 불사용).
  - 꼬리 라벨 감사(`docs/audit/srbh_tail_label_audit.csv`, LLM 1인 판정): 1차 확신 오답의 70.9% 가 라벨 불일치(대조 31.8%) — SR-BH Normal 에 스캐너 탐침이 남아 있다.
  - 클래스별 τ(G2)도 사전 판정은 **기각**(배율 −18.7%, p 0.42, docs/14 §8.6). 운영점 넘김 −7.6pp·배포 배수 1.34→1.75배는 **사후 탐색**이라 "개선"으로 확정해 쓰지 말 것.
  - ⚠️ T1 운영점 50% 는 `_bal` 체크포인트(F1 0.81) 값이고, CV(class weight, F1 0.89)의 같은 게이트는 ~25% 다 → 두 값을 출처와 함께 병기(docs/14 §8.6).
  - ⚠️ **공정 지연(E1, 2026-10-11, docs/14 §8.21)**: 전처리를 넣고 2차를 실제 길이로 재면 CPU·배치 1 에서 1차 전체 1.311ms > 2차 전체 0.393ms(스레드 1 p50) → 캐스케이드는 char-CNN 단독의 **0.27배**. **"CPU·배치 1 속도 이득" 주장은 철회**(스레드 1·4·8 모두). 위의 1.22배·1.45배·1.01배·1.28배 등은 전부 "추론만·2,304 채움" 값 → **채움 기준이라고 밝혀 병기만**, 헤드라인 금지. 지연은 `latency_fair.py` 로 재고(조용한 기기), `cascade.measure_latency` 값은 채움 기준이다.
  - 📌 **재분석(docs/14 §8.22)**: §8.6 판정의 20% 감소 검정력은 7~10% — "기각"은 "효과 없음"이 아니라 "이 설계로 판정 불가"로 서술. 운영점 실사용자 r_N: B0 69.3% [65.3, 73.0] · G2 49.6% [45.5, 53.8](n=550).
  - 📌 **확인 실험 판정(2026-10-11, docs/14 §8.23~§8.29 · 규칙은 `docs/paper_revision/03_prereg_v2.md`)**: 클래스별 τ(G2)는 seed 43·44·45 **모두 경계가 위**(1차 규칙 3/3)지만 어느 seed 도 유의하지 않고(p 0.17~0.41), 운영점 이득은 판정 seed 43 에서 +15.7%(기준 20% 미달, seed 간 16~44%) → ✅ "경계가 위이나 유의성·운영점 이득 미확인" / ❌ "개선".
    RGB CNN CV 평균은 seed 만 바꿔도 표준편차 1.15%p(범위 2.8%p) — **seed 42 가 네 seed 중 최저**(0.8949, 평균 0.9078). 단일 seed 의 +2%p 차이는 seed 잡음 크기다.
  - 📌 **1차 대안(docs/14 §8.25·§8.26)**: `cnn_1d` 가 E2 세 조건을 모두 채워 **채택**(+3.9%p, Holm p 0.009, 지연 −2~3%) — 넘김 43.7→29.8% 인데 배수는 0.27→0.28배(상한 t2/t1 0.31). 이미지 변환만 0.41ms 로 2차 전체(0.39ms)보다 길어 **같은 변환을 쓰는 1차는 2차보다 빨라질 수 없다**.
    ⚠️ **TF-IDF+LR 1차 → char-CNN 은 1.24배**(검증 분포; **실사용자 0.86배**)로 RGB CNN 1차의 4.7배 → **"표현 이질성은 속도 이득의 필요조건이 아니다"**(E3 판정). Q 통계량 0.954(RGB) vs 0.996(TF-IDF) — 오류는 덜 겹치나 운영점 F1 은 같다. 교수 기준("그래프에서 위")으로 제안 구성은 위가 **아니다** → "개선"이라 쓰지 말 것.
  - 📌 **견고성(docs/14 §8.27·§8.28)**: 템플릿 묶음 CV 에서 점수 유지(T5). 1차 class weight 결정은 CV 에서도 +27%(채움 기준·검증 분포) — test 의 0.99→1.45배(+46%)는 **확인용**으로 내리고, 실사용자 분포(+5%)·공정 지연(+1~6%)에서는 이득이 없다고 병기. 고정 test 는 아홉 번 평가에 쓰였다(감사표 §8.28). T8(2인 라벨 감사)은 미실시.
- 📌 **딥러닝 한정 후속 후보(2026-10-11 등록 · 미측정, docs/14 §8.30)**: 사용자 방향 = 제안 구성은 딥러닝만(고전 ML 은 비교 기준으로만), RGB CNN 은 필수 아님 — ⚠️ 마스터 문서·제목에는 미반영(교수 확인 전).
  `DL-A` char-CNN → 트랜스포머 · `DL-B` 더 작은 문자 모델 → char-CNN · `DL-C` char-CNN 내부 조기 종료. 관문: DL-B·C 는 char-CNN 층별 시간 측정(학습 불필요), DL-A 는 트랜스포머 단독 CV 가 0.9884 를 넘는가. 비교 대상은 char-CNN 단독.
  ✅ **DL-B·C 관문 통과(2026-10-25, E7, docs/14 §8.31)**: `latency_fair.py --charcnn-breakdown` — 호출 고정비는 t2 의 8.5% 뿐, 91% 가 합성곱 → 상한 DL-B 7.24배·DL-C 3.03배(넘김 5.3% 가정). DL-A 관문은 E10 에서 통과(아래).
  📌 **본 측정 E8(2026-10-25, docs/14 §8.32 · `dl_cascade.py --latency|--analyze`)**: DL-C(`charcnn_ee`) **char-CNN 단독보다 위**(실사용자 3.53배, F1 −0.10%p) · DL-B(`charcnn_tiny`) **위가 아니다**(2.39배지만 F1 −0.45%p > 0.3%p).
    ⚠️ DL-C 이득은 넘김이 아니라 **보조 헤드(k3 분기) 자체**에서 나온다 — 5 fold 중 3개가 넘김 0%. "넘김 판단 개선"으로 쓰지 말 것.
    📌 **E9(docs/14 §8.33)**: seed 43 확인 **통과(2/2, 3.62배·F1 +0.33%p, 전 fold 넘김 0%)** · k3 단독 대조 `charcnn_k3` 0.9748(−1.36%p) → E8 미충족 = "결합 학습의 몫"(단 보조 헤드와 차이 p 0.34, k3 단독은 조기 종료로 불안정 — 최적화 안정성일 수 있음).
    → DL-C 는 넘김 축이 아니라 **2차 모델 압축(결합 학습한 k3 출구)**으로 서술. "유사 연구 대비 개선" 금지(그림은 `literature_compare.py --dl-only` 에 점만 추가).
  📌 **DL-A(E10, 2026-10-25, docs/14 §8.34 · `dl_a.py --latency|--analyze`, 모델 `distilbert` = WAMM'25 DistilBERT 같은 fold 재구현)**: 관문 통과(0.9914 > 0.9884, 단 p 0.57). t_R 45.15ms = char-CNN 의 80배(같은 실행).
    ✅ **char-CNN → DistilBERT 캐스케이드는 재구현 DistilBERT 단독보다 위**(실사용자 **평균 비용 기준 12.3배** · F1 −0.03%p, seed 42, fold 4 는 2.95배). ⚠️ 처음 적은 51.6배는 fold 산술평균이라 정정(2026-10-27) — 배수 평균은 `dl_a.pooled_speedup`(조화평균)으로 낸다 — 교수 기준을 **재구현 기준으로** 충족한 첫 구성. char-CNN 단독은 80배지만 −0.303%p 로 0.003%p 차 미달.
    ❌ **확인 seed 43 재현 실패(E11, 2026-10-27, docs/14 §8.35)**: 합산 10.6배·전 fold > 1 은 충족, F1 −0.32%p(허용 0.3%p, 다섯 fold 모두 음수·p 0.038) — DistilBERT 가 0.9914 → 0.9947 로 강해졌고 캐스케이드는 0.9914 에 머묾. → "재구현 DistilBERT 보다 위"는 **seed 42 한정**. seed 43 의 G2(42.5배·−0.27%p)는 판정 대상 밖이라 뒤집는 데 쓰지 말 것.
    ⚠️ 캐스케이드의 char-CNN 대비 몫은 +0.28%p(p 0.34)·지연 평균 6.5배 — "char-CNN 보다 정확"·"원 논문 대비 개선" 금지. ⚠️ `distilbert` 는 토큰 id 입력이라 `cross_validate.TOKEN_MODELS` 에만 있고 `train.TEXT_MODELS` 에는 **일부러 없다**(train.py 는 바이트 경로뿐).
- ⚠️ `docs/thetics/` 는 **.gitignore 대상**(PDF 미추적) → **수집 목록의 유일한 추적본은 `docs/12` §1**. 논문을 추가하면 docs/12 도 함께 갱신.

## 데이터 파이프라인
- `data/processed`, `data/images` 는 .gitignore(대용량). 재생성으로 확보:
  - `python src/data/preprocess.py`  (seed=42 결정론, CSV 3분할 생성)
  - `python src/imaging/build_image_dataset.py --track <트랙>`  (.npz 생성)
- 트랙: `payload_4class`(RQ1 주), `payload_4class_csicnorm`(Normal=CSIC 실트래픽, RQ2용), `csic_binary`.
- **3기 주 트랙 `srbh_4class`**(SR-BH 2020, docs/13 §2.5.3) — 재생성 순서가 고정이다:
  `download_srbh.py` → `audit_srbh_labels.py`(E1-a 목록 `data/processed/srbh_audit_flags.csv`) → `process_track('srbh_4class')` → `build_image_dataset.py --track srbh_4class --channels rgb`.
  ⚠️ 감사 CSV 없이 트랙을 만들면 에러로 멈춘다(라벨 노이즈가 조용히 섞이는 것 방지). 클래스명은 `CodeInjection`(XSS 아님 — 실데이터는 PHP 코드 주입 탐침).
  ⚠️ 현재 트랙 `text_raw` 는 **URI+body(F2)**. E2 단계2(RGB CNN CV)는 F3(+cookie)가 높았지만, 쿠키 내용을 지운 F3M 과 F3 의 차이가 유의하지 않아(T0-a, Holm p 0.17) **F2 유지로 확정**(2026-10-04, docs/14 §8.2). F3 의 이득은 출처(내용 vs 존재·길이) 판정불가로 기록.
  "UA 는 지름길"은 원본 행 기준 관찰이었고 중복 제거 후엔 성립하지 않았다(정정 기록 있음). F3 는 Macro-F1 은 높지만 TPR@1%FPR 은 F2 보다 낮다(**RGB CNN 한정** — char-CNN 은 반대) → 병기.
  ⚠️ 실사용자 층 FPR 이 스캐너 층의 약 8배(char-CNN F2 2.73% vs 0.36%) — 3기 FPR 은 층 병기 없이 쓰지 말 것.
  ⚠️ `run_evasion.py` 에는 이 트랙이 **의도적으로 없다**(`mutations.py` 가 `XSS` 라벨명에 묶임).
  ⚠️ `payload_3class` 원본이 없는 기기에서 `preprocess.py` 를 인자 없이 돌리면 docs/03 의 기존 트랙 기록이 "[건너뜀]"으로 지워진다 → `process_track` 직접 호출.
- RGB(교수 요구): `build_image_dataset.py --channels rgb [--rgb-encoders raw_byte,char_class,local_entropy]`.
  채널 인코더는 `src/imaging/channel_encoders.py` 레지스트리(교체 가능 — G/B ablation용). npz 는 `_rgb` 접미사, CNN in_channels 자동 추론.
- 흐름 트랙 `ustc_flow_binary`(RQ4b, 악성/정상 이진): `python src/data/download_ustc.py`(USTC-TFC2016, dpkt+py7zr 필요)
  → `python src/imaging/build_flow_dataset.py [--channels rgb]`. flow_to_image 가 세션→앞2304B→48x48, IP/MAC 무력화.
- ⚠️ `preprocess.py <단일트랙>` 은 `docs/03_preprocessing_notes.md` 를 그 트랙만으로 덮어씀 → 노트 유지하려면 인자 없이 전체 실행.
- ⚠️ 새 트랙 추가 시 `--track` choices 를 5곳에서 함께 갱신: preprocess.py(TRACKS/REQUIRED_FILES), build_image_dataset.py, baseline_tfidf.py, diagnose_payload_bias.py, train.py.
  + CV·캐스케이드에 쓸 트랙이면 `cross_validate.py`·`cascade.py` choices 도(둘 다 트랙 목록을 따로 들고 있다 — `srbh_4class` 는 7곳 등록).
- ⚠️ 비-CSV 트랙(예: `ustc_flow_binary`)은 preprocess/build_image_dataset(CSV 경로)를 안 거침 → `--track` choices 를 train.py 한 곳만 갱신(위 "5곳"은 CSV 트랙 한정).
- ⚠️ 새 `data/raw/<dataset>/` 는 자동 무시 안 됨 → `.gitignore` 에 수동 추가(대용량 커밋 사고 방지). 재현은 다운로드 스크립트로 보장.
- ⚠️ `.gitignore` 는 **인라인 주석 불가**(줄 전체가 패턴). `experiments/checkpoints/  # 주석` 형태라 규칙이 무효였고 `.pt` 가 노출돼 있었음(2026-07 수정). 규칙 추가 후 `git check-ignore -v <경로>` 로 반드시 확인.
- ⚠️ RGB npz 파일명 약어 `_ENCODER_ABBR` 는 build_image_dataset.py 와 data_image.py 두 곳에 중복 → 인코더 추가 시 동시 갱신(저장/로드 파일명 일치).

## 모델 / 학습
- 이미지 모델(같은 npz 입력): `cnn`(93,988p) · `cnn_ee`(조기종료, 백본 동일 + 보조 헤드 2개) · `vit`(단일 ViT, 2.7M) · `hybrid`(CNN stem+Transformer, 0.58M) — `src/models/cnn.py`, `vit.py`.
  텍스트 모델: `charcnn` · `charcnn_tiny`(DL-B, 임베딩 32·k3·필터 64) · `charcnn_ee`(DL-C, k3 분기 보조 헤드 + 가중 0.3 결합 손실) · `charcnn_k3`(E9 대조, 임베딩 64·k3·필터 128 단독) · `bilstm`(`text_models.py`) · `distilbert`(E10, HF `distilbert-base-uncased`, 토큰 512, CV 전용), 별도 sklearn 베이스라인 `baseline_tfidf.py`.
  ⚠️ 새 텍스트 모델은 `train.TEXT_MODELS`·`cross_validate.TEXT_NN_MODELS` 2곳, 다중 헤드면 `train.MULTI_HEAD_MODELS` 도. `charcnn_ee --save-probs` 는 보조 헤드 확률을 `cvprobs_{tag}_aux.npz` 로 따로 남긴다.
- ⚠️ 새 이미지 모델 추가 시 `IMAGE_MODELS` 를 **2곳** 갱신: `train.py`, `cross_validate.py`(트랙의 "5곳"과는 별개).
  1차 재설계 후보(E2, `cnn.build_e2_variant`): `cnn_crop8`(위 8행=384바이트) · `cnn_gmp`(전역 최대 풀링) · `cnn_1d`(3×2,304 1차원 conv).
  ⚠️ `cnn_1d` 가 E2 판정에서 채택됐다(docs/14 §8.25). 단 3기의 다른 실험(G1~G6·T 계열)은 전부 현행 `cnn` 1차 위에서 잰 값이다 — 1차를 바꿔 인용하면 수치가 섞인다.
- CV 축: `cross_validate.py --seed N`(tag `_sN`, 42 는 생략) · `--group-template`(tag `_gtpl`, 숫자·hex·긴 토큰을 가린 템플릿으로 묶어 fold 분할 — T5). 사전 판정 규칙은 `docs/paper_revision/03_prereg_v2.md`.
- ⚠️ SR-BH 필드 조합은 `srbh_track.FIELD_COMBINATIONS`(CV)와 `measure_srbh_fields.COMBINATIONS`(E2 단계1) **2곳** — `tests/test_srbh_track.py` 가 일치를 검사.
- ⚠️ CPU 지연 측정 중 같은 기기에서 pytest·학습을 돌리지 말 것. 실행 간 절대값이 4~6% 표류하므로 후보끼리는 **한 실행 안**(번갈아 측정)의 값만 비교.
- CV 판정 도구: `gate_confirm.py --seed N [--group-template]`(C-G2 규칙, 보정 t 는 `gate_confirm.corrected_t`) · `linear_stage1.py --fit|--latency|--analyze`(선형 1차 E3 + E2 변형 경계 + T11) · `literature_compare.py [--dl-only]`(유사 연구 비교 그림, `--dl-only` 는 고전 ML 을 뺀 판을 `_dl_only` 파일로 따로 저장) · `latency_fair.py --stage1-variants`(1차 후보 지연, 반복마다 번갈아 측정).
  ⚠️ `cv_compare.py` 의 t 검정은 **보정 없는 대응 t**(fold 간 학습 집합 중복 무시) → 사전 판정에는 `corrected_t`(Nadeau–Bengio) + Holm 을 쓴다. ⚠️ `cross_validate.py --balance`(tag `_bal`)는 fold 내부 train 만 언더샘플링한다(T11).
- 장시간 작업 로그는 `experiments/logs/`(gitignore).
- ⚠️ `train.py` 를 `--smoke` 없이 돌리면 **추적 대상 그림**(`docs/figures/models/cm_*.png`)을 덮어씀 → 로컬 코드 검증은 반드시 `--smoke`(저장 생략).
- 체크포인트가 필요한 스크립트(cascade 등)의 로컬 스모크: 스크래치에서 소량 학습한 임시 `.pt` 를 `experiments/checkpoints/` 에 만들고 검증 후 삭제.
- ⚠️ 산출물 tag 는 **실험을 가르는 하이퍼파라미터를 전부 반영**해야 함. 안 그러면 서로 덮어씀(커밋 5eede2f 사고: RGB 조합이 전부 `_rgb` 로 저장돼 상호 덮어쓰기).
  현행 규칙: 채널 `_rgb-rb-cc-bd` (`data_image._channel_suffix`) + ViT 패치 `_p1x48` + 학습률 `_lr0.0003` + 균형화 `_bal`.
  `_lr` 은 기본값(`train.DEFAULT_LR`=1e-3)이면 생략 → 기존 산출물과 파일명 호환. 규칙은 train.py·cross_validate.py **2곳** 동시 갱신.
- ⚠️ **기본 lr=1e-3 은 CNN 기준값** — 단일 ViT 는 이 값에서 발산함(val Macro-F1 이 에폭 간 0.18 폭락).
  lr 만 3e-4/1e-4 로 낮추면 MCC 가 최대 +23.9pp 이동(docs/08 §9.1 실측). ViT 계열 비교는 반드시 lr 을 맞춰서 할 것.
  → "같은 학습 루프를 썼으니 공정"은 성립하지 않음. 기본값이 한쪽 아키텍처에 맞춰져 있으면 나머지가 자동으로 불리해짐.
- 통계 검증은 `src/eval/cross_validate.py`(5-fold, train/val/test 를 풀로 합쳐 재분할). fold 배정이 (라벨, seed)에만 의존 → 설정 간 paired 비교 성립, `label_fingerprint` 로 정렬 검증.
  - **설계 선택(입력 필드·하이퍼파라미터 고르기)용 CV 는 `--exclude-test`**(train+val 만, tag `_sealed`). 기본 동작은 test 까지 합치므로 선택에 쓰면 이후 본 측정이 누수된다.
  - `srbh_4class` 는 `--fields F1|F2|F3|F3M|F4|UC`(tag `_fF3` 등, F2=기본 입력이라 접미사 없음)로 입력 필드 조합을 바꾼다. `F3M` = F3 에서 쿠키 값만 같은 길이의 `x` 로 지운 절제 입력(T0-a, docs/14 §8.2 — F3 와 유의차 없어 트랙 F2 유지).
  - `--save-probs` 는 fold 별 내부 val·fold-test softmax 를 풀 절대 인덱스와 함께 `cvprobs_{tag}.npz` 로 남긴다(게이트 비교 G1 용, `src/eval/gate_cv.py` 가 두 파일을 정렬 검증해 읽는다). 이미지는 npz 대신 메모리 변환(실행당 ~77초, npz 와 픽셀 동일 확인).
  - ⚠️ `srbh_4class --fields` CV 는 `srbh_normal_strata.py` 산출 층 CSV 가 없으면 학습 전에 멈춘다(층별 FPR·입력 충돌 수를 싣기 때문, 새 기기에서 먼저 실행).
  - ⚠️ `cv_compare.py` 의 Holm 은 패턴에 걸린 **전 쌍**에 보정한다 → 사전 고정 절차가 일부 쌍만 정했으면(E2: F1~F4 6쌍) 그 쌍으로 재계산해 판정.
  - ⚠️ `cv_compare.py` 는 기본으로 `cv_*.json` 전부를 모아 **라벨 지문이 다르면 멈춘다** → 트랙·봉인 여부가 다른 결과가 섞이면 `--pattern "cv_srbh_4class_*_sealed.json"` 으로 거르고, `--fig-name` 으로 추적 그림 `cv_ranking_{metric}.png` 덮어쓰기를 피할 것.
  ⚠️ 단일 split 은 실행 간 ±0.11pp 흔들림(cuDNN 비결정성) → 조합 우열 주장은 반드시 CV 로 판정.

## 실행 환경 (Windows)
- 한글 콘솔(cp949)에서 파이썬 비-ASCII 출력이 깨짐 → Bash 로 파이썬 실행 시 `PYTHONIOENCODING=utf-8` 프리픽스 사용(또는 스크립트가 stdout 재설정).
- matplotlib 그림 라벨/범례/제목은 ASCII 만(DejaVu Sans 에 Hangul 없음 → □ 로 깨짐+경고). 한글은 콘솔·JSON 에만.
- **GPU 는 데스크톱 한정**(RTX 4080 SUPER, 17GB / torch cu124, RAM 34GB) — 전면 학습은 거기서 실행.
  노트북은 Intel Arc iGPU + torch CPU 빌드라 학습 불가 → 착수 시 `torch.cuda.is_available()` 로 어느 머신인지 먼저 확인.
  ⚠️ GPU 는 1대뿐 → 학습 작업은 **순차 실행**(동시 실행 시 재현성 악화). 학습 중 다른 검증이 필요하면 `CUDA_VISIBLE_DEVICES=` 로 CPU 강제.
- ⚠️ 백그라운드 파이썬은 stdout 버퍼링으로 진행 로그가 안 보임 → 진행 추적은 산출 JSON 존재 여부로 하거나 `python -u` 사용.
- ⚠️ **백그라운드 작업이 "killed" 로 보고돼도 파이썬 자식 프로세스는 살아 있을 수 있음**(셸 래퍼만 종료됨).
  로그·산출물만 보고 "죽었다" 판단하면 재실행 시 GPU 동시 실행이 발생해 순차 실행 규칙이 깨짐(2026-07-19 실제 사고, docs/08 §9.6).
  → 재실행 전 반드시 생존 확인: `Get-CimInstance Win32_Process -Filter "Name='python.exe'"` 또는 `nvidia-smi` 로 GPU 점유 확인.
- 장시간 작업(CV 등)은 완료 감지를 `until [ -f <산출 JSON> ]; do sleep 20; done` 로 걸 것(래퍼 종료와 무관하게 동작).
- ⚠️ Bash 도구에서 `git show <ref>:<path>` 는 `:` 가 `;` 로 변환돼 실패 → `git diff <a> <b> -- <path>` 사용.
- `nvidia-smi` 는 PATH 에 없음 → GPU 확인은 `torch.cuda.is_available()` 또는 PowerShell `Get-CimInstance Win32_VideoController`.
- ⚠️ CPU 강제는 Bash 에서 `CUDA_VISIBLE_DEVICES=` 이지만 **PowerShell 에서 빈 값 대입은 변수 삭제**라 GPU 가 그대로 보인다 → PowerShell 은 `$env:CUDA_VISIBLE_DEVICES='-1'`.
- 클래스 불균형 트랙 학습: `train.py --balance`(train 만 언더샘플링, val/test 실분포 유지).
- 모든 모델 평가는 `src/eval/metrics.py` 로 일원화(계산 방식 차이 배제). 지표: Acc·Macro-F1·PR-AUC·ROC-AUC·TPR@FPR + attack_focused(benign-evasion/FPR).
  ⚠️ **MCC 는 3기에 교수 지시로 완전 제거**됐다(`metrics.py` 머리 주석, 근거·반대 근거 docs/13 §1.1). 기준 지표는 Macro-F1. MCC 를 요구하는 코드는 `KeyError` 로 멈춘다(2026-10-04 실제 발생). 아래 2기 서술의 MCC 수치는 기록으로만 읽을 것.

## 모델 용어 (2026-08-02 변경 — 혼동 사고 후 확정)
- **제안 모델 = 캐스케이드**(1차 RGB CNN + 2차 char-CNN). 논문이 주장하는 대상은 이것.
- ⚠️ 단일 CNN 은 제안 지위를 내려놓고 **지표 비교 기준**이 됐다. `제안 CNN` 이라는 이름은 **쓰지 말 것** — 그 말이 (ㄱ)제안 모델과 (ㄴ)캐스케이드 1차 부품 두 뜻으로 읽혀 실제로 혼동이 났다(docs/10 §0.1).
  - `RGB CNN` = 48×48×3 단일 CNN(기본 비교 기준) · `gray CNN` = 1채널 판본 · `얕은 CNN` = 채널 무관 아키텍처 지칭.
  - ⚠️ 과거 문서의 회피 수치 0.594/0.5944/0.0022 는 **gray 실측**이다. RGB 기준선은 **0.5981**(2026-08-02 측정).

## 캐스케이드 (Phase 10, docs/09 · Phase 11 확장 docs/10 §5.1)
- ⚠️ **`hybrid`(모델, docs/08 ViT 계열 = CNN stem+Transformer)와 `cascade`(배치 구조)는 다른 것** — 이름 섞지 말 것.
- `src/models/cascade.py`: 1차=RGB CNN(전량) → 확신도<τ 인 소수만 2차=char-CNN. **학습 없음**(기존 `_bal` 체크포인트 재사용). τ→0 은 CNN 단독, τ→1 은 char-CNN 단독으로 정확히 수렴.
- τ 는 **val 에서 확정 후 test 에 1회 적용**(test 로 고르면 누수). `--select match-teacher|budget`.
- 회피 실험 편입: `run_evasion.py --model cascade` — τ 는 산출 JSON 에서 읽음(공격 데이터로 재튜닝 금지).
- 방어 구성(Phase 11): `cascade.py --defense advtrain --mutation-split {S0,SA} --aug-ratio ρ` → **1·2차 양쪽** 방어본을 조립하고 τ 를 그 구성의 val 에서 **재선택**한다(확신도 분포가 달라 기존 τ 재사용 불가).
- 체크포인트/산출물 tag 는 `tagging.build_tag` 단일 진실 소스 + 캐스케이드 고유 축(2차 모델·τ 선택 규칙·방어) 추가. ⚠️ cascade.py 가 규칙 사본을 들고 있었으나 tagging.py 로 통합됨(2026-08-02).
- ⚠️ **"막아둔 조합"이 tag 버그를 가린다**: `run_evasion` 이 캐스케이드+채널을 `p.error` 로 막고 있던 동안, 저장 tag 의 채널 축이 `model=="cnn"` 조건에 묶여 있는 걸 아무도 못 봤다. 차단을 푼 즉시 RGB 결과가 gray 결과를 덮어씀(2026-08-02, docs/10 §5.1). → **축은 그 조합을 쓸 수 있게 되기 전에 tag 에 넣어둘 것.** 차단을 풀 때는 저장 경로의 tag 부터 점검.
- 조기종료 1차(Phase 12/M4): `cascade.py --stage1 cnn_ee`. 종료 임계값은 **val 에서 선택 → test 1회 적용**(τ 와 같은 원칙, 순서는 임계값 → τ).
  ⚠️ 조기종료는 **마스킹이 아니라 축소 배치**로 구현돼 있다(그래야 지연이 실제로 준다). 근거는 eval 모드 BatchNorm 의 running stats — 학습 모드에서 이 경로를 타게 만들면 배치 구성에 따라 결과가 흔들린다(`tests/test_early_exit.py` 가 고정).
  ⚠️ 조기종료 모델의 지연은 **배치 구성이 곧 비용** → `measure_latency(sample_idx=...)` 로 무작위 대표 배치를 쓴다(앞 N개만 재면 클래스 편향).
- ⚠️ 캐스케이드가 줄이는 건 **추론 지연**뿐. 두 모델을 다 학습하므로 **학습 비용은 합산** — 근거는 처리량(docs/04 §5: 81,964 vs 6,010 /s)이지 학습 지표(s/epoch)가 아님.
- ⚠️ 캐스케이드 고유 리스크: 회피 변형이 1차 확신도를 낮추면 에스컬레이션↑ → **정확도 그대로인데 지연만 폭증**(비용 기반 공격). `run_evasion` 이 조건별 `escalation_rate` 를 함께 기록.
