# Phase 1 — 데이터셋 확보 및 검증 계획

> `web_attack_image_cnn_thesis_design.md`의 3번(데이터셋 설계) 항목을 실행 단계로 구체화한 문서입니다.
> 이 문서는 VS Code의 Claude Code 터미널에 전달해 실제 다운로드/검증 스크립트를 작성·실행시키기 위한 **지시서(spec)** 이며, 코드 자체는 포함하지 않습니다.

---

## 0. 이 단계의 목표

- 4개 소스 그룹의 데이터를 **원본 그대로(raw, immutable)** 확보
- 모든 데이터셋에 대해 **출처·라이선스·접근일·레코드 수·체크섬**을 `DATA_MANIFEST.md`에 기록 → 논문 재현성·심사 대응의 근거
- 다운로드 직후 **1차 검증**(행 수, 컬럼 스키마, 중복률, 클래스 분포)까지 완료한 상태로 Phase 2(EDA)에 넘김

**중요한 원칙**: 링크가 살아있다고 바로 신뢰하지 않습니다. 아래 각 소스마다 "검증 상태"를 표시했고, 미확정 소스는 Claude Code가 직접 재검색해서 사용자 확인을 받은 뒤 확정하도록 설계했습니다.

---

## 1. 프로젝트 디렉토리 스켈레톤 (전체 구조 미리보기)

```
web-attack-image-cnn/
├── data/
│   ├── raw/                          # 원본, 손대지 않은 다운로드 사본 (덮어쓰기 금지)
│   │   ├── payload_3class/           # SQLi/XSS/CmdI 다중클래스 페이로드셋
│   │   ├── csic2010/
│   │   └── payload_supplement/       # PayloadsAllTheThings, OWASP, payloadbox
│   ├── interim/                      # 디코딩·정규화 등 중간 산출물
│   ├── processed/                    # 정제·분할 완료된 최종 테이블 (train/val/test)
│   └── images/                       # (Phase 3) 바이트→이미지 변환 결과
├── docs/
│   ├── DATA_MANIFEST.md              # 데이터 출처/라이선스/체크섬/접근일 기록 (핵심 산출물)
│   ├── EDA_notes.md                  # (Phase 2)
│   └── thesis_design.md              # 업로드한 원본 설계 문서 사본
├── notebooks/                        # EDA·프로토타이핑
├── src/
│   ├── data/                         # 다운로드·검증·정규화·분할 스크립트
│   ├── imaging/                      # (Phase 3) 페이로드→이미지 변환
│   ├── models/                       # (Phase 4) CNN/베이스라인
│   ├── attacks/                      # (Phase 5) RQ2 회피 공격
│   ├── defense/                      # (Phase 6) RQ3 adversarial training
│   └── eval/                         # 지표 계산·유의성 검정
├── experiments/
│   ├── configs/
│   └── results/
├── tests/                            # 변환 함수·분할 무결성 단위 테스트
├── requirements.txt
└── README.md
```

이번 단계에서는 `data/raw/`, `docs/DATA_MANIFEST.md`, `src/data/`만 채웁니다. 나머지는 뼈대만 생성.

---

## 2. 데이터 소스 상세 및 검증 상태

### 2.1 공격 페이로드 3-class (SQLi / XSS / Command Injection)

| 항목 | 내용 |
|---|---|
| 목표 데이터셋 | "SQLi-XSS-CommandInjection" (Kaggle, 약 5만 건, 3-class 라벨링) |
| 검증 상태 | ⚠️ **부분 검증**. 데이터셋 이름·규모(49,998 records)는 2026년 6월 발표 논문("Detection of SQL Injection, XSS, and Command Injection Attacks in Web Payloads Using SVM, Random Forest, and XGBoost", *Journal of Information Systems and Informatics*, DOI: 10.63158/journalisi.v8i3.1655)에서 그대로 인용된 걸 확인함. 다만 **정확한 Kaggle URL(slug)은 웹 검색으로 확정하지 못함.** |
| 해결 절차 | 아래 "3.1 확정 절차" 참고 — Claude Code가 Kaggle API로 직접 검색·확정, 실패 시 대안 조합 사용 |

**대안(fallback) — 검증된 개별 데이터셋 조합**: 위 슬러그를 못 찾을 경우, 아래처럼 클래스별로 검증된 소스를 합쳐 동등한 3-class 데이터셋을 직접 구성합니다.

| 클래스 | 후보 소스 | 검증 상태 |
|---|---|---|
| SQLi | `kaggle.com/datasets/syedsaqlainhussain/sql-injection-dataset` (33,726건) | ✅ 여러 논문에서 인용 확인 |
| SQLi (대안) | `kaggle.com/datasets/sajid576/sql-injection-dataset` (30,919건) | ✅ 논문 인용 확인 |
| XSS | `kaggle.com/datasets/syedsaqlainhussain/cross-site-scripting-xss-dataset-for-deep-learning` | ✅ 서베이 논문(XSS 22년치 매핑 연구)에서 인용 확인 |
| Command Injection | `github.com/payloadbox/command-injection-payload-list` (페이로드 리스트, 라벨 없음 → benign 대조군 직접 구성 필요) | ✅ 여러 논문 레퍼런스에서 확인 |

> 대안 경로를 쓰게 되면 클래스 간 수집 방식이 달라 도메인 편차(Jensen-Shannon Divergence 등)가 생길 수 있습니다 — 이 경우 논문 4장(데이터셋)에서 반드시 명시하고, 가능하면 클래스 간 JSD를 계산해 보고할 것.

### 2.2 웹 트래픽 (HTTP 요청) — CSIC 2010 HTTP dataset

| 소스 | 유형 | 검증 상태 |
|---|---|---|
| `http://www.isi.csic.es/dataset/` | 공식 원본 | ⚠️ 커뮤니티에서 접속 불안정 리포트 다수 (OWASP CRS 이슈트래커 등) |
| `kaggle.com/datasets/ispangler/csic-2010-web-application-attacks` | Kaggle 미러 | ✅ |
| `gitlab.fing.edu.uy/gsi/web-application-attacks-datasets` | 대학 미러 (CSIC 이슈트래커에 백업 링크로 명시됨) | ✅ |
| `github.com/msudol/Web-Application-Attack-Datasets` | 정제된 CSV 버전 | ✅ |

공식 통계: 정상 요청 36,000건 + 이상 요청 25,000건 이상, SQLi/버퍼오버플로우/CRLF injection/XSS/parameter tampering 등 포함. **다운로드 후 이 수치와 실제 레코드 수를 대조해 검증**할 것.

### 2.3 페이로드 다양성 보강용 소스

| 소스 | URL | 용도 |
|---|---|---|
| PayloadsAllTheThings | `github.com/swisskyrepo/PayloadsAllTheThings` | SQLi/XSS/CmdI 등 실전 페이로드 패턴 보강 (MIT 라이선스) |
| OWASP XSS Filter Evasion Cheat Sheet | `owasp.org/www-community/xss-filter-evasion-cheatsheet` | XSS 회피 패턴 사전 (RQ2 회피 공격 설계에도 재사용 가능) |
| payloadbox/sql-injection-payload-list | `github.com/payloadbox/sql-injection-payload-list` | SQLi 패턴 보강 |
| payloadbox/command-injection-payload-list | `github.com/payloadbox/command-injection-payload-list` | CmdI 패턴 보강 (라벨 데이터 부족 시 핵심 소스) |

---

## 3. 다운로드 절차

### 3.1 "SQLi-XSS-CommandInjection" 데이터셋 확정 절차 (중요)

이 소스는 정확한 링크가 불확실하므로, Claude Code가 **추측으로 다운로드하지 말고** 아래 순서를 따르게 합니다:

1. Kaggle API(`kagglehub` 또는 `kaggle datasets list -s`)로 `"SQLi XSS Command Injection"` 키워드 검색
2. 검색 결과 중 레코드 수가 약 49,998건(±수백 건 오차 허용)이고 3개 라벨(SQLi/XSS/CmdI)을 가진 데이터셋이 있는지 확인
3. **후보가 여러 개거나 하나도 없으면 다운로드를 중단하고, 후보 목록(제목/작성자/URL/건수)을 사용자에게 보고 → 사용자 확인 후 확정**
4. 확정되면 `docs/DATA_MANIFEST.md`에 정확한 URL·작성자·라이선스를 기록

이 절차가 실패하면 2.1의 "대안 조합"으로 전환하고, 그 사실을 매니페스트와 논문 4장 초안 메모에 남깁니다.

### 3.2 CSIC 2010

1. Kaggle 미러(`ispangler/csic-2010-web-application-attacks`)를 1차 소스로 다운로드
2. GitLab 미러에서 동일 데이터를 받아 **행 수·정상/이상 비율이 일치하는지 교차검증**
3. 불일치 시 원인을 매니페스트에 기록 (버전 차이 가능성 있음 — 원 논문은 v01/v02 두 버전 존재)

### 3.3 보강 소스

- PayloadsAllTheThings, payloadbox 리스트: `git clone` (버전 고정을 위해 특정 커밋 해시를 매니페스트에 기록)
- OWASP 치트시트: 페이지 텍스트를 스크래핑해 원문 그대로 저장 (사이트 개편 대비 접근일 기록 필수)

---

## 4. 데이터 검증 체크리스트 (졸업논문 재현성 확보용)

각 데이터셋마다 다음을 **다운로드 직후** 수행하고 결과를 `docs/DATA_MANIFEST.md`에 남깁니다.

- [ ] 원본 URL, 작성자/기관, 접근일(YYYY-MM-DD), 라이선스 명시
- [ ] SHA256 체크섬 기록 (재현성·무결성 확인용)
- [ ] 실제 행 수 vs. 문서화된 공식 수치 대조 (일치 여부와 오차율 기록)
- [ ] 컬럼 스키마 확인 (payload/query 컬럼명, 라벨 컬럼명, 인코딩)
- [ ] 완전 중복 행 비율 계산 (제거 전 수치 — 3장 전처리 체크리스트의 "제거 전/후 비교"에 사용)
- [ ] 클래스별 샘플 수 및 불균형 비율
- [ ] 결측치/빈 문자열 비율
- [ ] (텍스트 인코딩) UTF-8 디코딩 실패율 — 바이트 변환 단계(Phase 3)에서 중요

---

## 5. `DATA_MANIFEST.md` 템플릿

Claude Code가 이 템플릿 형식으로 실제 매니페스트를 채우게 합니다.

```markdown
# Data Manifest

## <데이터셋 이름>
- Source URL:
- Author/Org:
- Accessed:
- License:
- Commit/Version (해당 시):
- Local path: data/raw/...
- SHA256:
- Row count (documented): 
- Row count (actual):
- Class distribution:
- Duplicate rate (raw):
- Notes:
```

---

## 6. Claude Code에 전달할 작업 지시문 (그대로 붙여넣기용)

```
web-attack-image-cnn 졸업논문 프로젝트의 Phase 1(데이터셋 확보·검증)을 진행해줘.

작업 범위:
1. 아래 디렉토리 스켈레톤을 생성 (data/raw, data/interim, data/processed, data/images,
   docs, notebooks, src/{data,imaging,models,attacks,defense,eval}, experiments/{configs,results}, tests)
2. src/data/ 아래에 소스별 다운로드 스크립트를 작성:
   - "SQLi-XSS-CommandInjection" 3-class 페이로드셋: Kaggle API로 키워드 검색 →
     레코드 수(~49,998건)와 라벨 3종이 일치하는 후보를 찾으면 다운로드,
     후보가 모호하면 절대 임의로 선택하지 말고 후보 목록을 출력하고 멈출 것.
   - CSIC 2010: Kaggle 미러(ispangler/csic-2010-web-application-attacks)와
     GitLab 미러(gitlab.fing.edu.uy/gsi/web-application-attacks-datasets)에서
     각각 받아 행 수를 교차검증.
   - PayloadsAllTheThings, payloadbox/sql-injection-payload-list,
     payloadbox/command-injection-payload-list: git clone (커밋 해시 기록).
   - OWASP XSS Filter Evasion Cheat Sheet: 페이지 원문 저장.
3. 모든 다운로드는 data/raw/ 아래 원본 그대로 보존 (가공 금지).
4. 각 데이터셋에 대해 SHA256 체크섬, 접근일, 라이선스, 실제 행 수,
   문서화된 공식 수치와의 대조 결과를 docs/DATA_MANIFEST.md에 기록.
5. 4번 체크리스트(중복률, 클래스 분포, 결측치, 컬럼 스키마)를 계산하는
   검증 스크립트를 src/data/validate.py로 작성하고 결과를 docs/DATA_MANIFEST.md에 append.
6. 마지막에 무엇이 확정됐고 무엇이 사용자 확인이 필요한지(특히 3-class 데이터셋 후보 선택)
   요약해서 보고해줘.
```

---

## 7. 다음 단계

- **Phase 2 (EDA)**: 페이로드 길이 분포 확인 → 이미지 변환 시 고정 폭 `W` 결정 근거 마련 (원 설계 문서 4장 Step 2)
- 확정이 필요한 3-class 데이터셋 후보가 나오면, 그 결과를 알려주시면 다음 마크다운 문서(Phase 2 EDA 계획)에 반영하겠습니다.
