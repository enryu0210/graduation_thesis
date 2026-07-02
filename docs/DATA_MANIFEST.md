# Data Manifest

> 이 문서는 졸업논문 재현성·심사 대응의 근거 자료입니다.
> 각 데이터셋의 출처·라이선스·접근일·레코드 수·체크섬을 기록합니다.
> 하단의 "자동 검증 결과" 섹션은 `python src/data/validate.py` 실행 시 자동으로 갱신됩니다.

접근일 기준: 2026-07-02

---

## SQLi / XSS / Command Injection 페이로드셋 (확보 완료 — 4-class 채택)

> **결정**: 본 논문은 **4-class 데이터셋**(`SQLInjection_XSS_CommandInjection_MixDataset.1.0.0.csv`,
> Normal/SQLi/XSS/CmdI)을 페이로드 학습용 주 데이터셋으로 **확정**한다.
> 3-class 버전(`SQLInjection_XSS_MixDataset.1.0.0.csv`, CmdI 제외)은 참고용으로만 보관하며
> 학습/평가에는 사용하지 않는다.

- Source URL: (Kaggle — 정확한 slug 사용자 확인 필요, 아래 Notes 참고)
- Author/Org: (확인 필요)
- Accessed: 2026-07-02
- License: (확인 필요 — Kaggle 데이터셋 페이지에서 확인 후 기입)
- Commit/Version: 파일명 기준 `1.0.0`
- Local path (주 데이터셋):
  - `data/raw/payload_3class/SQLInjection_XSS_CommandInjection_MixDataset.1.0.0.csv` (✅ **채택** — 4-class: Normal/SQLi/XSS/CmdI)
  - `data/raw/payload_3class/SQLInjection_XSS_MixDataset.1.0.0.csv` (참고용 보관 — 3-class: Normal/SQLi/XSS)
- Row count (documented): 계획서 예상 ~49,998 건
- Row count (actual): **206,636 건** (아래 자동 검증 결과 참고 — 예상치와 크게 다름)
- Class distribution (4-class): SQLi 57,316 / XSS 40,799 / CmdI 50,000 / Normal 58,521 (아래 자동 검증 결과 참고)
- Duplicate rate (raw): 3.31% (아래 자동 검증 결과 참고)
- Notes:
  - **⚠️ 중요 불일치**: 계획서(01_data_acquisition_plan.md 2.1)는 약 49,998건을 예상했으나,
    실제 확보된 파일은 206,636건으로 훨씬 큼. 계획서가 지목한 데이터셋과 **다른(더 큰) 버전**일 가능성이 높음.
    논문 4장(데이터셋)에 실제 사용한 파일의 정확한 출처/규모를 반드시 명시할 것.
  - 컬럼 구조가 원-핫 멀티라벨(`Sentence, SQLInjection, XSS, CommandInjection, Normal`)이나,
    자동 검증 결과 **다중라벨/무라벨 행이 0** 이므로 사실상 단일라벨(4-class) 데이터로 안전하게 사용 가능.
  - 정확한 Kaggle URL/작성자/라이선스는 아직 미확정 → **사용자 확인 필요 항목**.

---

## CSIC 2010 HTTP dataset (확보 완료 — 사용자 수동 다운로드)

- Source URL: 사용자가 직접 내려받아 제공 (파일명 `csic_database.csv`).
  컬럼 구조(`Method, User-Agent, ..., classification, URL` + 선두 `Unnamed: 0` = Normal/Anomalous)가
  `github.com/msudol/Web-Application-Attack-Datasets` 의 정제 CSV 버전과 일치함 → **정확한 원본 출처는 사용자 확인 필요**.
- 후보 미러(재현용): `kaggle.com/datasets/ispangler/csic-2010-web-application-attacks`,
  `gitlab.fing.edu.uy/gsi/web-application-attacks-datasets`
- Accessed: 2026-07-02
- License: (확인 필요 — CSIC 2010 원본은 연구·교육용 공개 데이터셋)
- Commit/Version: 파일 mtime 기준 2019-10-11 스냅샷
- Local path: `data/raw/csic2010/csic_database.csv`
- SHA256: `c420f0bc0464376de75b6c419a0ac226fe69fe12c8ac4908843273721e44e637`
- Row count (documented): 정상 36,000 + 이상 25,000+ 건
- Row count (actual): **61,065 건 (정상 36,000 + 이상 25,065)**
- Class distribution: `classification` 컬럼 — 0(Normal)=36,000(58.95%), 1(Anomalous)=25,065(41.05%)
- Notes:
  - ✅ **공식 통계와 교차검증 일치**: 정상 36,000건은 정확히 일치, 이상 25,065건은 "25,000+" 범위에 부합.
  - 라벨 컬럼이 2개(`Unnamed: 0`=문자열 Normal/Anomalous, `classification`=0/1)로 동일 정보를 이중 표현함.
    학습 시에는 `classification`(0/1)을 정답 라벨로 사용하면 됨.
  - **원본 보존 원칙**에 따라 `data/raw/csic2010/` 아래 그대로 두고 가공하지 않음 (해당 폴더는 .gitignore 대상 → git 미추적, 재현은 SHA256으로 보장).
  - GitLab/Kaggle 미러와의 추가 행수 교차검증은 선택 사항(원본 CSIC 은 v01/v02 두 버전 존재).

---

## 페이로드 보강 소스 (미확보 — 다운로드 스크립트 준비됨)

| 소스 | URL | Local path | Commit hash |
|---|---|---|---|
| PayloadsAllTheThings | github.com/swisskyrepo/PayloadsAllTheThings | `data/raw/payload_supplement/PayloadsAllTheThings` | (미완료) |
| payloadbox/sql-injection-payload-list | github.com/payloadbox/sql-injection-payload-list | `data/raw/payload_supplement/sql-injection-payload-list` | (미완료) |
| payloadbox/command-injection-payload-list | github.com/payloadbox/command-injection-payload-list | `data/raw/payload_supplement/command-injection-payload-list` | (미완료) |

- `src/data/download_supplements.py` 실행 시 clone 되며 커밋 해시가 출력됨 → 위 표에 기록할 것.
- OWASP XSS Filter Evasion Cheat Sheet: 페이지 원문 스크래핑 저장 예정 (접근일 기록 필수).

---

# 자동 검증 결과 (validate.py)

## csic_database.csv

- Local path: `data\raw\csic2010\csic_database.csv`
- File size: 28.2 MB
- SHA256: `c420f0bc0464376de75b6c419a0ac226fe69fe12c8ac4908843273721e44e637`
- UTF-8 decode error (head sample): 0.0
- Row count (actual): 61,065
- Columns: ['Unnamed: 0', 'Method', 'User-Agent', 'Pragma', 'Cache-Control', 'Accept', 'Accept-encoding', 'Accept-charset', 'language', 'host', 'cookie', 'content-type', 'connection', 'lenght', 'content', 'classification', 'URL']
- Duplicate rate (raw, 'URL' 컬럼 기준): 0.7790 (47,567 rows)
- Missing/empty in 'URL': NaN=0, empty-string=0
- Class distribution ('classification' 컬럼 값 분포):
    - 0: 36,000 (58.95%)
    - 1: 25,065 (41.05%)

## SQLInjection_XSS_CommandInjection_MixDataset.1.0.0.csv

- Local path: `data\raw\payload_3class\SQLInjection_XSS_CommandInjection_MixDataset.1.0.0.csv`
- File size: 93.5 MB
- SHA256: `e221ea9e118cdac6d7dabf085a224f42b6cde800a71c446f71c2dec609a2c233`
- UTF-8 decode error (head sample): 0.0
- Row count (actual): 206,636
- Columns: ['Sentence', 'SQLInjection', 'XSS', 'CommandInjection', 'Normal']
- Duplicate rate (raw, 'Sentence' 컬럼 기준): 0.0331 (6,839 rows)
- Missing/empty in 'Sentence': NaN=0, empty-string=616
- Class distribution (원-핫 라벨 합계):
    - SQLInjection: 57,316 (27.74%)
    - XSS: 40,799 (19.74%)
    - CommandInjection: 50,000 (24.20%)
    - Normal: 58,521 (28.32%)
    - 라벨 무결성: 다중라벨 행=0, 무라벨 행=0 (0이면 깔끔한 단일라벨 데이터)

## SQLInjection_XSS_MixDataset.1.0.0.csv

- Local path: `data\raw\payload_3class\SQLInjection_XSS_MixDataset.1.0.0.csv`
- File size: 55.2 MB
- SHA256: `81fac1957910970264705585e0f8816a0e9b7c31e8c6eceef876cca31c7accc7`
- UTF-8 decode error (head sample): 0.0
- Row count (actual): 156,636
- Columns: ['Sentence', 'SQLInjection', 'XSS', 'Normal']
- Duplicate rate (raw, 'Sentence' 컬럼 기준): 0.0310 (4,853 rows)
- Missing/empty in 'Sentence': NaN=0, empty-string=616
- Class distribution (원-핫 라벨 합계):
    - SQLInjection: 57,316 (36.59%)
    - XSS: 40,799 (26.05%)
    - Normal: 58,521 (37.36%)
    - 라벨 무결성: 다중라벨 행=0, 무라벨 행=0 (0이면 깔끔한 단일라벨 데이터)

