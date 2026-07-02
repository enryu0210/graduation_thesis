# Data Manifest

> 이 문서는 졸업논문 재현성·심사 대응의 근거 자료입니다.
> 각 데이터셋의 출처·라이선스·접근일·레코드 수·체크섬을 기록합니다.
> 하단의 "자동 검증 결과" 섹션은 `python src/data/validate.py` 실행 시 자동으로 갱신됩니다.

접근일 기준: 2026-07-02

---

## SQLi / XSS / Command Injection 3-class 페이로드셋 (확보 완료)

- Source URL: (Kaggle — 정확한 slug 사용자 확인 필요, 아래 Notes 참고)
- Author/Org: (확인 필요)
- Accessed: 2026-07-02
- License: (확인 필요 — Kaggle 데이터셋 페이지에서 확인 후 기입)
- Commit/Version: 파일명 기준 `1.0.0`
- Local path:
  - `data/raw/payload_3class/SQLInjection_XSS_CommandInjection_MixDataset.1.0.0.csv` (4-class: Normal/SQLi/XSS/CmdI)
  - `data/raw/payload_3class/SQLInjection_XSS_MixDataset.1.0.0.csv` (3-class: Normal/SQLi/XSS, CmdI 제외 버전)
- Row count (documented): 계획서 예상 ~49,998 건
- Row count (actual): **아래 자동 검증 결과 참고 (예상치와 크게 다름 — 확인 필요)**
- Class distribution: 아래 자동 검증 결과 참고
- Duplicate rate (raw): 아래 자동 검증 결과 참고
- Notes:
  - **⚠️ 중요 불일치**: 계획서(01_data_acquisition_plan.md 2.1)는 약 49,998건을 예상했으나,
    실제 확보된 파일은 그보다 훨씬 큼. 계획서가 지목한 데이터셋과 **다른(더 큰) 버전**일 가능성이 높음.
    논문 4장(데이터셋)에 실제 사용한 파일의 정확한 출처/규모를 반드시 명시할 것.
  - 컬럼 구조가 원-핫 멀티라벨(`Sentence, SQLInjection, XSS, CommandInjection, Normal`)임.
    라벨 무결성(한 행에 라벨 1개인지)은 자동 검증 결과에서 확인.
  - 정확한 Kaggle URL/작성자/라이선스는 아직 미확정 → **사용자 확인 필요 항목**.

---

## CSIC 2010 HTTP dataset (미확보 — 다운로드 스크립트 준비됨)

- Source URL: (Kaggle 미러) `kaggle.com/datasets/ispangler/csic-2010-web-application-attacks`
- 교차검증 미러: `gitlab.fing.edu.uy/gsi/web-application-attacks-datasets`
- Accessed: (미완료)
- License: (확인 필요)
- Local path: `data/raw/csic2010/` (예정)
- Row count (documented): 정상 36,000 + 이상 25,000+ 건
- Row count (actual): (미완료)
- Notes:
  - `src/data/download_csic2010.py` 로 다운로드 가능하나 **Kaggle 인증(kaggle.json) 필요**.
  - 인증 설정 후 실행 → GitLab 미러와 행 수 교차검증 → 이 표 갱신 예정.

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

