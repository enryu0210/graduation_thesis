# Phase 3 — 전처리 및 이미지 변환 설계/결정 문서

> 이 문서는 사람이 작성한 **설계·의사결정 기록**입니다.
> 실행 통계(중복률·분할 크기 등)는 `python src/data/preprocess.py` 가 자동 생성하는
> `docs/03_preprocessing_notes.md` 를, 이미지 크기 근거는 `docs/EDA_notes.md` 를 참고하세요.

---

## 1. 데이터 트랙 분리 결정

라벨 공간과 데이터 granularity 가 달라 **두 트랙을 독립적으로 전처리**한다.

| 트랙 | 데이터 | 라벨 | 용도 |
|---|---|---|---|
| `payload_4class` | 공격 페이로드셋 (199,793건, 중복 제거 후) | Normal/SQLi/XSS/CmdI (4-class) | **RQ1 주 데이터셋** |
| `csic_binary` | CSIC 2010 HTTP 트래픽 | Normal/Anomalous (2-class) | 실제 트래픽·일반화 트랙 |

- 두 데이터셋을 **하나로 합치지 않은 이유**: 페이로드셋은 짧은 페이로드 문자열, CSIC 은 전체 HTTP 요청으로
  텍스트 길이·구조가 다르다(EDA: 페이로드 p50=315B vs CSIC URL p50=59B).
  섣불리 합치면 도메인 편차가 라벨과 얽혀 모델이 "데이터 출처"를 학습할 위험이 있다.
- 결합/전이(transfer) 여부는 Phase 4(모델) 실험 설계에서 별도로 결정한다. 두 트랙 산출물은 모두 준비돼 있다.

---

## 2. 전처리 결정 (`src/data/preprocess.py`)

1. **디코딩 ablation 대비 — raw/decoded 동시 보존**
   설계 3장은 "디코딩 적용 vs 미적용" 비교를 요구한다. 따라서 원본 `text_raw` 와
   URL·HTML 디코딩을 적용한 `text_decoded` 를 **두 컬럼 모두** 저장한다.
   - URL 디코딩은 이중 인코딩(`%2527`)까지 풀도록 변화가 없을 때까지 최대 3회 반복.
   - 이후 HTML 엔티티(`&lt;`) 디코딩.

2. **중복 제거를 '분할 전'에 수행 — 누수(leakage) 방지**
   동일 페이로드가 train 과 test 에 동시에 들어가면 성능이 부풀려진다.
   따라서 `text_raw` 기준 완전 중복을 **분할 전에** 제거한다.

3. **Stratified 70/15/15 분할, seed=42**
   클래스 비율을 세 split 에서 동일하게 유지(sklearn `train_test_split` 2단계).

산출물: `data/processed/{track}_{train,val,test}.csv` (컬럼 `text_raw, text_decoded, label`).
`data/processed/` 는 `.gitignore` 대상 → git 미추적, 재현은 스크립트+시드로 보장.

### ⚠️ 리스크 — CSIC 의 URL 중복 문제 (Phase 4 결정 필요)

- CSIC 은 **URL 기준 중복률이 77.9%** 로 매우 높다(같은 URL 요청을 반복 수집).
  URL 만으로 중복 제거하면 61,065건 → **13,498건**으로 급감하고,
  클래스 균형도 뒤집힌다(Normal 이 더 많이 중복되어 제거됨 → Anomalous 다수).
- 함의: CSIC 을 이미지화할 때 **URL 만 쓰는 것은 부적절**할 수 있다.
  전체 요청(메서드+헤더+본문) 직렬화를 이미지화 대상으로 삼으면 중복이 줄고 정보량도 는다.
  → Phase 4 에서 "CSIC 이미지화 단위(URL vs 전체 요청)"와 "중복 제거 기준"을 확정할 것.
- 현재 파이프라인은 **URL 기준**으로 동작하며(가장 단순한 baseline), 위 결정 시 손쉽게 교체 가능하도록 모듈화돼 있다.

---

## 3. 이미지 변환 결정 (`src/imaging/`)

- **핵심 함수** `payload_to_image(text, side=48)` (`payload_to_image.py`)
  - UTF-8 바이트 → 앞에서부터 `side*side` 바이트 사용(길면 truncation) → 부족하면 zero-padding → `(side, side)` 리셰이프.
  - 픽셀 = 바이트 값(0~255) uint8 로 **손실 없이 저장**, 모델 입력 직전에 `normalize_01` 로 0~1 스케일링.
  - **앞에서 자르는 가정**: 페이로드는 앞부분(구문 시작)에 판별 정보가 몰린다는 가정 — 논문에서 명시.
- **기본 폭 W=48**: EDA 근거(페이로드 98.6% 무손실 커버리지). 32/64 는 ablation 후보.
- **배치 빌드** `build_image_dataset.py` → `data/images/{track}_{split}_{text}_{side}.npz`
  - `.npz` 구성: `images`(N,side,side) uint8 / `labels`(N,) int / `classes`(K,) str.
  - PNG 수십만 장 대신 배열 하나로 저장(로딩 빠름, zero-padding 덕에 압축률 높음 — train 139,855장이 37MB).
  - 클래스별 샘플 그림을 `docs/figures/imaging/` 에 저장해 변환을 눈으로 검증.

### 관찰
- 대부분의 페이로드가 짧아 이미지 상단 일부만 채워지고 나머지는 0(검정)이다.
  이는 분포상 자연스러운 결과이며, CNN 이 "상단의 바이트 패턴"을 학습하게 된다.
- CmdI 는 길이가 길어 이미지를 더 많이 채운다(클래스 식별 신호가 될 수 있음 — 분석 대상).

---

## 4. 실행 순서 (재현)

```bash
python src/data/preprocess.py                          # 1) 전처리·분할 → data/processed/
python src/imaging/build_image_dataset.py --track payload_4class --text raw   # 2) 이미지화
python src/imaging/build_image_dataset.py --track csic_binary   --text raw
python -m pytest tests/ -q                             # 3) 변환·분할 무결성 테스트
```

디코딩본으로 만들려면 `--text decoded`, 크기 ablation 은 `--side 32` / `--side 64`.

---

## 5. Phase 4 로 넘기는 열린 결정

- [ ] CSIC 이미지화 단위(URL vs 전체 요청 직렬화)와 중복 제거 기준 확정 (2절 리스크)
- [ ] payload_4class 와 csic_binary 를 결합/전이할지, 독립 평가할지
- [ ] 디코딩 raw vs decoded, 이미지 크기 32/48/64 ablation 조합 확정
- [ ] 클래스 불균형(특히 CSIC) 대응: class weight vs oversampling
