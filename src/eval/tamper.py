"""Phase 12 (M1) — 표현 간 불일치를 '회피 시도' 신호로 쓰는 순수 로직

무엇을 하나:
    캐스케이드는 1차(RGB CNN, 이미지 표현)와 2차(char-CNN, 텍스트 표현)의 확률을 **둘 다**
    가지고 있다. 이 모듈은 그 두 확률분포의 불일치를 하나의 점수(tampering_score)로 바꾸고,
    "이 요청이 탐지 회피를 시도했는가"를 판별하는 데 필요한 임계값·지표를 계산한다.

왜 이게 성립하나(전부 기존 실측, docs/11 §3.1):
    · clean       : 두 모델의 탐지집합이 **완전 일치**(F4) → 잡음이 거의 없다.
    · 변형 k=5    : any-misclass 가 char-CNN 0.224 vs RGB CNN 0.598(F9) → 격렬하게 갈린다.
    잡음이 작고 신호가 큰 조합이므로, '불일치'는 그 자체로 변형 탐지기가 된다.

⚠️ 주장 강도(docs/11 §3.3, docs/12 §4.2):
    "표현 불일치를 이상 신호로 쓴다"는 발상 자체는 **Feature Squeezing(NDSS 2018)이 선행**이다.
    새로 주장하는 것은 단 하나 — 캐스케이드는 어차피 2차를 돌리므로 **모드 A 에서 추가 비용이
    정확히 0** 이라는 점이다. 코드에서도 모드 A/B 를 분리해 계산하는 이유가 이것이다.

이 파일에 torch·모델 로딩은 없다(확률 행렬만 받는다). 그래야 단위 테스트가 GPU 없이 돈다.
CLI·모델 로딩·변형 생성은 run_tamper.py 가 담당한다.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

# 사전 지정 주 점수 함수(docs/11 §3.2 의 ablation 3종 중 하나를 **실측 전에** 못박는다).
# 왜 js_divergence 인가: 분포 전체를 쓰고(top-1 만 보는 이진 지표보다 정보량이 많다),
# 대칭이며(어느 쪽이 '정답'이라 가정하지 않는다), [0,1] 로 유계라 임계값 해석이 쉽다.
# ⚠️ 결과를 보고 주 지표를 바꾸면 HARKing 이다. 나머지 둘은 ablation 으로만 보고한다.
PRIMARY_SCORER = "js_divergence"

# 사전 지정 주 운영 모드. 모드 A(에스컬레이션분 한정)만이 "추가 비용 0"이라는 M1 의 유일한
# 기여를 만족한다. 모드 B(전량)는 상한 참고용이다(비용이 융합형과 같아진다).
PRIMARY_MODE = "mode_a"


# ---------------------------------------------------------------------------
# 점수 함수 레지스트리 — D(p_stage1, p_stage2)
# ---------------------------------------------------------------------------
def _top1_disagree(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """① top-1 불일치(이진). 두 표현이 서로 다른 클래스로 판정하면 1.

    가장 단순하고 운영자가 이해하기 쉽지만, 값이 0/1 뿐이라 임계값을 움직일 수 없다
    (ROC 곡선이 점 하나로 무너진다). ablation 의 하한선 역할.
    """
    return (p1.argmax(axis=1) != p2.argmax(axis=1)).astype(np.float64)


def _js_divergence(p1: np.ndarray, p2: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """② Jensen-Shannon divergence(밑 2). 두 확률분포 전체의 거리 → [0, 1].

    KL 과 달리 대칭이고 유계라, "한쪽이 옳다"는 가정 없이 불일치의 크기만 잰다.
    top-1 이 같아도 확신도 분포가 갈리면 값이 오르므로 이진 지표보다 민감하다.
    """
    m = 0.5 * (p1 + p2)

    def kl(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # a*log(a/b) 에서 a=0 인 항은 0 으로 수렴하지만 log(0) 이 -inf 라 eps 로 막는다.
        return np.sum(a * (np.log2(a + eps) - np.log2(b + eps)), axis=1)

    js = 0.5 * kl(p1, m) + 0.5 * kl(p2, m)
    # 수치오차로 아주 작은 음수/1 초과가 나올 수 있어 정의역으로 되돌린다.
    return np.clip(js, 0.0, 1.0)


def _conf_gap(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """③ 확신도 차 = max(p2) − max(p1). 부호 있는 값으로 둔다.

    왜 이 방향인가: 회피 변형은 이미지 표현(1차)을 크게 흔들고 텍스트 표현(2차)은 상대적으로
    덜 흔든다(F9: 변형 k=5 에서 any-misclass 0.598 vs 0.224). 즉 변형되면 1차 확신도가
    떨어지고 2차는 버티므로 이 값이 **커진다**. 절댓값을 쓰면 그 방향 정보가 사라진다.
    """
    return p2.max(axis=1) - p1.max(axis=1)


SCORERS = {
    "top1_disagree": _top1_disagree,
    "js_divergence": _js_divergence,
    "conf_gap": _conf_gap,
}


def tampering_scores(p1: np.ndarray, p2: np.ndarray, scorer: str) -> np.ndarray:
    """(N,K) 확률 두 개 → (N,) tampering 점수. 클수록 '회피 시도 의심'."""
    if scorer not in SCORERS:
        raise ValueError(f"알 수 없는 점수 함수: {scorer} (가능: {sorted(SCORERS)})")
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    if p1.shape != p2.shape:
        raise ValueError(f"두 단계의 확률 형상이 다릅니다: {p1.shape} vs {p2.shape}")
    return SCORERS[scorer](p1, p2)


# ---------------------------------------------------------------------------
# 모드 A — 에스컬레이션된 샘플에만 정의된다
# ---------------------------------------------------------------------------
def escalated_mask(p1: np.ndarray, tau: float) -> np.ndarray:
    """캐스케이드가 2차를 실제로 호출한 샘플(= 모드 A 의 판정 가능 집합).

    cascade.cascade_apply 와 **같은 규칙**(확신도 < τ)이어야 한다. 규칙이 어긋나면
    "추가 비용 0" 이라는 주장 자체가 거짓이 된다(안 돌린 2차의 확률을 쓰게 되므로).
    """
    return np.asarray(p1).max(axis=1) < tau


# ---------------------------------------------------------------------------
# 임계값 — clean val 에서 확정하고 test 에 1회 적용 (docs/09 §3.1 원칙 계승)
# ---------------------------------------------------------------------------
def select_threshold(clean_scores: np.ndarray, target_fpr: float) -> float:
    """clean(정상 운영) 점수 분포에서 목표 오탐률에 해당하는 임계값을 고른다.

    왜 clean 만 보는가: 공격(변형) 데이터를 보고 임계값을 고르면 공격 실험으로 운영점을
    튜닝하는 셈이라 누수다. 캐스케이드의 τ 를 clean val 에서만 고르는 것과 같은 원칙이다.

    판정 규칙은 `score > threshold` 다(초과, 이상 아님). 이진 점수(top1_disagree)처럼
    분포가 뭉쳐 있을 때 분위수가 0.0 으로 나와도, 초과 규칙이면 "불일치한 것만" 플래그되어
    의도대로 동작한다.
    """
    scores = np.asarray(clean_scores, dtype=np.float64)
    if scores.size == 0:
        # 판정 가능한 clean 샘플이 없다 = 임계값을 정할 근거가 없다.
        # 무한대를 돌려 아무것도 플래그하지 않게 한다(조용히 전부 탐지하는 것보다 안전).
        return float("inf")
    return float(np.quantile(scores, 1.0 - target_fpr))


def flag(scores: np.ndarray, threshold: float) -> np.ndarray:
    """임계값 초과 = '회피 시도 의심' 플래그."""
    return np.asarray(scores, dtype=np.float64) > threshold


# ---------------------------------------------------------------------------
# 이진 판별 성능
# ---------------------------------------------------------------------------
def binary_eval(neg_scores: np.ndarray, pos_scores: np.ndarray,
                threshold: float) -> dict:
    """음성(clean) vs 양성(변형) 점수로 판별 성능을 낸다.

    ROC-AUC 는 임계값과 무관한 판별력(H5-1 의 기준)이고, TPR/FPR 은 확정된 임계값에서의
    실제 운영 성능이다. 둘을 함께 봐야 "AUC 는 높은데 쓸 임계값이 없다"는 함정을 피한다.
    """
    neg = np.asarray(neg_scores, dtype=np.float64)
    pos = np.asarray(pos_scores, dtype=np.float64)

    auc = None
    if len(neg) and len(pos):
        y = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))])
        s = np.concatenate([neg, pos])
        # 점수가 전부 같으면(예: 모두 불일치 없음) AUC 가 정의되지 않는다 → None.
        if np.ptp(s) > 0:
            auc = float(roc_auc_score(y, s))
        else:
            auc = 0.5  # 완전 무정보. 계산 실패가 아니라 '판별력 없음'이므로 값으로 남긴다.

    return {
        "n_negative": int(len(neg)),
        "n_positive": int(len(pos)),
        "roc_auc": auc,
        "tpr": float(flag(pos, threshold).mean()) if len(pos) else None,
        "fpr": float(flag(neg, threshold).mean()) if len(neg) else None,
    }


# ---------------------------------------------------------------------------
# G3 — 변형 조건에서의 탐지 상보성
# ---------------------------------------------------------------------------
def complementarity(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
                    normal_idx: int, mask: np.ndarray | None = None) -> dict:
    """두 모델의 '무엇을 잡고 무엇을 놓치는가'를 두 기준으로 교차 집계한다.

    - detection : 예측이 Normal 이 아니면 성공(= WAF 관점의 미탐 여부). detection_analysis.py
                  의 clean 전용 분석과 **같은 정의**라 표를 나란히 놓을 수 있다.
    - correct   : 예측이 진짜 클래스와 일치해야 성공.

    ⚠️ 왜 두 기준인가(docs/11 §3.4): 변형 하에서 벌어지는 격차는 대부분 **미탐이 아니라
    오귀속**(공격을 다른 공격 유형으로 부름)이다. detection 기준만 보면 격차가 0 처럼 보이고,
    correct 기준만 보면 "더 많이 탐지한다"로 과장된다. 둘을 같이 내야 정직한 서술이 된다.
    """
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    if not (len(y_true) == len(pred_a) == len(pred_b)):
        raise ValueError("표본 수가 다릅니다 — 같은 조건의 예측인지 확인하세요.")

    # 진짜 공격 샘플만 대상(정상→정상은 탐지 이슈가 아니다). mask 로 추가 제한 가능.
    target = y_true != normal_idx
    if mask is not None:
        target = target & np.asarray(mask, dtype=bool)

    out = {"n_attacks": int(target.sum())}
    for name, ok_a, ok_b in (
        ("detection", pred_a != normal_idx, pred_b != normal_idx),
        ("correct", pred_a == y_true, pred_b == y_true),
    ):
        ok_a = ok_a & target
        ok_b = ok_b & target
        out[name] = {
            "both": int((ok_a & ok_b).sum()),
            "a_only": int((ok_a & ~ok_b & target).sum()),
            "b_only": int((~ok_a & ok_b & target).sum()),
            "neither": int((~ok_a & ~ok_b & target).sum()),
            "a_rate": float(ok_a[target].mean()) if target.any() else None,
            "b_rate": float(ok_b[target].mean()) if target.any() else None,
        }
    return out


# ---------------------------------------------------------------------------
# 판정 (docs/11 §7 — 실측 **전에** 고정된 기준)
# ---------------------------------------------------------------------------
def verdict(auc: float | None, clean_normal_fpr: float | None,
            coverage: float | None) -> dict:
    """H5-1 / H5-2 를 기준표 그대로 자동 판정한다.

    왜 코드가 판정하나: 사람이 결과를 보고 문턱을 조정하는 것을 구조적으로 막기 위해서다.
    기준은 docs/11 §7 에 실측 전 고정돼 있고, 여기 상수와 그 표가 어긋나면 안 된다.
        H5-1: ROC-AUC ≥ 0.90 효과 있음 / 0.75~0.90 보조 지표로만 / < 0.75 기각
        H5-2: clean Normal 오탐 ≤ 1% 이고 모드 A 커버리지 ≥ 30%
    """
    if auc is None:
        h51 = "판정 불가(양성/음성 표본 부족)"
    elif auc >= 0.90:
        h51 = "효과 있음"
    elif auc >= 0.75:
        h51 = "보조 지표로만 보고"
    else:
        h51 = "기각"

    practical = (clean_normal_fpr is not None and clean_normal_fpr <= 0.01
                 and coverage is not None and coverage >= 0.30)
    return {
        "H5-1": {"criterion": "ROC-AUC >= 0.90", "value": auc, "result": h51},
        "H5-2": {"criterion": "clean Normal FPR <= 0.01 and mode-A coverage >= 0.30",
                 "clean_normal_fpr": clean_normal_fpr, "coverage": coverage,
                 "result": "충족" if practical else "미충족"},
    }
