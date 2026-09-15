"""Normal 텍스트의 원본 UA를 모두 보존해 출처 혼합을 숨기지 않는다."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# CLI와 패키지 호출에서 같은 매핑 상수를 사용한다.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.analysis.build_srbh_e1 import load_inputs
from src.data.srbh_track import CLASS_CODES, compose_fields, detect_label_columns
from src.eval.metrics import wilson_ci

STRATA = ("real_user", "scanner", "mixed")


def assign_normal_strata(raw, flags, track) -> tuple[pd.DataFrame, dict]:
    """충돌 제거 전 매핑을 사용해야 스캐너 UA 최빈값의 모집단이 달라지지 않는다."""
    labels = detect_label_columns(raw)
    by_code = {name.split(" - ", 1)[0]: name for name in labels}
    if (not raw.index.is_unique or not raw[labels].isin(["0", "1"]).all().all()
            or not {code for codes in CLASS_CODES.values() for code in codes} <= by_code.keys()):
        raise ValueError("원본 인덱스 또는 매핑 라벨이 잘못되었습니다.")
    targets = pd.DataFrame({
        label: raw[[by_code[code] for code in codes]].eq("1").any(axis=1)
        for label, codes in CLASS_CODES.items()}, index=raw.index)
    single = targets.sum(axis=1).eq(1)
    attack = single & ~targets.Normal
    counts = raw.loc[attack, "request_user_agent"].value_counts()
    if counts.empty:
        raise ValueError("스캐너 UA를 정할 단일 클래스 공격 행이 없습니다.")
    scanner_ua = str(counts.index[0])
    ids = flags.row_id.astype(str)
    if not ids.str.fullmatch(r"[0-9]+").all():
        raise ValueError("감사 row_id는 0 이상의 정수여야 합니다.")
    ids = ids.map(int)
    normal = single & targets.Normal
    if not ids.isin(raw.index[normal]).all():
        raise ValueError("감사 row_id가 매핑 후 Normal 행을 가리키지 않습니다.")
    source = raw.loc[normal & ~raw.index.isin(ids)].copy()
    source["text_raw"] = compose_fields(source, "F2")
    source["is_scanner"] = source.request_user_agent.eq(scanner_ua)
    grouped = source.groupby("text_raw").is_scanner.agg(["all", "any"])
    mapping = pd.Series(np.where(grouped["all"], "scanner",
                                 np.where(grouped["any"], "mixed", "real_user")), index=grouped.index)
    frames, split_counts = [], {}
    for split, frame in track.items():
        selected = frame.loc[frame.label.eq("Normal"), ["row_id", "text_raw"]].copy()
        selected["normal_source"] = selected.text_raw.map(mapping)
        if selected.normal_source.isna().any():
            raise ValueError(f"{split} Normal 텍스트가 감사 제외 후 원본에 없습니다.")
        selected.insert(0, "split", split)
        frames.append(selected)
        split_counts[split] = {name: int(selected.normal_source.eq(name).sum()) for name in STRATA}
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["split", "row_id", "text_raw", "normal_source"])
    return result, {"scanner_ua": scanner_ua, "by_split": split_counts,
                    "total": {name: int(result.normal_source.eq(name).sum()) for name in STRATA}}


def fpr_by_stratum(y_true, y_pred, strata, normal_label="Normal") -> dict:
    """층별 분모를 Normal 정답으로 한정하여 공격 행이 FPR을 희석하지 않게 한다."""
    truth, pred, source = map(np.asarray, (y_true, y_pred, strata))
    if any(value.ndim != 1 for value in (truth, pred, source)) or not len(truth) == len(pred) == len(source):
        raise ValueError("정답·예측·층은 길이가 같은 1차원 배열이어야 합니다.")
    normal = truth == normal_label
    if not np.isin(source[normal], STRATA).all():
        raise ValueError("Normal 행에 알 수 없는 출처 층이 있습니다.")
    report = {}
    for name in (*STRATA, "all"):
        mask = normal if name == "all" else normal & (source == name)
        n, alarms = int(mask.sum()), int((mask & (pred != normal_label)).sum())
        report[name] = {"n": n, "false_alarms": alarms, "fpr": wilson_ci(alarms, n)}
    return report


def main() -> int:
    """출처 표만 별도 저장하여 기존 분할과 라벨 지문을 보존한다."""
    try:
        frame, stats = assign_normal_strata(*load_inputs())
        csv_path = PROJECT_ROOT / "data/processed/srbh_4class_normal_strata.csv"
        json_path = PROJECT_ROOT / "experiments/results/srbh_normal_strata.json"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(csv_path, index=False, encoding="utf-8")
        summary = json.dumps(stats, ensure_ascii=False, indent=2, allow_nan=False)
        json_path.write_text(summary + "\n", encoding="utf-8")
        print(summary)
    except (OSError, ValueError, KeyError) as error:
        print(f"Normal 출처 층 생성 실패: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
