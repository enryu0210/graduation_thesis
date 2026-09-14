"""선택용 train·val만으로 SR-BH 필드 길이, 신호 위치와 지름길 대리치를 측정한다."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib.externals.loky import get_reusable_executor
from joblib.externals.loky.backend.utils import kill_process_tree
from sklearn.feature_extraction.text import TfidfVectorizer

# 직접 실행과 패키지 import가 같은 공용 구현을 사용하게 한다.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.data.audit_srbh_labels import SIGNATURES_V1, scan_frame
from src.data.download_srbh import RAW_DIR, FILENAME
from src.data.preprocess import normalize_text
from src.models.baseline_tfidf import build_classifier
from src.eval.metrics import compute_metrics

FIELDS = dict(uri="request_http_request", body="request_body",
              cookie="request_cookie", ua="request_user_agent", referer="request_referer")
COMBINATIONS = {f"F{i}": tuple(FIELDS)[:i] for i in range(1, 5)}
BYTE_LIMIT = 2304
TRAINING_LIMIT_SECONDS = 600


def build_combinations(frame: pd.DataFrame) -> pd.DataFrame:
    """빈 필드의 구분자도 보존해야 기존 트랙과 같은 바이트 표현이 된다."""
    required = {*FIELDS.values(), "text_raw", "label", "row_id"}
    missing = sorted(required - set(frame.columns))
    if missing or frame.empty:
        raise ValueError(f"입력이 비었거나 필수 컬럼이 없습니다: {missing}")
    if not frame[list(required - {"row_id"})].map(lambda x: isinstance(x, str)).all().all():
        raise ValueError("row_id 이외의 입력은 결측 없는 문자열이어야 합니다.")
    if (not pd.api.types.is_integer_dtype(frame.row_id) or frame.row_id.lt(0).any()
            or frame.row_id.duplicated().any()):
        raise ValueError("row_id는 중복 없는 0 이상의 정수여야 합니다.")
    result = pd.DataFrame(index=frame.index)
    for name, fields in COMBINATIONS.items():
        result[name] = frame[FIELDS[fields[0]]]
        for field in fields[1:]:
            result[name] = result[name] + "\n" + frame[FIELDS[field]]
    mismatch = result.F2.ne(frame.text_raw)
    if mismatch.any():
        raise ValueError(f"F2 != text_raw: row_id={frame.loc[mismatch, 'row_id'].head().tolist()}")
    return result


def measure_lengths(frame: pd.DataFrame) -> dict:
    combinations = build_combinations(frame)
    texts = frame[list(FIELDS.values())].rename(columns={v: k for k, v in FIELDS.items()})
    lengths = pd.concat([texts, combinations], axis=1).map(lambda x: len(x.encode("utf-8")))
    result = {"byte_limit": BYTE_LIMIT, "quantile_method": "linear", "by_class": {}}
    for label in ["overall", *sorted(frame.label.unique())]:
        subset = lengths if label == "overall" else lengths.loc[frame.label.eq(label)]
        result["by_class"][label] = {
            "n_rows": len(subset),
            "lengths": {name: dict(zip(("p50", "p95", "p99", "max"),
                           [*map(float, subset[name].quantile([.5, .95, .99])), int(subset[name].max())]))
                        for name in subset},
            "exceeds": {name: {"count": int(subset[name].gt(BYTE_LIMIT).sum()),
                               "rate": float(subset[name].gt(BYTE_LIMIT).mean())}
                        for name in COMBINATIONS},
        }
    return result


def measure_signals(frame: pd.DataFrame) -> dict:
    build_combinations(frame)
    indexed = frame.set_index("row_id")
    # 감사 함수가 디코딩과 IGNORECASE 검색을 담당하므로 패턴 사본이 생기지 않는다.
    flags = scan_frame(indexed, SIGNATURES_V1).set_index("row_id")
    hit_sets = flags.fields.str.split(";").map(set).to_dict()
    result = {"signature_version": "v1", "signature_count": len(SIGNATURES_V1), "by_class": {}}
    for label, subset in indexed.groupby("label", sort=True):
        counts = dict(in_f2=0, outside_f2_only=0, no_hit=0)
        outside_counts = dict(cookie=0, ua=0, referer=0)
        samples = []
        for row_id, row in subset.sort_index().iterrows():
            hits = hit_sets.get(row_id, set())
            if hits & {FIELDS["uri"], FIELDS["body"]}:
                counts["in_f2"] += 1
            elif hits:
                counts["outside_f2_only"] += 1
                fields = [name for name in outside_counts if FIELDS[name] in hits]
                for name in fields:
                    outside_counts[name] += 1
                if len(samples) < 5:
                    samples.append({"row_id": int(row_id), "hit_fields": fields,
                                    "decoded_prefixes": {name: normalize_text(row[FIELDS[name]])[:200]
                                                         for name in fields}})
            else:
                counts["no_hit"] += 1
        assert sum(counts.values()) == len(subset), "신호 위치 집계 합계가 행 수와 다릅니다."
        result["by_class"][label] = {
            "n_rows": len(subset),
            **{name: {"count": count, "rate": count / len(subset)} for name, count in counts.items()},
            "partition_verified": True, "outside_field_counts": outside_counts, "samples": samples,
        }
    return result


def _classifier_worker(connection, train_texts, val_texts, y_train, y_val, class_names):
    """Windows에서도 시간 초과 학습을 확실히 종료할 수 있도록 프로세스를 분리한다."""
    try:
        started = time.perf_counter()
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), max_features=50000,
                                     lowercase=False, dtype=np.float32)
        x_train = vectorizer.fit_transform(train_texts)
        vectorize_seconds = time.perf_counter() - started
        classifier = build_classifier("logreg", len(class_names))
        started = time.perf_counter()
        classifier.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - started
        x_val = vectorizer.transform(val_texts)
        metrics = compute_metrics(y_val, classifier.predict(x_val), class_names,
                                  y_score=classifier.predict_proba(x_val))
        outcome = {"metrics": metrics, "fit_seconds": fit_seconds,
                   "vectorize_seconds": vectorize_seconds}
    except Exception as error:
        outcome = {"error": f"{type(error).__name__}: {error}"}
    finally:
        # n_jobs=-1이 만든 하위 프로세스를 남긴 채 부모만 끝내지 않는다.
        get_reusable_executor().shutdown(wait=True, kill_workers=True)
        connection.send(outcome)
        connection.close()


def measure_shortcuts(train: pd.DataFrame, val: pd.DataFrame) -> dict:
    train_combinations, val_combinations = build_combinations(train), build_combinations(val)
    classes = sorted(train.label.unique())
    if len(classes) < 2 or not set(val.label) <= set(classes):
        raise ValueError("학습 클래스가 2개 미만이거나 검증에 미등록 클래스가 있습니다.")
    # 최빈값 동률은 문자열 정렬로 고정하며 val은 기준 선택에 관여하지 않는다.
    common_ua = sorted(train[FIELDS["ua"]].mode().tolist())[0]
    result = {"proxy": True, "note": "참고용 대리치이며 단계2 CV 판정이 아닙니다.",
              "class_names": classes, "most_common_train_ua": common_ua,
              "by_split": {}, "classifiers": {}, "status": "complete",
              "training_limit_seconds": TRAINING_LIMIT_SECONDS,
              "timeout_scope": "세 분류기의 프로세스 시작·벡터화·학습·평가 전체 합계"}
    for split, frame in (("train", train), ("val", val), ("train_val", pd.concat([train, val]))):
        result["by_split"][split] = {
            label: {"n_rows": len(group),
                    "common_ua_rate": float(group[FIELDS["ua"]].eq(common_ua).mean()),
                    "cookie_nonempty_rate": float(group[FIELDS["cookie"]].ne("").mean()),
                    "unique_ua_count": int(group[FIELDS["ua"]].nunique())}
            for label, group in frame.groupby("label", sort=True)}
    encoding = {name: i for i, name in enumerate(classes)}
    y_train, y_val = train.label.map(encoding).to_numpy(), val.label.map(encoding).to_numpy()
    deadline = time.perf_counter() + TRAINING_LIMIT_SECONDS
    context = mp.get_context("spawn")
    for name, combination in (("ua_cookie_only", None), ("f2", "F2"), ("f4", "F4")):
        texts = [(frame[FIELDS["ua"]] + "\n" + frame[FIELDS["cookie"]]).tolist()
                 if combination is None else combinations[combination].tolist()
                 for frame, combinations in ((train, train_combinations), (val, val_combinations))]
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(target=_classifier_worker,
                                  args=(sender, *texts, y_train, y_val, classes))
        try:
            process.start()
            sender.close()
            if not receiver.poll(max(0, deadline - time.perf_counter())):
                result.update(status="timeout", interrupted_classifier=name)
                break
            outcome = receiver.recv()
            if "error" in outcome:
                raise RuntimeError(f"{name}: {outcome['error']}")
            result["classifiers"][name] = outcome
        finally:
            # 응답을 받기 전 join하면 파이프 버퍼가 가득 차 교착될 수 있다.
            if process.pid is not None:
                process.join(timeout=1)
                if process.is_alive():
                    kill_process_tree(process)
                    process.join()
            receiver.close()
            sender.close()
    return result


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = [PROJECT_ROOT / "data/processed" / f"srbh_4class_{split}.csv" for split in ("train", "val")]
    source = RAW_DIR / FILENAME
    for path in [*paths, source]:
        if not path.is_file():
            raise FileNotFoundError(f"입력 파일 없음: {path.name}. 재생성 순서: "
                                    "download_srbh.py → audit_srbh_labels.py → process_track('srbh_4class')")
    frames = [pd.read_csv(path, dtype=str, keep_default_na=False) for path in paths]
    referers = pd.read_csv(source, usecols=[FIELDS["referer"]], dtype=str, keep_default_na=False)
    for frame in frames:
        frame["row_id"] = frame.row_id.astype(int)
        if not frame.row_id.between(0, len(referers) - 1).all():
            raise ValueError("row_id가 원본 행 범위를 벗어났습니다.")
        frame[FIELDS["referer"]] = referers.iloc[frame.row_id][FIELDS["referer"]].to_numpy()
    if set(frames[0].row_id) & set(frames[1].row_id):
        raise ValueError("train과 val에 중복 row_id가 있습니다.")
    return tuple(frames)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "experiments/results/srbh_e2_stage1.json")
    args = parser.parse_args()
    started = time.perf_counter()
    result = {"timings_seconds": {}}
    try:
        train, val = load_inputs()
        frame = pd.concat([train, val], ignore_index=True)
        build_combinations(frame)
        result.update(n_train=len(train), n_val=len(val), n_total=len(frame), f2_equals_text_raw=True)
        print(f"행 수: train={len(train)}, val={len(val)}, 합계={len(frame)}; F2 == text_raw 전체 검증 통과", flush=True)
        result["timings_seconds"]["load_validate"] = time.perf_counter() - started
        for name, function, inputs in (("A", measure_lengths, (frame,)),
                                       ("B", measure_signals, (frame,)),
                                       ("C", measure_shortcuts, (train, val))):
            begin = time.perf_counter()
            result[name] = function(*inputs)
            result["timings_seconds"][name] = time.perf_counter() - begin
            if name == "A":
                summary = result[name]["by_class"]["overall"]["exceeds"]
            elif name == "B":
                summary = {label: {key: stats[key] for key in ("in_f2", "outside_f2_only", "no_hit")}
                           for label, stats in result[name]["by_class"].items()}
            else:
                summary = {key: value["metrics"]["macro_f1"] for key, value in result[name]["classifiers"].items()}
            print(f"{name}: {json.dumps(summary, ensure_ascii=False)} ({result['timings_seconds'][name]:.2f}초)", flush=True)
            # 이후 학습이 실패해도 완료된 측정값은 잃지 않는다.
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        if result["C"]["status"] != "complete":
            print("학습 10분 제한 초과로 중단했습니다.", flush=True)
            return 1
        return 0
    except (OSError, ValueError, RuntimeError, EOFError) as error:
        print(f"측정 실패: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
