"""G4 고정 split 부가(docs/14 §8.14) — 라벨상 정상인 공격 탐침(E1-a)에서 넘김률·탐지율을 서술한다.

E1-a 는 SR-BH 원본에서 Normal 로 라벨됐지만 감사 패턴이 공격 탐침으로 판정한 행이다(분포 이동 세트).
트랙과 텍스트가 겹치는 행(overlap_split != "")은 학습·평가에 이미 들어간 것이라 뺀다.
판정이 아니라 서술이다: 권장 구성의 운영점(τ)을 고정 split val 에서 고른 값 그대로 쓰고 재튜닝하지 않는다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.eval.gate_g6_cv import gate_mask
from src.eval.metrics import wilson_ci
from src.models import cascade

E1A_CSV = PROJECT_ROOT / "data/processed/srbh_e1a.csv"
GATES = ("b0", "g2")   # H-G6 채택 점수가 없어 §8.14 의 게이트 집합은 B0·G2 뿐이다(§8.15).


def load_rows(path):
    frame = pd.read_csv(path, keep_default_na=False, dtype={"overlap_split": str})
    if not {"text_f2", "overlap_split", "audit_classes"} <= set(frame.columns):
        raise ValueError("E1-a CSV 에 text_f2·overlap_split·audit_classes 가 필요합니다.")
    return frame[frame.overlap_split == ""].reset_index(drop=True)


def build_tensors(texts, max_len=2304):
    """트랙 npz 와 같은 변환 함수를 메모리에서 호출한다(cross_validate.load_image_pool 과 같은 경로)."""
    import data_image
    import data_text
    from channel_encoders import DEFAULT_RGB_ENCODERS
    from src.imaging.payload_to_image import payload_to_rgb_image
    images = np.empty((len(texts), 48, 48, 3), dtype=np.uint8)
    for i, payload in enumerate(texts):
        images[i] = payload_to_rgb_image(payload, side=48, encoders=DEFAULT_RGB_ENCODERS)
    x_img = data_image.make_torch_dataset(images, np.zeros(len(texts), dtype=np.int64)).tensors[0]
    x_seq = torch.from_numpy(data_text.encode_byte_matrix(list(texts), max_len))
    return x_img, x_seq


def rate(mask):
    hits, n = int(mask.sum()), int(len(mask))
    return {"rate": hits / n if n else None, "ci95": list(wilson_ci(hits, n)) if n else None, "n": n}


def analyze(args):
    g6 = json.loads(args.g6_json.read_text(encoding="utf-8"))
    classes = g6["classes"]
    normal = classes.index("Normal")
    points = {g: g6["gates"][g]["operating_point"] for g in GATES}
    frame = load_rows(args.csv)
    if args.limit:
        frame = frame.head(args.limit)
    x_img, x_seq = build_tensors(frame.text_f2.tolist())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net1 = cascade.load_net("cnn", len(classes), "srbh_4class", "raw", False, device, "rgb", 3,
                            ("raw_byte", "char_class", "local_entropy"))
    net2 = cascade.load_net("charcnn", len(classes), "srbh_4class", "raw", True, device)
    p1 = cascade.predict_probs(net1, x_img, device, 512)
    p2 = cascade.predict_probs(net2, x_seq, device, 512)
    conf = p1.max(1)
    groups = {"all": np.ones(len(frame), bool)}
    groups.update({f"audit={name}": (frame.audit_classes == name).to_numpy()
                   for name in sorted(frame.audit_classes.unique())})
    result = {"n_rows": len(frame), "classes": classes, "tau": {g: points[g].get("tau", points[g].get("taus"))
                                                               for g in GATES},
              "stage_alone": {}, "gates": {}}
    for name, probs in (("RGB CNN", p1), ("char-CNN", p2)):
        result["stage_alone"][name] = {k: {"attack_rate": rate(probs[m].argmax(1) != normal)}
                                       for k, m in groups.items()}
    for g in GATES:
        mask = gate_mask(points[g], p1, conf)
        pred = np.where(mask, p2.argmax(1), p1.argmax(1))
        result["gates"][g] = {k: {"escalation": rate(mask[m]), "attack_rate": rate(pred[m] != normal),
                                  "val_r_N": points[g]["r_N_val"], "test_r_N": points[g]["r_N"]}
                              for k, m in groups.items()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g6-json", type=Path, required=True, help="gate_g6.py 고정 split 결과(운영점 τ 출처)")
    parser.add_argument("--csv", type=Path, default=E1A_CSV)
    parser.add_argument("--limit", type=int, help="스모크용 행 수 제한(저장하지 않는다)")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "experiments/results/gate_g4_e1a.json")
    args = parser.parse_args()
    try:
        if not args.limit and args.out.exists():
            raise FileExistsError(f"기존 결과를 덮어쓰지 않습니다: {args.out.name}")
        result = analyze(args)
        text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
        if not args.limit:
            args.out.write_text(text + "\n", encoding="utf-8")
        brief = {g: {k: (v["escalation"]["rate"], v["attack_rate"]["rate"], v["escalation"]["n"])
                     for k, v in item.items()} for g, item in result["gates"].items()}
        print(json.dumps({"stage_alone": {s: {k: v["attack_rate"]["rate"] for k, v in d.items()}
                                          for s, d in result["stage_alone"].items()},
                          "gates(escalation, attack_rate, n)": brief}, ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"G4 E1-a 서술 실패: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
