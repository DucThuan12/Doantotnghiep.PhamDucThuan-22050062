from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation_metrics import evaluate_csv, write_json


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tính các chỉ số thực nghiệm FitMotion AI từ CSV có ground truth và prediction."
    )
    parser.add_argument("csv_path", help="Đường dẫn file CSV kết quả thực nghiệm")
    parser.add_argument("--output", help="Ghi kết quả ra JSON")
    args = parser.parse_args()

    summary = evaluate_csv(args.csv_path)
    payload = summary.to_dict()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.output:
        write_json(summary, args.output)
        print(f"Đã ghi: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
