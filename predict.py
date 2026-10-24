"""Make predictions with the trained model.

Examples:
    python predict.py --sample                       # score 5 held-out rows saved at training time
    python predict.py --csv new_rows.csv             # score your own rows
    python predict.py --csv new_rows.csv --out scored.csv

Every run is appended to logs/predictions.jsonl for drift monitoring (disable with --no-log).
"""
import argparse
from pathlib import Path

import pandas as pd

from common import default_model_dir, load_model, log_predictions, predict_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", help="CSV of rows to score (target column not needed)")
    parser.add_argument("--sample", action="store_true", help="Score 5 held-out rows from training")
    parser.add_argument("--out", help="Write inputs + predictions to this CSV")
    parser.add_argument("--model-dir", help="Model folder (default: ./model)")
    parser.add_argument("--no-log", action="store_true", help="Do not write to the prediction log")
    args = parser.parse_args()

    try:
        pipeline, meta = load_model(args.model_dir)
    except FileNotFoundError as err:
        raise SystemExit(str(err))

    model_dir = Path(args.model_dir) if args.model_dir else default_model_dir()
    if args.csv:
        df = pd.read_csv(args.csv)
    elif args.sample:
        df = pd.read_csv(model_dir / "reports" / "holdout_sample.csv").head(5)
    else:
        raise SystemExit("Provide --csv FILE or --sample.")

    try:
        result = predict_frame(pipeline, meta, df)
    except ValueError as err:
        raise SystemExit(f"Error: {err}")

    print(result.to_string())
    if meta.get("positive_class") is not None:
        print(f"\n'probability' is P({meta['target']} = {meta['positive_class']}).")
    if meta["target"] in df.columns and args.sample:
        print("Actual values:", df[meta["target"]].tolist())

    if not args.no_log:
        n = log_predictions(df[meta["features"]], result, source="cli")
        print(f"Logged {n} predictions for monitoring.")
    if args.out:
        pd.concat([df, result], axis=1).to_csv(args.out, index=False)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
