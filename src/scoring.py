"""Score leads with the saved preprocessing and model pipeline."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "final_model.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"


def load_pipeline():
    """Load the saved pipeline and its feature metadata."""
    pipeline = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text())
    return pipeline, metadata


def score_leads(leads: pd.DataFrame, include_segment: bool = True) -> pd.DataFrame:
    """Return the input leads with a conversion score and optional segment."""
    if not isinstance(leads, pd.DataFrame):
        raise TypeError("leads must be a pandas DataFrame")

    pipeline, metadata = load_pipeline()
    feature_names = metadata["features"]
    missing_features = [name for name in feature_names if name not in leads.columns]
    if missing_features:
        raise ValueError(f"Input is missing required features: {missing_features}")

    features = leads[feature_names].copy()
    features = features.replace(r"^\s*$", np.nan, regex=True).replace("Select", np.nan)
    scored = leads.copy()
    scored["final_score"] = pipeline.predict_proba(features)[:, 1]

    if include_segment:
        cutoffs = metadata["segmentation"]
        low_max = cutoffs["low_max_score"]
        medium_max = cutoffs["medium_max_score"]
        scored["lead_segment"] = np.select(
            [scored["final_score"] <= low_max, scored["final_score"] <= medium_max],
            ["Low", "Medium"],
            default="High",
        )

    return scored


def main():
    parser = argparse.ArgumentParser(description="Score leads with the saved lead-conversion model.")
    parser.add_argument("--input", required=True, help="CSV file containing the model input columns")
    parser.add_argument("--output", required=True, help="Where to write the scored CSV")
    parser.add_argument(
        "--no-segmentation",
        action="store_true",
        help="Return final_score without adding a lead_segment column",
    )
    args = parser.parse_args()

    leads = pd.read_csv(args.input)
    scored = score_leads(leads, include_segment=not args.no_segmentation)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_path, index=False)
    print(f"Scored {len(scored):,} leads and saved results to {output_path}")


if __name__ == "__main__":
    main()
