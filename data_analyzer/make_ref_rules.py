from pathlib import Path
import pandas as pd

from dq_pipeline import (
    read_batch_csv, parse_types, clean_batch,
    mine_rules, load_yaml
)

def build_reference_from_raw(raw_dir: str, config_path: str, n_batches: int = 1):
    cfg = load_yaml(config_path)

    raw_dir = Path(raw_dir)
    batch_paths = sorted(raw_dir.glob("*.csv"))[:n_batches]
    if not batch_paths:
        raise FileNotFoundError(f"No csv batches found in {raw_dir}")

    dfs = []
    for p in batch_paths:
        df = read_batch_csv(p)
        df = parse_types(df, cfg)
        # light cleaning is OK for analysis
        df = clean_batch(df, cfg)
        dfs.append(df)

    base = pd.concat(dfs, ignore_index=True)

    # Try mining rules; if <5 rules found, relax thresholds
    apr = cfg["dq"]["apriori"]
    top_k_needed = int(apr.get("top_k_rules", 5))

    rules = mine_rules(base, cfg)

    # Auto-relax loop
    supports = [apr["min_support"], 0.03, 0.02, 0.01]
    confidences = [apr["min_confidence"], 0.4, 0.3, 0.2]
    lifts = [apr["min_lift"], 1.05, 1.0]

    if len(rules) < top_k_needed:
        for s in supports:
            for c in confidences:
                for lift_val in lifts:
                    cfg["dq"]["apriori"]["min_support"] = float(s)
                    cfg["dq"]["apriori"]["min_confidence"] = float(c)
                    cfg["dq"]["apriori"]["min_lift"] = float(lift_val)
                    rules = mine_rules(base, cfg)
                    if len(rules) >= top_k_needed:
                        break
                if len(rules) >= top_k_needed:
                    break
            if len(rules) >= top_k_needed:
                break

    ref_path = Path(cfg["dq"]["rules_reference"]["path"])
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    rules.to_csv(ref_path, index=False)

    print(f"Saved reference rules to: {ref_path}")
    print(f"Rules found: {len(rules)}")
    print(rules.head(10).to_string(index=False))


if __name__ == "__main__":
    # Use the local raw_data directory and fix config path
    build_reference_from_raw(
        raw_dir="./raw_data",
        config_path="data_analyzer/config.yaml",
        n_batches=5,   # Use fewer batches for faster processing
    )


