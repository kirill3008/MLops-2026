from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from mlxtend.frequent_patterns import apriori, association_rules


# ---------- config ----------
def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------- parsing ----------
def read_batch_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)

def _parse_items(items_str: str) -> List[str]:
    # "A, B, C" -> ["A","B","C"]
    if items_str is None or (isinstance(items_str, float) and np.isnan(items_str)):
        return []
    s = str(items_str).strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def evaluate_reference_rules_on_batch(parsed_df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    cons_cfg = cfg["dq"].get("consistency", {})
    if not cons_cfg.get("enabled", True):
        return {"enabled": False}

    ref_path = Path(cfg["dq"]["rules_reference"]["path"])
    if not ref_path.exists():
        return {"enabled": True, "error": f"reference rules not found at {ref_path}"}

    ref_rules = pd.read_csv(ref_path)
    if ref_rules.empty:
        return {"enabled": True, "error": "reference rules file is empty"}

    # Build binary items for this batch
    bin_df = binary_transactions(parsed_df, cfg)  # columns = items

    # Thresholds
    min_ant_cnt = int(cons_cfg.get("min_antecedent_count", 30))
    min_conf_abs = float(cons_cfg.get("min_confidence_abs", 0.30))
    conf_drop_ratio = float(cons_cfg.get("confidence_drop_ratio", 0.70))
    supp_drop_ratio = float(cons_cfg.get("support_drop_ratio", 0.50))

    results = []
    n_rows = len(parsed_df)

    for _, r in ref_rules.iterrows():
        ant = _parse_items(r.get("antecedents_str", ""))
        con = _parse_items(r.get("consequents_str", ""))

        # If some item column is missing in this batch -> treat as all zeros (rule can't trigger)
        # We'll create masks safely.
        def col_mask(item: str) -> pd.Series:
            if item in bin_df.columns:
                return bin_df[item].astype(bool)
            # missing column => never true
            return pd.Series(False, index=bin_df.index)

        if not ant or not con:
            continue

        ant_mask = pd.Series(True, index=bin_df.index)
        for item in ant:
            ant_mask &= col_mask(item)

        con_mask = pd.Series(True, index=bin_df.index)
        for item in con:
            con_mask &= col_mask(item)

        ant_count = int(ant_mask.sum())
        both_count = int((ant_mask & con_mask).sum())

        support_new = both_count / n_rows if n_rows else 0.0
        confidence_new = (both_count / ant_count) if ant_count > 0 else 0.0

        # Baseline from reference file
        support_ref = float(r.get("support", np.nan))
        confidence_ref = float(r.get("confidence", np.nan))
        lift_ref = float(r.get("lift", np.nan))

        # Flags
        low_sample = ant_count < min_ant_cnt
        flag_conf_drop = (not low_sample) and (confidence_ref == confidence_ref) and (confidence_new < confidence_ref * conf_drop_ratio)
        flag_support_drop = (not low_sample) and (support_ref == support_ref) and (support_new < support_ref * supp_drop_ratio)
        flag_conf_abs = (not low_sample) and (confidence_new < min_conf_abs)

        results.append({
            "antecedents": ant,
            "consequents": con,
            "support_ref": support_ref,
            "confidence_ref": confidence_ref,
            "lift_ref": lift_ref,
            "support_new": support_new,
            "confidence_new": confidence_new,
            "antecedent_count": ant_count,
            "both_count": both_count,
            "low_sample": low_sample,
            "flag_conf_drop": bool(flag_conf_drop),
            "flag_support_drop": bool(flag_support_drop),
            "flag_conf_abs": bool(flag_conf_abs),
        })

    df_res = pd.DataFrame(results)
    if df_res.empty:
        return {"enabled": True, "n_rules_checked": 0, "any_issue": False, "rules": []}

    # overall flags
    any_issue = bool((df_res["flag_conf_drop"] | df_res["flag_support_drop"] | df_res["flag_conf_abs"]).any())

    return {
        "enabled": True,
        "n_rows": n_rows,
        "n_rules_checked": int(len(df_res)),
        "any_issue": any_issue,
        "rules": df_res.to_dict(orient="records"),
    }


def build_reference_rules_if_missing(parsed_df: pd.DataFrame, cfg: Dict[str, Any]) -> Optional[Path]:
    ref_cfg = cfg["dq"].get("rules_reference", {})
    if not ref_cfg.get("enabled", True):
        return None

    ref_path = Path(ref_cfg.get("path", "artifacts/rules/reference_rules.csv"))
    ref_path.parent.mkdir(parents=True, exist_ok=True)

    if ref_path.exists():
        return ref_path

    if not ref_cfg.get("build_if_missing", True):
        return None

    rules = mine_rules(parsed_df, cfg)  # returns support/confidence/lift + strings
    # Важно: убедитесь что top_k_rules >= 5 и min_support/ min_confidence не слишком строгие
    rules.to_csv(ref_path, index=False)
    return ref_path


def parse_types(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    parsing = cfg["dq"]["parsing"]
    date_format = parsing["date_format"]
    date_cols = parsing.get("date_cols", [])

    for col in date_cols:
        if col in df.columns:
            s = df[col].astype("string").str.strip().str.upper()
            df[col] = pd.to_datetime(s, format=date_format, errors="coerce")

    # numeric coercion for known numeric columns if exist
    numeric_candidates = [
        "EFFECTIVE_YR", "INSR_TYPE", "SEX",
        "INSURED_VALUE", "PREMIUM", "OBJECT_ID", "PROD_YEAR",
        "SEATS_NUM", "CARRYING_CAPACITY", "CCM_TON", "CLAIM_PAID",
    ]
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ---------- DQ metrics ----------
def compute_dq_metrics(df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    n = len(df)
    missing_per_col = df.isna().mean().to_dict() if n else {}
    missing_total_ratio = float(df.isna().mean().mean()) if n else 1.0
    duplicate_ratio = float(df.duplicated().mean()) if n else 0.0

    time_col = cfg["dq"]["parsing"]["time_col"]
    bad_time_ratio = float(df[time_col].isna().mean()) if (n and time_col in df.columns) else 1.0

    bad_date_ratio = 0.0
    if "INSR_BEGIN" in df.columns and "INSR_END" in df.columns and n:
        wrong_order = (df["INSR_BEGIN"].notna() & df["INSR_END"].notna() & (df["INSR_BEGIN"] > df["INSR_END"])).mean()
        bad_begin = df["INSR_BEGIN"].isna().mean()
        bad_end = df["INSR_END"].isna().mean()
        bad_date_ratio = float(max(wrong_order, bad_begin, bad_end))
    elif n:
        # если дат нет — считаем как 0 (или 1, зависит от политики)
        bad_date_ratio = 0.0

    validity = cfg["dq"]["validity"]
    invalid_breakdown: Dict[str, float] = {}

    def add_invalid(name: str, mask: pd.Series):
        invalid_breakdown[name] = float(mask.mean()) if n else 0.0

    if "SEX" in df.columns and n:
        allowed = set(validity.get("allowed_sex", []))
        add_invalid("sex_unknown", df["SEX"].notna() & (~df["SEX"].astype("Int64").isin(allowed)))

    if "INSR_TYPE" in df.columns and n:
        allowed = set(validity.get("allowed_insr_type", []))
        add_invalid("insr_type_unknown", df["INSR_TYPE"].notna() & (~df["INSR_TYPE"].astype("Int64").isin(allowed)))

    if "PROD_YEAR" in df.columns and n:
        mn, mx = validity["prod_year_min"], validity["prod_year_max"]
        add_invalid("prod_year_out_of_range", df["PROD_YEAR"].notna() & ((df["PROD_YEAR"] < mn) | (df["PROD_YEAR"] > mx)))

    if "SEATS_NUM" in df.columns and n:
        mn, mx = validity["seats_min"], validity["seats_max"]
        add_invalid("seats_out_of_range", df["SEATS_NUM"].notna() & ((df["SEATS_NUM"] < mn) | (df["SEATS_NUM"] > mx)))

    for col in validity.get("non_negative_cols", []):
        if col in df.columns and n:
            add_invalid(f"{col}_negative", df[col].notna() & (df[col] < 0))

    invalid_ratio = float(max(invalid_breakdown.values())) if invalid_breakdown else 0.0

    return {
        "n_rows": int(n),
        "n_cols": int(df.shape[1]),
        "missing_total_ratio": missing_total_ratio,
        "missing_per_column": missing_per_col,
        "duplicate_ratio": duplicate_ratio,
        "bad_time_ratio": bad_time_ratio,
        "bad_date_ratio": bad_date_ratio,
        "invalid_ratio": invalid_ratio,
        "invalid_breakdown": invalid_breakdown,
    }


def quality_flags(metrics: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    thr = cfg["dq"]["thresholds"]
    flags = {
        "too_many_missing": metrics["missing_total_ratio"] > thr["max_missing_total"],
        "too_many_duplicates": metrics["duplicate_ratio"] > thr["max_duplicate_ratio"],
        "too_many_bad_dates": metrics["bad_date_ratio"] > thr["max_bad_date_ratio"],
        "too_many_invalid": metrics["invalid_ratio"] > thr["max_invalid_ratio"],
    }
    flags["any_issue"] = any(flags.values())
    return flags


# ---------- cleaning ----------
def clean_batch(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    df = df.copy()
    cln = cfg["dq"]["cleaning"]
    validity = cfg["dq"]["validity"]
    time_col = cfg["dq"]["parsing"]["time_col"]

    if cln.get("drop_duplicates", True):
        df = df.drop_duplicates()

    # drop bad date order
    if cln.get("drop_bad_date_order", True) and "INSR_BEGIN" in df.columns and "INSR_END" in df.columns:
        df = df[~(df["INSR_BEGIN"].notna() & df["INSR_END"].notna() & (df["INSR_BEGIN"] > df["INSR_END"]))]

    # drop missing time
    if cln.get("drop_missing_time", True) and time_col in df.columns:
        df = df.dropna(subset=[time_col])

    # out-of-range -> NaN
    if cln.get("out_of_range_to_nan", True):
        if "PROD_YEAR" in df.columns:
            mn, mx = validity["prod_year_min"], validity["prod_year_max"]
            m = df["PROD_YEAR"].notna() & ((df["PROD_YEAR"] < mn) | (df["PROD_YEAR"] > mx))
            df.loc[m, "PROD_YEAR"] = np.nan

        if "SEATS_NUM" in df.columns:
            mn, mx = validity["seats_min"], validity["seats_max"]
            m = df["SEATS_NUM"].notna() & ((df["SEATS_NUM"] < mn) | (df["SEATS_NUM"] > mx))
            df.loc[m, "SEATS_NUM"] = np.nan

        # неизвестные коды можно оставить как NaN (а потом импутировать для моделей)
        # но для analyzed_data лучше не трогать — решите политикой:
        # df.loc[unknown_mask, "INSR_TYPE"] = np.nan

    # negative -> NaN
    if cln.get("negative_to_nan", True):
        for col in validity.get("non_negative_cols", []):
            if col in df.columns:
                df.loc[df[col] < 0, col] = np.nan

    # impute numeric
    imp_num = cln.get("impute_numeric", "median")
    if imp_num and imp_num.lower() != "none":
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        for c in num_cols:
            if df[c].isna().any():
                fill = df[c].median() if imp_num == "median" else df[c].mean()
                df[c] = df[c].fillna(fill)

    # impute categorical
    imp_cat = cln.get("impute_categorical", "Unknown")
    if imp_cat and str(imp_cat).lower() != "none":
        cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
        for c in cat_cols:
            df[c] = df[c].astype("string").fillna(imp_cat)

    return df


# ---------- Apriori ----------
def binary_transactions(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    apr = cfg["dq"]["apriori"]
    out = pd.DataFrame(index=df.index)

    # numeric conditions (устойчивые, не зависят от доменной расшифровки)
    if "PREMIUM" in df.columns:
        out["premium_gt_0"] = (df["PREMIUM"] > 0).astype(int)
        out["premium_high"] = (df["PREMIUM"] >= df["PREMIUM"].quantile(0.75)).astype(int)

    if "INSURED_VALUE" in df.columns:
        out["insured_value_high"] = (df["INSURED_VALUE"] >= df["INSURED_VALUE"].quantile(0.75)).astype(int)

    if "SEATS_NUM" in df.columns:
        out["seats_ge_5"] = (df["SEATS_NUM"] >= 5).astype(int)

    # date-derived
    if "INSR_BEGIN" in df.columns and "INSR_END" in df.columns and pd.api.types.is_datetime64_any_dtype(df["INSR_BEGIN"]):
        dur = (df["INSR_END"] - df["INSR_BEGIN"]).dt.days
        out["duration_ge_365"] = (dur >= 365).fillna(False).astype(int)

    # categorical (берем top-K значений)
    top_k = int(apr.get("top_k_cat_values", 8))
    cat_cols = [c for c in ["INSR_TYPE", "TYPE_VEHICLE", "USAGE", "MAKE", "SEX"] if c in df.columns]
    for col in cat_cols:
        vc = df[col].astype("string").value_counts().head(top_k).index
        for v in vc:
            out[f"{col}={v}"] = (df[col].astype("string") == v).astype(int)

    # include target as item (для правил вида "условия -> claim_paid_pos")
    if apr.get("include_target", True) and "CLAIM_PAID" in df.columns:
        out["claim_paid_pos"] = (df["CLAIM_PAID"] > 0).astype(int)

    # remove constants
    nun = out.nunique(dropna=False)
    out = out.loc[:, nun > 1]
    return out

def save_consistency_report(batch_id: str, report: Dict[str, Any], cfg: Dict[str, Any]) -> Path:
    report_dir = Path(cfg["dq"]["consistency"].get("report_dir", "artifacts/rules"))
    report_dir.mkdir(parents=True, exist_ok=True)
    out_path = report_dir / f"consistency_{batch_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return out_path


def mine_rules(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    apr = cfg["dq"]["apriori"]
    if not apr.get("enabled", True):
        return pd.DataFrame()

    bin_df = binary_transactions(df, cfg)
    if bin_df.shape[1] == 0:
        return pd.DataFrame()

    freq = apriori(bin_df.astype(bool), min_support=float(apr["min_support"]), use_colnames=True)
    if freq.empty:
        return pd.DataFrame()

    rules = association_rules(freq, metric="confidence", min_threshold=float(apr["min_confidence"]))
    if rules.empty:
        return pd.DataFrame()

    rules = rules[rules["lift"] >= float(apr["min_lift"])].copy()

    rules["antecedents_str"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(list(s))))
    rules["consequents_str"] = rules["consequents"].apply(lambda s: ", ".join(sorted(list(s))))

    rules = rules.sort_values(["lift", "confidence", "support"], ascending=False)
    top_k = int(apr.get("top_k_rules", 5))
    return rules[["support", "confidence", "lift", "antecedents_str", "consequents_str"]].head(top_k)


# ---------- pipeline entrypoint ----------
def analyze_batch_file(batch_path: str | Path, config_path: str | Path) -> Dict[str, Any]:
    cfg = load_yaml(config_path)
    batch_path = Path(batch_path)

    io_cfg = cfg["dq"]["io"]
    analyzed_dir = Path(io_cfg["analyzed_dir"])
    artifacts_dir = Path(io_cfg["artifacts_dir"])
    analyzed_format = io_cfg.get("analyzed_format", "parquet").lower()

    dq_dir = artifacts_dir / "dq"
    rules_dir = artifacts_dir / "rules"
    dq_dir.mkdir(parents=True, exist_ok=True)
    rules_dir.mkdir(parents=True, exist_ok=True)
    analyzed_dir.mkdir(parents=True, exist_ok=True)

    batch_id = batch_path.stem

    # load -> parse
    raw = read_batch_csv(batch_path)
    parsed = parse_types(raw, cfg)

    ref_path = build_reference_rules_if_missing(parsed, cfg)

    cons_report = evaluate_reference_rules_on_batch(parsed, cfg)
    cons_path = save_consistency_report(batch_id, cons_report, cfg)


    # metrics + flags
    metrics_before = compute_dq_metrics(parsed, cfg)
    flags_before = quality_flags(metrics_before, cfg)

    # rules
    rules = mine_rules(parsed, cfg)

    # clean (always, per your decision)
    cleaned = clean_batch(parsed, cfg)

    metrics_after = compute_dq_metrics(cleaned, cfg)
    flags_after = quality_flags(metrics_after, cfg)

    # save artifacts
    metrics_path = dq_dir / f"{batch_id}_dq.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {"before": metrics_before, "after": metrics_after, "flags_before": flags_before, "flags_after": flags_after},
            f, ensure_ascii=False, indent=2
        )

    rules_path = rules_dir / f"{batch_id}_apriori_rules.csv"
    if rules.empty:
        pd.DataFrame(columns=["support", "confidence", "lift", "antecedents_str", "consequents_str"]).to_csv(rules_path, index=False)
    else:
        rules.to_csv(rules_path, index=False)

    # save analyzed batch
    if analyzed_format == "parquet":
        analyzed_path = analyzed_dir / f"{batch_id}.parquet"
        cleaned.to_parquet(analyzed_path, index=False)
    else:
        analyzed_path = analyzed_dir / f"{batch_id}.csv"
        # даты в csv лучше писать ISO-форматом
        out_df = cleaned.copy()
        for col in cfg["dq"]["parsing"].get("date_cols", []):
            if col in out_df.columns and pd.api.types.is_datetime64_any_dtype(out_df[col]):
                out_df[col] = out_df[col].dt.strftime("%Y-%m-%d")
        out_df.to_csv(analyzed_path, index=False)

    return {
       "batch_id": batch_id,
       "analyzed_path": str(analyzed_path),
       "dq_metrics_path": str(metrics_path),
       "rules_path": str(rules_path),
       "flags_before": flags_before,
       "flags_after": flags_after,
       "reference_rules_path": str(ref_path) if ref_path else None,
       "consistency_report_path": str(cons_path),
       "rules_any_issue": cons_report.get("any_issue", False),
    }



def main():
    res = analyze_batch_file(
        batch_path="../raw_data/motor_data11-14lats_batch0032_20260322_153227.csv",
        config_path="config.yaml"
    )

    print(res["analyzed_path"])
    print(res["dq_metrics_path"])
    print(res["rules_path"])
    print(res["consistency_report_path"])
    print(f"{res['rules_any_issue']=}")
    print("Flags after:", res["flags_after"])


if __name__ == "__main__":
    main()
