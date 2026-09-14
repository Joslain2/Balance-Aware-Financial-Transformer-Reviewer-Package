#!/usr/bin/env python3
"""Create an auditable, training-ready BAFT transaction dataset.

The raw archive is never modified. The script validates schema and chronology,
resolves same-timestamp ordering when a complete accounting chain can be found
(without proving uniqueness), applies the study's accounting convention, and
labels records as Gold, Silver, or Excluded.

Primary accounting convention (validated on the training-period data):
    credit: post_balance = previous_balance + amount
    debit:  post_balance = previous_balance - amount - fee

Negative balances are not silently removed. They are retained as a Silver
overdraft regime when other structural validity checks are satisfied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd


DATE_FORMAT = "%d-%b-%y %I.%M.%S.%f %p"
SEQ_LEN = 100
STRIDE = 20
TRAIN_END = pd.Timestamp("2026-01-05 21:48:42")
VALIDATION_END = pd.Timestamp("2026-02-02 10:17:32")
MIN_TRAIN_SERVICE_COUNT = 4200


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def money_to_cents(series: pd.Series) -> pd.Series:
    """Convert currency values to nullable integer cents."""
    numeric = pd.to_numeric(series, errors="coerce")
    return pd.Series(np.rint(numeric * 100), index=series.index).astype("Int64")


def split_label(timestamp: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [timestamp <= TRAIN_END, timestamp <= VALIDATION_END],
            ["train", "validation"],
            default="test",
        ),
        index=timestamp.index,
        dtype="string",
    )


def euler_event_order(group: pd.DataFrame, starting_balance_cents: int | None) -> list[int] | None:
    """Find an accounting-contiguous event order for one timestamp group.

    Each transaction is an edge from (post balance - transaction delta) to its
    observed post balance. A valid order is an Euler trail through all edges.
    """
    edges: list[tuple[int, int, int]] = []
    for idx, row in group.iterrows():
        if pd.isna(row["BALANCE_CENTS"]) or pd.isna(row["DELTA_CENTS"]):
            return None
        end = int(row["BALANCE_CENTS"])
        start = end - int(row["DELTA_CENTS"])
        edges.append((int(idx), start, end))

    def trail(candidate_start: int) -> list[int] | None:
        adjacency: dict[int, deque[tuple[int, int]]] = defaultdict(deque)
        for event_idx, start, end in edges:
            adjacency[start].append((event_idx, end))
        vertex_stack = [candidate_start]
        event_stack: list[int] = []
        reverse_events: list[int] = []
        while vertex_stack:
            current = vertex_stack[-1]
            if adjacency[current]:
                event_idx, end = adjacency[current].popleft()
                vertex_stack.append(end)
                event_stack.append(event_idx)
            else:
                vertex_stack.pop()
                if event_stack:
                    reverse_events.append(event_stack.pop())
        order = list(reversed(reverse_events))
        if len(order) != len(edges):
            return None
        # Verify the order explicitly because event labels, not just vertices,
        # are required downstream.
        balance = candidate_start
        edge_lookup = {event_idx: (start, end) for event_idx, start, end in edges}
        for event_idx in order:
            start, end = edge_lookup[event_idx]
            if start != balance:
                return None
            balance = end
        return order

    if starting_balance_cents is not None:
        return trail(starting_balance_cents)

    # At the first timestamp for a user, the opening balance is unobserved.
    # Try candidate starts in raw-row order and accept the first complete chain.
    for _, start, _ in edges:
        order = trail(start)
        if order is not None:
            return order
    return None


def reorder_same_timestamp_groups(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    tie_status = pd.Series("NONE", index=df.index, dtype="string")
    final_order = np.arange(len(df), dtype=np.int64)
    stats = {
        "same_timestamp_groups": 0,
        "same_timestamp_rows": 0,
        "resolved_without_reorder_groups": 0,
        "resolved_with_reorder_groups": 0,
        "unresolved_groups": 0,
    }

    tied = df.duplicated(["USER_ID", "COMPLETION_TIME"], keep=False) & df["COMPLETION_TIME"].notna()
    tied_rows = df.loc[tied]
    prior_tie_user: str | None = None
    prior_tie_last_position: int | None = None
    prior_tie_final_balance: int | None = None
    for _, group in tied_rows.groupby(["USER_ID", "COMPLETION_TIME"], sort=False, dropna=False):
        original = group.index.to_list()
        first_position = original[0]
        previous_balance: int | None = None
        current_user = str(df.at[first_position, "USER_ID"])
        if (
            prior_tie_user == current_user
            and prior_tie_last_position is not None
            and prior_tie_last_position == first_position - 1
        ):
            previous_balance = prior_tie_final_balance
        elif first_position > 0 and df.at[first_position - 1, "USER_ID"] == df.at[first_position, "USER_ID"]:
            value = df.at[first_position - 1, "BALANCE_CENTS"]
            previous_balance = None if pd.isna(value) else int(value)
        stats["same_timestamp_groups"] += 1
        stats["same_timestamp_rows"] += len(group)
        proposed = euler_event_order(group, previous_balance)
        if proposed is None:
            tie_status.loc[original] = "UNRESOLVED"
            stats["unresolved_groups"] += 1
        elif proposed == original:
            tie_status.loc[original] = "RESOLVED_ORIGINAL_ORDER"
            stats["resolved_without_reorder_groups"] += 1
        else:
            final_order[np.asarray(original, dtype=np.int64)] = np.asarray(proposed, dtype=np.int64)
            tie_status.loc[original] = "RESOLVED_REORDERED"
            stats["resolved_with_reorder_groups"] += 1
        selected = original if proposed is None else proposed
        final_balance = df.at[selected[-1], "BALANCE_CENTS"]
        prior_tie_user = current_user
        prior_tie_last_position = original[-1]
        prior_tie_final_balance = None if pd.isna(final_balance) else int(final_balance)

    result = df.loc[final_order].copy().reset_index(drop=True)
    result["TIE_STATUS"] = tie_status.loc[final_order].to_numpy()
    return result, stats


def load_and_standardize(input_zip: Path) -> pd.DataFrame:
    df = pd.read_csv(input_zip, compression="zip", encoding="utf-8-sig", low_memory=False)
    expected = ["USER_ID", "COMPLETION_TIME", "TAB", "SERVICE", "AMOUNT", "MMT_FEE", "BALANCE"]
    missing_columns = sorted(set(expected) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    df = df[expected].copy()
    df.insert(0, "RAW_ROW_ID", np.arange(1, len(df) + 1, dtype=np.int64))
    df["USER_ID"] = df["USER_ID"].astype("string").str.strip()
    df["COMPLETION_TIME_RAW"] = df["COMPLETION_TIME"].astype("string")
    df["COMPLETION_TIME"] = pd.to_datetime(df["COMPLETION_TIME"], format=DATE_FORMAT, errors="coerce")
    df["TAB"] = df["TAB"].astype("string").str.strip().str.lower()
    df["SERVICE_RAW"] = df["SERVICE"].astype("string")
    df["SERVICE"] = df["SERVICE_RAW"].str.replace(r"\s+", " ", regex=True).str.strip()
    for col in ["AMOUNT", "MMT_FEE", "BALANCE"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["AMOUNT_CENTS"] = money_to_cents(df["AMOUNT"])
    df["FEE_CENTS"] = money_to_cents(df["MMT_FEE"])
    df["BALANCE_CENTS"] = money_to_cents(df["BALANCE"])
    df["DELTA_CENTS"] = pd.Series(
        np.select(
            [df["TAB"].eq("credit"), df["TAB"].eq("debit")],
            [df["AMOUNT_CENTS"].fillna(0), -(df["AMOUNT_CENTS"].fillna(0) + df["FEE_CENTS"].fillna(0))],
            default=0,
        ),
        index=df.index,
    ).astype("Int64")
    return df.sort_values(["USER_ID", "COMPLETION_TIME", "RAW_ROW_ID"], kind="mergesort").reset_index(drop=True)


def quality_flags(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    exact_dup_extra = df.duplicated(
        ["USER_ID", "COMPLETION_TIME", "TAB", "SERVICE", "AMOUNT_CENTS", "FEE_CENTS", "BALANCE_CENTS"],
        keep="first",
    )
    invalid_user = df["USER_ID"].isna() | df["USER_ID"].eq("")
    invalid_time = df["COMPLETION_TIME"].isna()
    invalid_direction = ~df["TAB"].isin(["credit", "debit"])
    invalid_numeric = df[["AMOUNT_CENTS", "FEE_CENTS", "BALANCE_CENTS"]].isna().any(axis=1)
    negative_amount_or_fee = df["AMOUNT_CENTS"].lt(0).fillna(False) | df["FEE_CENTS"].lt(0).fillna(False)
    hard_excluded = invalid_user | invalid_time | invalid_direction | invalid_numeric | negative_amount_or_fee | exact_dup_extra

    flag_columns = {
        "INVALID_USER": invalid_user,
        "INVALID_TIMESTAMP": invalid_time,
        "INVALID_DIRECTION": invalid_direction,
        "INVALID_NUMERIC": invalid_numeric,
        "NEGATIVE_AMOUNT_OR_FEE": negative_amount_or_fee,
        "EXACT_DUPLICATE_EXTRA": exact_dup_extra,
        "FIRST_EVENT_UNVERIFIABLE": df["IS_FIRST_EVENT"],
        "MISSING_SERVICE": ~df["HAS_SERVICE_LABEL"],
        "ZERO_AMOUNT": ~df["POSITIVE_AMOUNT"],
        "ACCOUNTING_EXCEPTION": (~df["IS_FIRST_EVENT"]) & (~df["ACCOUNTING_OK"]),
        "NEGATIVE_PRE_BALANCE": (~df["IS_FIRST_EVENT"]) & (~df["PRE_BALANCE_NONNEG"]),
        "NEGATIVE_POST_BALANCE": ~df["POST_BALANCE_NONNEG"],
        "OBSERVED_OVERDRAFT": (~df["IS_FIRST_EVENT"]) & (~df["OBSERVED_FEASIBLE"].fillna(False)),
        "UNRESOLVED_TIME_TIE": df["TIE_STATUS"].eq("UNRESOLVED"),
    }
    labels = np.array(list(flag_columns), dtype=object)
    matrix = np.column_stack([s.to_numpy(bool) for s in flag_columns.values()])
    flags = pd.Series(["|".join(labels[row]) if row.any() else "NONE" for row in matrix], index=df.index, dtype="string")
    exclusion_reason = pd.Series(
        ["|".join(labels[:6][row[:6]]) if row[:6].any() else pd.NA for row in matrix],
        index=df.index,
        dtype="string",
    )
    return hard_excluded, flags, exclusion_reason


def build_sequence_manifest(df: pd.DataFrame) -> pd.DataFrame:
    records: list[tuple] = []
    sequence_id = 0
    for user_id, group in df.groupby("USER_ID", sort=False):
        n = len(group)
        if n < SEQ_LEN + 1:
            continue
        idx = group.index.to_numpy(np.int64)
        post_nonneg = group["POST_BALANCE_NONNEG"].to_numpy(bool)
        accounting_ok = group["ACCOUNTING_OK"].to_numpy(bool)
        unresolved = group["TIE_STATUS"].eq("UNRESOLVED").to_numpy(bool)
        missing_history = (~group["HAS_SERVICE_LABEL"]).to_numpy(bool)
        for local_start in range(0, n - SEQ_LEN, STRIDE):
            target_local = local_start + SEQ_LEN
            target_idx = int(idx[target_local])
            window = slice(local_start, target_local + 1)
            sequence_id += 1
            records.append(
                (
                    sequence_id,
                    str(user_id),
                    int(idx[local_start]),
                    int(idx[target_local - 1]),
                    target_idx,
                    df.at[target_idx, "COMPLETION_TIME"],
                    df.at[target_idx, "SPLIT"],
                    bool(df.at[target_idx, "GOLD_TARGET_ELIGIBLE"]),
                    bool(post_nonneg[window].all() and accounting_ok[window].all() and (~unresolved[window]).all() and df.at[target_idx, "GOLD_TARGET_ELIGIBLE"]),
                    int((~post_nonneg[window]).sum()),
                    int((~accounting_ok[window]).sum()),
                    int(unresolved[window].sum()),
                    int(missing_history[local_start:target_local].sum()),
                )
            )
    return pd.DataFrame(
        records,
        columns=[
            "SEQUENCE_ID", "USER_ID", "HISTORY_START_EVENT_INDEX", "HISTORY_END_EVENT_INDEX",
            "TARGET_EVENT_INDEX", "TARGET_TIME", "SPLIT", "GOLD_TARGET", "STRICT_GOLD_WINDOW",
            "NEGATIVE_BALANCE_EVENTS_101", "ACCOUNTING_EXCEPTIONS_101", "UNRESOLVED_TIES_101",
            "MISSING_SERVICE_HISTORY_100",
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="ZIP archive containing the source CSV")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-train-service-count", type=int, default=MIN_TRAIN_SERVICE_COUNT,
                        help="Training-period input vocabulary threshold; paper default: 4200")
    args = parser.parse_args()
    if args.min_train_service_count < 1:
        parser.error("--min-train-service-count must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_and_standardize(args.input)
    original_rows = len(raw)
    raw_duplicate_extra = int(raw.duplicated(
        ["USER_ID", "COMPLETION_TIME", "TAB", "SERVICE", "AMOUNT_CENTS", "FEE_CENTS", "BALANCE_CENTS"], keep="first"
    ).sum())
    df, tie_stats = reorder_same_timestamp_groups(raw)
    df.insert(1, "EVENT_INDEX", np.arange(len(df), dtype=np.int64))
    df["PREV_BALANCE_CENTS"] = df.groupby("USER_ID", sort=False)["BALANCE_CENTS"].shift(1).astype("Int64")
    df["PREV_TIMESTAMP"] = df.groupby("USER_ID", sort=False)["COMPLETION_TIME"].shift(1)
    df["TIME_GAP_SECONDS"] = (df["COMPLETION_TIME"] - df["PREV_TIMESTAMP"]).dt.total_seconds()
    df["IS_FIRST_EVENT"] = df["PREV_BALANCE_CENTS"].isna()
    df["EXPECTED_BALANCE_CENTS"] = (df["PREV_BALANCE_CENTS"] + df["DELTA_CENTS"]).astype("Int64")
    df["ACCOUNTING_RESIDUAL_CENTS"] = (df["BALANCE_CENTS"] - df["EXPECTED_BALANCE_CENTS"]).astype("Int64")
    df["ACCOUNTING_OK"] = df["IS_FIRST_EVENT"] | df["ACCOUNTING_RESIDUAL_CENTS"].eq(0)
    df["HAS_SERVICE_LABEL"] = df["SERVICE"].notna() & df["SERVICE"].ne("")
    df["POSITIVE_AMOUNT"] = df["AMOUNT_CENTS"].gt(0).fillna(False)
    df["PRE_BALANCE_NONNEG"] = df["PREV_BALANCE_CENTS"].ge(0).fillna(False)
    df["POST_BALANCE_NONNEG"] = df["BALANCE_CENTS"].ge(0).fillna(False)
    df["OBSERVED_FEASIBLE"] = (
        df["TAB"].eq("credit")
        | (df["AMOUNT_CENTS"] + df["FEE_CENTS"] <= df["PREV_BALANCE_CENTS"])
    ).astype("boolean")
    df.loc[df["IS_FIRST_EVENT"], "OBSERVED_FEASIBLE"] = pd.NA
    df["SPLIT"] = split_label(df["COMPLETION_TIME"])

    # Define the model service vocabulary from the training period only.
    train_counts = df.loc[(df["SPLIT"] == "train") & df["HAS_SERVICE_LABEL"], "SERVICE"].value_counts()
    retained_services = sorted(train_counts[train_counts >= args.min_train_service_count].index.tolist())
    df["SERVICE_MODEL"] = np.where(
        ~df["HAS_SERVICE_LABEL"],
        "__MISSING__",
        np.where(df["SERVICE"].isin(retained_services), df["SERVICE"], "__OTHER__"),
    )

    hard_excluded, flags, exclusion_reason = quality_flags(df)
    df["QUALITY_FLAGS"] = flags
    df["EXCLUSION_REASON"] = exclusion_reason
    df["GOLD_TARGET_ELIGIBLE"] = (
        (~hard_excluded)
        & (~df["IS_FIRST_EVENT"])
        & df["HAS_SERVICE_LABEL"]
        & df["POSITIVE_AMOUNT"]
        & df["ACCOUNTING_OK"]
        & df["PRE_BALANCE_NONNEG"]
        & df["POST_BALANCE_NONNEG"]
        & df["OBSERVED_FEASIBLE"].fillna(False)
        & (~df["TIE_STATUS"].eq("UNRESOLVED"))
    )
    df["QUALITY_TIER"] = np.select(
        [hard_excluded, df["GOLD_TARGET_ELIGIBLE"]],
        ["EXCLUDED", "GOLD"],
        default="SILVER",
    )

    excluded = df.loc[hard_excluded].copy()
    cleaned = df.loc[~hard_excluded].copy().reset_index(drop=True)
    cleaned["EVENT_INDEX"] = np.arange(len(cleaned), dtype=np.int64)

    # Recompute event-index dependent sequence anchors after hard exclusions.
    manifest = build_sequence_manifest(cleaned)

    # Compact research/audit tables.
    flag_masks = {
        "MISSING_SERVICE": ~cleaned["HAS_SERVICE_LABEL"],
        "ZERO_AMOUNT": ~cleaned["POSITIVE_AMOUNT"],
        "ACCOUNTING_EXCEPTION": (~cleaned["IS_FIRST_EVENT"]) & (~cleaned["ACCOUNTING_OK"]),
        "NEGATIVE_PRE_BALANCE": (~cleaned["IS_FIRST_EVENT"]) & (~cleaned["PRE_BALANCE_NONNEG"]),
        "NEGATIVE_POST_BALANCE": ~cleaned["POST_BALANCE_NONNEG"],
        "OBSERVED_OVERDRAFT": (~cleaned["IS_FIRST_EVENT"]) & (~cleaned["OBSERVED_FEASIBLE"].fillna(False)),
        "UNRESOLVED_TIME_TIE": cleaned["TIE_STATUS"].eq("UNRESOLVED"),
    }
    flag_table = []
    for flag, mask in flag_masks.items():
        flag_table.append({"rule_or_flag": flag, "rows": int(mask.sum()), "rate": float(mask.mean())})
    flag_table.extend([
        {"rule_or_flag": "GOLD", "rows": int(cleaned["QUALITY_TIER"].eq("GOLD").sum()), "rate": float(cleaned["QUALITY_TIER"].eq("GOLD").mean())},
        {"rule_or_flag": "SILVER", "rows": int(cleaned["QUALITY_TIER"].eq("SILVER").sum()), "rate": float(cleaned["QUALITY_TIER"].eq("SILVER").mean())},
        {"rule_or_flag": "EXCLUDED", "rows": int(len(excluded)), "rate": float(len(excluded) / original_rows)},
    ])
    flags_df = pd.DataFrame(flag_table)

    service_profile = (
        cleaned.assign(SERVICE_DISPLAY=cleaned["SERVICE"].fillna("__MISSING__"))
        .groupby(["SPLIT", "TAB", "SERVICE_DISPLAY", "SERVICE_MODEL"], dropna=False, observed=True)
        .agg(
            rows=("RAW_ROW_ID", "size"),
            users=("USER_ID", "nunique"),
            gold_rows=("GOLD_TARGET_ELIGIBLE", "sum"),
            negative_balance_rows=("POST_BALANCE_NONNEG", lambda s: int((~s).sum())),
            accounting_exception_rows=("ACCOUNTING_OK", lambda s: int((~s).sum())),
            median_amount=("AMOUNT", "median"),
        )
        .reset_index()
    )
    cohort_summary = (
        manifest.groupby("SPLIT", observed=True)
        .agg(
            all_sequences=("SEQUENCE_ID", "size"),
            gold_target_sequences=("GOLD_TARGET", "sum"),
            strict_gold_sequences=("STRICT_GOLD_WINDOW", "sum"),
            users=("USER_ID", "nunique"),
        )
        .reindex(["train", "validation", "test"])
        .reset_index()
    )
    class_distribution = (
        manifest.loc[manifest["GOLD_TARGET"]]
        .merge(cleaned[["EVENT_INDEX", "SERVICE_MODEL"]], left_on="TARGET_EVENT_INDEX", right_on="EVENT_INDEX", how="left")
        .groupby(["SPLIT", "SERVICE_MODEL"], observed=True)
        .size()
        .rename("gold_target_sequences")
        .reset_index()
    )

    exceptions = cleaned.loc[
        (~cleaned["IS_FIRST_EVENT"]) & (~cleaned["ACCOUNTING_OK"]),
        ["RAW_ROW_ID", "EVENT_INDEX", "USER_ID", "COMPLETION_TIME", "TAB", "SERVICE", "AMOUNT", "MMT_FEE",
         "PREV_BALANCE_CENTS", "BALANCE_CENTS", "EXPECTED_BALANCE_CENTS", "ACCOUNTING_RESIDUAL_CENTS", "TIE_STATUS"],
    ].copy()
    exceptions["PREV_BALANCE"] = exceptions["PREV_BALANCE_CENTS"] / 100
    exceptions["BALANCE"] = exceptions["BALANCE_CENTS"] / 100
    exceptions["EXPECTED_BALANCE"] = exceptions["EXPECTED_BALANCE_CENTS"] / 100
    exceptions["ACCOUNTING_RESIDUAL"] = exceptions["ACCOUNTING_RESIDUAL_CENTS"] / 100
    exceptions["ABS_RESIDUAL"] = exceptions["ACCOUNTING_RESIDUAL"].abs()
    exceptions = exceptions.sort_values("ABS_RESIDUAL", ascending=False)

    output_columns = [
        "RAW_ROW_ID", "EVENT_INDEX", "USER_ID", "COMPLETION_TIME", "TAB", "SERVICE", "SERVICE_MODEL",
        "AMOUNT", "MMT_FEE", "BALANCE", "PREV_TIMESTAMP", "TIME_GAP_SECONDS", "PREV_BALANCE_CENTS",
        "EXPECTED_BALANCE_CENTS", "ACCOUNTING_RESIDUAL_CENTS", "IS_FIRST_EVENT", "ACCOUNTING_OK",
        "HAS_SERVICE_LABEL", "POSITIVE_AMOUNT", "PRE_BALANCE_NONNEG", "POST_BALANCE_NONNEG",
        "OBSERVED_FEASIBLE", "TIE_STATUS", "SPLIT", "GOLD_TARGET_ELIGIBLE", "QUALITY_TIER", "QUALITY_FLAGS",
    ]
    cleaned[output_columns].to_parquet(
        args.output_dir / "mobile_money_cleaned_transactions.parquet", index=False, compression="zstd"
    )
    manifest.to_parquet(args.output_dir / "baft_sequence_manifest.parquet", index=False, compression="zstd")
    excluded.to_parquet(args.output_dir / "excluded_rows.parquet", index=False, compression="zstd")
    flags_df.to_csv(args.output_dir / "quality_rule_counts.csv", index=False)
    service_profile.to_csv(args.output_dir / "service_quality_profile.csv", index=False)
    cohort_summary.to_csv(args.output_dir / "sequence_cohort_summary.csv", index=False)
    class_distribution.to_csv(args.output_dir / "gold_sequence_class_distribution.csv", index=False)
    exceptions.to_csv(args.output_dir / "accounting_exceptions.csv", index=False)

    comparable = (~cleaned["IS_FIRST_EVENT"])
    accounting_compliance = float(cleaned.loc[comparable, "ACCOUNTING_OK"].mean())
    metadata = {
        "dataset": "mobile_money_dataset_6_month",
        "source_sha256": sha256(args.input),
        "raw_rows": int(original_rows),
        "cleaned_rows": int(len(cleaned)),
        "excluded_rows": int(len(excluded)),
        "users": int(cleaned["USER_ID"].nunique()),
        "date_min": str(cleaned["COMPLETION_TIME"].min()),
        "date_max": str(cleaned["COMPLETION_TIME"].max()),
        "raw_exact_duplicate_extra_rows": raw_duplicate_extra,
        "accounting_convention": {
            "balance_semantics": "post-transaction balance",
            "credit": "balance_t = balance_t-1 + amount_t",
            "debit": "balance_t = balance_t-1 - amount_t - fee_t",
            "calculation_precision": "integer cents",
        },
        "accounting_compliance_after_tie_resolution": accounting_compliance,
        "accounting_exception_rows_after_tie_resolution": int((comparable & ~cleaned["ACCOUNTING_OK"]).sum()),
        "negative_post_balance_rows": int((~cleaned["POST_BALANCE_NONNEG"]).sum()),
        "gold_target_rows": int(cleaned["GOLD_TARGET_ELIGIBLE"].sum()),
        "silver_rows": int(cleaned["QUALITY_TIER"].eq("SILVER").sum()),
        "tie_resolution": tie_stats,
        "sequence_protocol": {
            "history_length": SEQ_LEN,
            "stride": STRIDE,
            "train_end_inclusive": str(TRAIN_END),
            "validation_end_inclusive": str(VALIDATION_END),
            "all_sequences": int(len(manifest)),
            "gold_target_sequences": int(manifest["GOLD_TARGET"].sum()),
            "strict_gold_sequences": int(manifest["STRICT_GOLD_WINDOW"].sum()),
        },
        "service_vocabulary": {
            "selection_data": "training period only",
            "minimum_training_count": args.min_train_service_count,
            "retained_named_services": retained_services,
            "model_classes": sorted(cleaned["SERVICE_MODEL"].unique().tolist()),
            "missing_history_token": "__MISSING__",
            "rare_service_token": "__OTHER__",
            "missing service is not eligible as a Gold prediction target": True,
        },
        "quality_definition": {
            "Gold": "Known positive-amount target; exact accounting; nonnegative pre/post balance; observed debit does not exceed available balance; resolved chronology.",
            "Silver": "Structurally valid context/transaction with an overdraft, missing service, zero amount, accounting exception, first-event limitation, or unresolved time tie.",
            "Excluded": "Invalid required field, invalid direction/numeric value, negative amount/fee, or duplicate extra row.",
        },
    }
    (args.output_dir / "cleaning_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
