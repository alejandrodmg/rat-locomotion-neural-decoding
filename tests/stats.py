#!/usr/bin/env python3

import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import shapiro, friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests

def significance_code(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "na"
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "ns"

def summarize_shapiro(df):
    # Per-column Shapiro (excluding session_id)
    return {col: shapiro(df[col]) for col in df.columns[1:]}

def print_medians(title, df):
    print(f"\n=== Median performance for {title} ===")
    for col in df.columns[1:]:
        print(f"  {col:20s} median = {df[col].median():.3f}")

def print_shapiro(title, res, alpha=0.05):
    print(f"\n=== Shapiro–Wilk normality tests for {title} ===")
    for model, (W, p) in res.items():
        flag = "NON-NORMAL" if p < alpha else "normal-ish"
        print(f"  {model:20s} W = {W:6.3f}, p = {p:8.5f}  -> {flag}")

def run_friedman(df):
    models = list(df.columns[1:])
    if len(models) < 2:
        return np.nan, np.nan, np.nan
    stat, p = friedmanchisquare(*[df[m] for m in models])
    return stat, p, len(models) - 1

def print_friedman(title, stat, p, df_value, alpha=0.05):
    print(f"\n=== Friedman test for {title} ===")
    if np.isnan(stat):
        print("  Not enough models to run Friedman test.")
        return
    print(f"  DF = {df_value}, χ² = {stat:.3f}, p = {p:.5f}  ({significance_code(p)})")
    print("  -> Significant overall difference between models." if p < alpha
          else "  -> No significant overall difference between models.")

def run_pairwise_wilcoxon(df, alpha=0.05):
    models = list(df.columns[1:])
    if len(models) < 2:
        return pd.DataFrame()

    rows = []
    raw = []
    for a, b in combinations(models, 2):
        W, p = wilcoxon(df[a], df[b])
        raw.append((a, b, W, p))

    reject, p_corr, _, _ = multipletests([r[3] for r in raw], alpha=alpha, method="bonferroni")

    for (a, b, W, p), pc, rj in zip(raw, p_corr, reject):
        rows.append(
            {
                "Decoder A": a,
                "Decoder B": b,
                "Wilcoxon_W": W,
                "p_raw": p,
                "p_corr": pc,
                "Significance": significance_code(pc),
                "Reject_H0": bool(rj),
            }
        )
    return pd.DataFrame(rows)

def print_pairwise(title, df_pair, alpha=0.05):
    print(f"\n=== Pairwise Wilcoxon signed-rank tests for {title} (Bonferroni, α={alpha}) ===")
    if df_pair is None or df_pair.empty:
        print("  (Not enough models to compare.)")
        return
    for _, r in df_pair.iterrows():
        print(
            f"  {r['Decoder A']:15s} vs {r['Decoder B']:15s} "
            f"W = {r['Wilcoxon_W']:7.3f}, "
            f"p_raw = {r['p_raw']:.5f}, "
            f"p_corr = {r['p_corr']:.5f}, "
            f"sig = {r['Significance']}"
        )

def run_statistical_analysis(correlation_df, r2_df, alpha=0.05):
    print("============================================================")
    print("Statistical analysis of decoder performance")
    print(f"Number of sessions: {len(correlation_df)}")
    print(f"Number of models (correlation): {correlation_df.shape[1] - 1}")
    print(f"Number of models (R²):         {r2_df.shape[1] - 1}")
    print("============================================================")

    # Medians
    print_medians("correlation (r)", correlation_df)
    print_medians("R²", r2_df)

    # Shapiro
    print_shapiro("correlation (r)", summarize_shapiro(correlation_df), alpha=alpha)
    print_shapiro("R²", summarize_shapiro(r2_df), alpha=alpha)

    # Friedman
    fc_stat, fc_p, fc_df = run_friedman(correlation_df)
    fr2_stat, fr2_p, fr2_df = run_friedman(r2_df)

    print_friedman("correlation (r)", fc_stat, fc_p, fc_df, alpha=alpha)
    print_friedman("R²", fr2_stat, fr2_p, fr2_df, alpha=alpha)

    friedman_table = pd.DataFrame(
        [
            {"Metric": "r",  "Test": "Friedman", "DF": fc_df,  "chi2": fc_stat,  "p_value": fc_p,  "Significance": significance_code(fc_p)},
            {"Metric": "R²", "Test": "Friedman", "DF": fr2_df, "chi2": fr2_stat, "p_value": fr2_p, "Significance": significance_code(fr2_p)},
        ]
    )

    # Pairwise (only if Friedman significant)
    pair_r = pd.DataFrame()
    pair_r2 = pd.DataFrame()

    if not np.isnan(fc_p) and fc_p < alpha:
        pair_r = run_pairwise_wilcoxon(correlation_df, alpha=alpha)
        pair_r.insert(0, "Metric", "r")
        print_pairwise("correlation (r)", pair_r, alpha=alpha)
    else:
        print("\nSkipping pairwise Wilcoxon for correlation (r) (Friedman not significant / not applicable).")

    if not np.isnan(fr2_p) and fr2_p < alpha:
        pair_r2 = run_pairwise_wilcoxon(r2_df, alpha=alpha)
        pair_r2.insert(0, "Metric", "R²")
        print_pairwise("R²", pair_r2, alpha=alpha)
    else:
        print("\nSkipping pairwise Wilcoxon for R² (Friedman not significant / not applicable).")

    return friedman_table, pair_r, pair_r2

def run_two_model_analysis(correlation_df, r2_df, alpha=0.05, title="", meta=None):
    meta = meta or {}

    m_corr = list(correlation_df.columns[1:])
    m_r2 = list(r2_df.columns[1:])

    m1c, m2c = m_corr[0], m_corr[1]
    m1r, m2r = m_r2[0], m_r2[1]

    print("============================================================")
    print(f"Two-model analysis of decoder performance: {title}")
    print(f"Number of sessions: {len(correlation_df)}")
    print(f"Models (correlation): {m1c}, {m2c}")
    print(f"Models (R²):          {m1r}, {m2r}")
    print("============================================================")

    print_medians("correlation (r)", correlation_df)
    print_medians("R²", r2_df)

    print_shapiro("correlation (r)", summarize_shapiro(correlation_df), alpha=alpha)
    print_shapiro("R²", summarize_shapiro(r2_df), alpha=alpha)

    Wc, pc = wilcoxon(correlation_df[m1c], correlation_df[m2c])
    Wr, pr = wilcoxon(r2_df[m1r], r2_df[m2r])

    print("\n=== Wilcoxon signed-rank test for correlation (r) ===")
    print(f"  {m1c:20s} vs {m2c:20s} W = {Wc:7.3f}, p = {pc:.5f}, sig = {significance_code(pc)}")

    print("\n=== Wilcoxon signed-rank test for R² ===")
    print(f"  {m1r:20s} vs {m2r:20s} W = {Wr:7.3f}, p = {pr:.5f}, sig = {significance_code(pr)}")

    return pd.DataFrame(
        [
            {**meta, "Metric": "r",  "Model_1": m1c, "Model_2": m2c, "Wilcoxon_W": Wc, "p_value": pc, "Significance": significance_code(pc)},
            {**meta, "Metric": "R²", "Model_1": m1r, "Model_2": m2r, "Wilcoxon_W": Wr, "p_value": pr, "Significance": significance_code(pr)},
        ]
    )

def insert_reference_column(wide_df, ref_map, new_col_name):
    # Insert as first column after session_id
    wide_df.insert(1, new_col_name, wide_df["session_id"].map(ref_map))
    return wide_df

if __name__ == "__main__":

    RESULTS_PATH = "results"

    base_analyses = [
        ("Localized",         f"{RESULTS_PATH}/localized_correlation.csv",
                              f"{RESULTS_PATH}/localized_r2_score.csv"),
        ("Transfer learning", f"{RESULTS_PATH}/transfer_learning_correlation.csv",
                              f"{RESULTS_PATH}/transfer_learning_r2_score.csv"),
        ("Model selection",   f"{RESULTS_PATH}/model_selection_correlation.csv",
                              f"{RESULTS_PATH}/model_selection_r2_score.csv"),
        ("Frequency bands",   f"{RESULTS_PATH}/frequency_bands_correlation.csv",
                              f"{RESULTS_PATH}/frequency_bands_r2_score.csv"),
    ]

    # Pull RNN (benchmark) from it and inject into localized/frequency bands
    ms_corr = pd.read_csv(f"{RESULTS_PATH}/model_selection_correlation.csv")
    ms_r2   = pd.read_csv(f"{RESULTS_PATH}/model_selection_r2_score.csv")

    ms_corr_rnn_map = dict(zip(ms_corr["session_id"], ms_corr["RNN"]))
    ms_r2_rnn_map   = dict(zip(ms_r2["session_id"],   ms_r2["RNN"]))

    # Predictions
    directions = ["forward", "reversed"]
    delays = [0, 10, 20, 50, 100]
    prediction_results = []

    output_excel = "decoder_statistical_tests.xlsx"
    with pd.ExcelWriter(output_excel) as writer:

        # 1) Wide datasets (multi-model analyses)
        for title, corr_path, r2_path in base_analyses:
            print("\n" + "=" * 80)
            print(f"Running analysis: {title}")
            print("=" * 80)

            correlation_df = pd.read_csv(corr_path)
            r2_df = pd.read_csv(r2_path)

            # Inject the extra comparison columns (benchmark)
            if title == "Localized":
                correlation_df = insert_reference_column(correlation_df, ms_corr_rnn_map, "AllRegions")
                r2_df          = insert_reference_column(r2_df,          ms_r2_rnn_map,   "AllRegions")

            if title == "Frequency bands":
                correlation_df = insert_reference_column(correlation_df, ms_corr_rnn_map, "AllFrequencies")
                r2_df          = insert_reference_column(r2_df,          ms_r2_rnn_map,   "AllFrequencies")

            friedman_table, pairwise_r_df, pairwise_r2_df = run_statistical_analysis(
                correlation_df, r2_df, alpha=0.05
            )

            # Write to Excel (stack tables in one sheet)
            startrow = 0
            friedman_table.to_excel(writer, sheet_name=title, index=False, startrow=startrow)
            startrow += len(friedman_table) + 2

            if pairwise_r_df is not None and not pairwise_r_df.empty:
                pairwise_r_df.to_excel(writer, sheet_name=title, index=False, startrow=startrow)
                startrow += len(pairwise_r_df) + 2

            if pairwise_r2_df is not None and not pairwise_r2_df.empty:
                pairwise_r2_df.to_excel(writer, sheet_name=title, index=False, startrow=startrow)

        # 2) Predictions analyses (two-model Wilcoxon) per delay & direction
        for direction in directions:
            for delay in delays:
                corr_path = f"{RESULTS_PATH}/predictions_{direction}_{delay}_correlation.csv"
                r2_path   = f"{RESULTS_PATH}/predictions_{direction}_{delay}_r2_score.csv"

                print("\n" + "=" * 80)
                print(f"Running analysis: Predictions ({direction}, delay={delay})")
                print("=" * 80)

                correlation_df = pd.read_csv(corr_path)
                r2_df = pd.read_csv(r2_path)

                res_df = run_two_model_analysis(
                    correlation_df,
                    r2_df,
                    alpha=0.05,
                    title=f"Predictions ({direction}, delay={delay})",
                    meta={"Direction": direction, "Delay": delay},
                )
                prediction_results.append(res_df)

        if prediction_results:
            predictions_table = pd.concat(prediction_results, ignore_index=True)
            predictions_table.to_excel(writer, sheet_name="Predictions", index=False)

    print(f"\nExcel file with statistical tables saved to: {output_excel}")