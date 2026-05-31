import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

DATA_PATH   = "similarity_results.json"
RANDOM_SEED = 42
K_DEFAULT   = 5
LOCAL_FEATURE_COLS = [
    "Bedrooms", "Bathrooms", "Acreage", "Square Footage",
    "Garage", "Parking", "Fireplace", "Waterfront",
    "Pool", "Garden", "Balcony",
    "Latitude", "Longitude",
]
MODELS = {
    "Linear Regression": (
        LinearRegression(),
        True,    # scale
    ),
    "Ridge Regression": (
        Ridge(alpha=10.0),
        True,
    ),
    "Random Forest": (
        RandomForestRegressor(
            n_estimators=200,
            max_depth=2,
            min_samples_leaf=1,
            bootstrap=True,
            random_state=RANDOM_SEED,
        ),
        False,
    ),
    "XGBoost": (
        XGBRegressor(
            n_estimators=200,
            max_depth=2,
            learning_rate=0.4,
            subsample=1.0,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=RANDOM_SEED,
            verbosity=0,
        ),
        False,
    ),
}

MODEL_COLORS = {
    "Linear Regression": "#2563EB", 
    "Ridge Regression":  "#16A34A",
    "Random Forest":     "#DC2626",
    "XGBoost":           "#D97706",
}

with open(DATA_PATH, "r") as fh:
    raw_data = json.load(fh)

records  = raw_data["results"]
metadata = raw_data.get("metadata", {})

K = int(metadata.get("k", K_DEFAULT))

if metadata:
    print(f"  Source file    : {metadata.get('source_file', 'unknown')}")
    print(f"  Generated at   : {metadata.get('generated_at', 'unknown')}")
    print(f"  Total props    : {metadata.get('total_properties', 'unknown')}")
    weights = metadata.get("weights", {})
    print(f"  KNN weights    : w_loc={weights.get('w_loc')}  w_feat={weights.get('w_feat')}")
    print(f"  K (neighbors)  : {K}")

p = len(LOCAL_FEATURE_COLS)
print(f"  Records loaded : {len(records):,}")
print(f"  Features (p)   : {p}   ← {'UNDERDETERMINED' if K < p else 'over/determined'} "
      f"(K={K} samples, p={p} features)")

all_idx               = list(range(len(records)))
train_idx, temp_idx   = train_test_split(all_idx, test_size=0.20, random_state=RANDOM_SEED)
val_idx,   test_idx   = train_test_split(temp_idx, test_size=0.50, random_state=RANDOM_SEED)

print(f"  Train : {len(train_idx):>4}  ({len(train_idx)/len(all_idx)*100:.1f}%)  "
      f"← set aside (no global model to train)")
print(f"  Val   : {len(val_idx):>4}  ({len(val_idx)/len(all_idx)*100:.1f}%)  ← model comparison")
print(f"  Test  : {len(test_idx):>4}  ({len(test_idx)/len(all_idx)*100:.1f}%)  ← final unbiased estimate")

def _safe_float(val):
    try:
        return float(val) if val is not None else 0.0
    except (TypeError, ValueError):
        return 0.0

def build_local_arrays(record, k):
    query     = record["query"]
    neighbors = record.get("similar_properties", [])[:k]

    def feature_row(d):
        return [_safe_float(d.get(col)) for col in LOCAL_FEATURE_COLS]

    X_local = np.array([feature_row(nbr) for nbr in neighbors], dtype=np.float64)
    y_local = np.array([_safe_float(nbr.get("Price")) for nbr in neighbors], dtype=np.float64)
    X_query = np.array([feature_row(query)], dtype=np.float64)

    return X_local, y_local, X_query


def local_predict(record, model_template, scale, k):
    try:
        X_local, y_local, X_query = build_local_arrays(record, k)
    except ValueError:
        return None

    if scale:
        scaler  = StandardScaler()
        X_local = scaler.fit_transform(X_local)   # fit & transform neighbors
        X_query = scaler.transform(X_query)        # transform query with same params

    try:
        local_model = clone(model_template)
        local_model.fit(X_local, y_local)
        return float(local_model.predict(X_query)[0])
    except Exception:
        return None


def evaluate_split(
    indices: List[int],
    split_label: str,
) -> Dict[str, List[Tuple[float, float]]]:
    """
    For every query in `indices`, run all four local models independently.

    Returns
    -------
    { model_name: [(actual_price, predicted_price), ...] }
    """
    results: Dict[str, List] = {name: [] for name in MODELS}
    fail_counts              = {name: 0  for name in MODELS}

    for qi in indices:
        record = records[qi]
        actual = _safe_float(record["query"].get("Price"))

        for name, (tmpl, do_scale) in MODELS.items():
            pred = local_predict(record, tmpl, do_scale, K)
            if pred is not None:
                results[name].append((actual, pred))
            else:
                fail_counts[name] += 1

    total_fails = sum(fail_counts.values())
    if total_fails:
        failed_str = ", ".join(f"{n}:{c}" for n, c in fail_counts.items() if c)
        print(f"    [{split_label}] Skipped predictions — {failed_str}")

    return results


def compute_metrics(
    preds_dict: Dict[str, List[Tuple[float, float]]],
    split_label: str,
) -> pd.DataFrame:
    """Compute MAE / RMSE / R² / MAPE from (actual, predicted) pair lists."""
    rows = []
    for name, pairs in preds_dict.items():
        if not pairs:
            continue
        actuals = np.array([p[0] for p in pairs], dtype=np.float64)
        preds   = np.array([p[1] for p in pairs], dtype=np.float64)

        mae  = mean_absolute_error(actuals, preds)
        rmse = np.sqrt(mean_squared_error(actuals, preds))
        r2   = r2_score(actuals, preds)
        mask = actuals != 0
        mape = (np.abs((actuals[mask] - preds[mask]) / actuals[mask])).mean() * 100

        rows.append({
            "Model": name, "Split": split_label, "N": len(pairs),
            "MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape,
        })
    return pd.DataFrame(rows)

print("\nSTEP 3 — Local regression evaluation")
print(f"  Fitting one fresh model per query per model type.")
print(f"  Total model fits: {(len(val_idx) + len(test_idx)) * len(MODELS):,}")

val_preds  = evaluate_split(val_idx,  "Validation")
test_preds = evaluate_split(test_idx, "Test")

val_metrics  = compute_metrics(val_preds,  "Validation")
test_metrics = compute_metrics(test_preds, "Test")
all_metrics  = pd.concat([val_metrics, test_metrics], ignore_index=True)

print("\nSTEP 4 — Results")

for split in ["Validation", "Test"]:
    sub = all_metrics[all_metrics["Split"] == split].sort_values("RMSE")
    n   = int(sub["N"].iloc[0]) if not sub.empty else 0
    print(f"\n  {'─' * 74}")
    print(f"  {split} set  (N = {n} queries)")
    print(f"  {'─' * 74}")
    print(f"  {'Model':<22} {'MAE':>14} {'RMSE':>14} {'R²':>9} {'MAPE':>10}")
    print(f"  {'─' * 74}")
    for _, row in sub.iterrows():
        r2_str = f"{row['R2']:.4f}" if row["R2"] > -9999 else "  n/a "
        print(f"  {row['Model']:<22} ${row['MAE']:>12,.0f} "
              f"${row['RMSE']:>12,.0f} {r2_str:>9} {row['MAPE']:>9.2f}%")

print("\nSTEP 5 — Generating plots")

# ── Figure 1: 2 × 4 Actual-vs-Predicted scatter grid ────────────────────────
n_models = len(MODELS)
fig1, axes1 = plt.subplots(2, n_models, figsize=(5.5 * n_models, 10))
fig1.suptitle(
    "KNN-Based Local Regression — Actual vs Predicted",
    fontsize=15, fontweight="bold",
)

splits_info = [("Validation", val_preds), ("Test", test_preds)]

for row_i, (split_label, preds_dict) in enumerate(splits_info):
    for col_i, name in enumerate(MODELS):
        ax    = axes1[row_i][col_i]
        pairs = preds_dict.get(name, [])
        color = MODEL_COLORS[name]

        if not pairs:
            ax.set_visible(False)
            continue

        actuals = np.array([p[0] for p in pairs])
        preds_a = np.array([p[1] for p in pairs])

        # Axis limits with breathing room
        lo = min(actuals.min(), preds_a.min()) * 0.95
        hi = max(actuals.max(), preds_a.max()) * 1.05

        ax.scatter(actuals, preds_a, alpha=0.5, s=20, color=color, edgecolors="none")
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.1, label="Perfect prediction")
        ax.set_xlim([lo, hi])
        ax.set_ylim([lo, hi])

        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"${v/1e6:.1f}M")
        )
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"${v/1e6:.1f}M")
        )
        ax.set_xlabel("Actual Price", fontsize=9)
        ax.set_ylabel("Predicted Price", fontsize=9)
        ax.set_title(f"{name}\n({split_label})", fontsize=10,
                     fontweight="bold", color=color)

        mae  = mean_absolute_error(actuals, preds_a)
        rmse = np.sqrt(mean_squared_error(actuals, preds_a))
        r2   = r2_score(actuals, preds_a)
        mask = actuals != 0
        mape = (np.abs((actuals[mask] - preds_a[mask]) / actuals[mask])).mean() * 100

        ax.text(
            0.04, 0.96,
            f"MAE  ${mae:,.0f}\nRMSE ${rmse:,.0f}\nR²   {r2:.4f}\nMAPE {mape:.2f}%",
            transform=ax.transAxes, va="top", ha="left", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.88),
        )
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)

fig1.tight_layout()
plot1_path = Path(DATA_PATH).stem + "_local_scatter.png"
fig1.savefig(plot1_path, dpi=150, bbox_inches="tight")
print(f"  Scatter grid saved  → {plot1_path}")

# ── Figure 2: RMSE + MAPE grouped bar comparison ────────────────────────────
fig2, (ax_rmse, ax_mape) = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle(
    "KNN Local Regression — Model Comparison (Val vs Test)",
    fontsize=13, fontweight="bold",
)

model_names = list(MODELS.keys())
x           = np.arange(len(model_names))
bar_w       = 0.35

# Index by model for safe lookup
val_row  = all_metrics[all_metrics["Split"] == "Validation"].set_index("Model")
test_row = all_metrics[all_metrics["Split"] == "Test"].set_index("Model")

for metric, ax, ylabel, fmt_fn in [
    ("RMSE", ax_rmse, "RMSE ($)", lambda v: f"${v/1e3:.0f}k"),
    ("MAPE", ax_mape, "MAPE (%)", lambda v: f"{v:.1f}%"),
]:
    val_vals  = [val_row.loc[m, metric]  for m in model_names]
    test_vals = [test_row.loc[m, metric] for m in model_names]
    colors    = [MODEL_COLORS[m] for m in model_names]

    bars_v = ax.bar(x - bar_w / 2, val_vals,  bar_w,
                    color=colors, alpha=0.55, edgecolor="white", label="Validation")
    bars_t = ax.bar(x + bar_w / 2, test_vals, bar_w,
                    color=colors, alpha=1.00, edgecolor="white", label="Test")

    # Value labels on top of each bar
    for bar, v in zip(bars_v, val_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01, fmt_fn(v),
                ha="center", va="bottom", fontsize=8, color="dimgray")
    for bar, v in zip(bars_t, test_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01, fmt_fn(v),
                ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=12, ha="right", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(ylabel, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

fig2.tight_layout()
plot2_path = Path(DATA_PATH).stem + "_local_comparison.png"
fig2.savefig(plot2_path, dpi=150, bbox_inches="tight")
print(f"  Comparison bar chart → {plot2_path}")

plt.show()

print("\n" + "=" * 65)
print("FINAL SUMMARY")
print("=" * 65)
fmt = all_metrics.copy()
fmt["MAE"]  = fmt["MAE"].map("${:,.0f}".format)
fmt["RMSE"] = fmt["RMSE"].map("${:,.0f}".format)
fmt["R2"]   = fmt["R2"].map("{:.4f}".format)
fmt["MAPE"] = fmt["MAPE"].map("{:.2f}%".format)
print(fmt[["Model", "Split", "N", "MAE", "RMSE", "R2", "MAPE"]].to_string(index=False))
