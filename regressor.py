import json
import math
import warnings
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


def get_metrics(y_true, y_pred):
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}


def flatten_record(record, k, query_cols, nbr_prop_cols, nbr_dist_cols):
    query     = record["query"]
    neighbors = record.get("similar_properties", [])
    if len(neighbors) < k:
        return None
    row = {"Price": float(query["Price"])}
    for col in query_cols:
        row[f"query_{col.lower().replace(' ', '_')}"] = query.get(col, np.nan)
    for i, nbr in enumerate(neighbors[:k], 1):
        pfx = f"neighbor{i}_"
        row[f"{pfx}price"] = float(nbr.get("Price", np.nan))
        for col in nbr_prop_cols:
            row[f"{pfx}{col.lower().replace(' ', '_')}"] = nbr.get(col, np.nan)
        for col in nbr_dist_cols:
            row[f"{pfx}{col}"] = nbr.get(col, np.nan)
    return row


def safe_float(val):
    try:
        return float(val) if val is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def local_predict(record, model_template, scale, k, feature_cols):
    query     = record["query"]
    neighbors = record.get("similar_properties", [])[:k]
    X_local   = np.array([[safe_float(n.get(c)) for c in feature_cols] for n in neighbors], dtype=np.float64)
    y_local   = np.array([safe_float(n.get("Price")) for n in neighbors], dtype=np.float64)
    X_query   = np.array([[safe_float(query.get(c)) for c in feature_cols]], dtype=np.float64)
    if scale:
        sc      = StandardScaler()
        X_local = sc.fit_transform(X_local)
        X_query = sc.transform(X_query)
    try:
        m = clone(model_template)
        m.fit(X_local, y_local)
        return float(m.predict(X_query)[0])
    except Exception:
        return None


def scatter_ax(ax, y_true, y_pred, metrics, title, color="#0EA5E9"):
    lo = min(y_true.min(), y_pred.min()) * 0.95
    hi = max(y_true.max(), y_pred.max()) * 1.05
    ax.scatter(y_true, y_pred, alpha=0.4, s=15, color=color, edgecolors="none")
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.2, label="Perfect")
    ax.set_xlim([lo, hi]); ax.set_ylim([lo, hi])
    ax.set_xlabel("Actual ($)", fontsize=9)
    ax.set_ylabel("Predicted ($)", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", color=color)
    fmt = mticker.FuncFormatter(lambda v, _: f"${v/1e6:.1f}M")
    ax.xaxis.set_major_formatter(fmt); ax.yaxis.set_major_formatter(fmt)
    ax.text(
        0.04, 0.96,
        f"MAE  ${metrics['MAE']:,.0f}\nRMSE ${metrics['RMSE']:,.0f}\nR²   {metrics['R2']:.4f}\nMAPE {metrics['MAPE']:.2f}%",
        transform=ax.transAxes, va="top", ha="left", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.8),
    )
    ax.legend(fontsize=7); ax.grid(linestyle="--", alpha=0.35); ax.set_axisbelow(True)


def sanitize(obj):
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        obj = float(obj)
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def main():
    with open("similarity_results.json") as f:
        raw = json.load(f)

    records  = raw["results"]
    metadata = raw.get("metadata", {})
    K        = int(metadata.get("k", 5))

    query_cols    = ["Bedrooms", "Bathrooms", "Acreage", "Square Footage", "Garage", "Parking",
                     "Fireplace", "Waterfront", "Pool", "Garden", "Balcony", "Latitude", "Longitude"]
    nbr_prop_cols = query_cols[:-2]
    nbr_dist_cols = ["location_dist_km", "location_dist_norm", "feature_dist_norm", "total_distance"]

    local_models = {
        "Linear Regression": (LinearRegression(), True),
        "Ridge Regression":  (Ridge(alpha=10.0), True),
        "Random Forest":     (RandomForestRegressor(n_estimators=200, max_depth=2,
                              min_samples_leaf=1, bootstrap=True, random_state=42), False),
        "XGBoost Local":     (XGBRegressor(n_estimators=200, max_depth=2, learning_rate=0.4,
                              subsample=1.0, colsample_bytree=0.8, objective="reg:squarederror",
                              random_state=42, verbosity=0), False),
    }
    model_colors = {
        "XGBoost Global":    "#7C3AED",
        "Linear Regression": "#2563EB",
        "Ridge Regression":  "#16A34A",
        "Random Forest":     "#DC2626",
        "XGBoost Local":     "#D97706",
    }

    # ── Global XGBoost ────────────────────────────────────────────────────────
    print("Running global XGBoost model...")
    rows, rec_indices = [], []
    for i, rec in enumerate(records):
        flat = flatten_record(rec, K, query_cols, nbr_prop_cols, nbr_dist_cols)
        if flat is not None:
            rows.append(flat)
            rec_indices.append(i)

    df         = pd.DataFrame(rows)
    feat_cols  = [c for c in df.columns if c != "Price"]
    X, y       = df[feat_cols], df["Price"]
    rec_to_row = {ri: pos for pos, ri in enumerate(rec_indices)}

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.20, random_state=42)
    X_val, X_test, y_val, y_test     = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

    gmodel = XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="reg:squarederror", random_state=42,
        early_stopping_rounds=30, eval_metric="rmse", verbosity=0,
    )
    gmodel.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    g_val_pred  = gmodel.predict(X_val)
    g_test_pred = gmodel.predict(X_test)
    g_val_m     = get_metrics(y_val.values, g_val_pred)
    g_test_m    = get_metrics(y_test.values, g_test_pred)

    feat_imp = (
        pd.DataFrame({"feature": feat_cols, "importance": gmodel.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    print(f"  Global — Val  MAE ${g_val_m['MAE']:>12,.0f}  RMSE ${g_val_m['RMSE']:>12,.0f}  R² {g_val_m['R2']:.4f}  MAPE {g_val_m['MAPE']:.2f}%")
    print(f"  Global — Test MAE ${g_test_m['MAE']:>12,.0f}  RMSE ${g_test_m['RMSE']:>12,.0f}  R² {g_test_m['R2']:.4f}  MAPE {g_test_m['MAPE']:.2f}%")

    # ── Local models ──────────────────────────────────────────────────────────
    print("\nRunning local regression models...")
    all_idx           = list(range(len(records)))
    _, temp_idx       = train_test_split(all_idx, test_size=0.20, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=42)

    local_preds   = {"Validation": {n: [] for n in local_models},
                     "Test":       {n: [] for n in local_models}}
    query_results = []

    for split, indices in [("Validation", val_idx), ("Test", test_idx)]:
        for qi in indices:
            rec    = records[qi]
            actual = safe_float(rec["query"].get("Price"))
            g_pred = float(gmodel.predict(X.iloc[[rec_to_row[qi]]])[0]) if qi in rec_to_row else None

            entry = {
                "query_index":        qi,
                "split":              split,
                "actual_price":       actual,
                "global_prediction":  g_pred,
                "local_predictions":  {},
                "query_features":     {c: rec["query"].get(c) for c in query_cols},
                "similar_properties": [
                    {
                        "price":     safe_float(n.get("Price")),
                        "features":  {c: n.get(c) for c in nbr_prop_cols},
                        "distances": {c: n.get(c) for c in nbr_dist_cols},
                    }
                    for n in rec.get("similar_properties", [])[:K]
                ],
            }

            for name, (tmpl, scale) in local_models.items():
                pred = local_predict(rec, tmpl, scale, K, query_cols)
                entry["local_predictions"][name] = pred
                if pred is not None:
                    local_preds[split][name].append((actual, pred))

            query_results.append(entry)

    local_metrics = {}
    for split in ["Validation", "Test"]:
        local_metrics[split] = {}
        for name, pairs in local_preds[split].items():
            if not pairs:
                continue
            acts, pa = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
            local_metrics[split][name] = {**get_metrics(acts, pa), "N": len(pairs)}

    print("\n  Local model results (Test set):")
    for name, m in local_metrics["Test"].items():
        print(f"  {name:<22} MAE ${m['MAE']:>12,.0f}  RMSE ${m['RMSE']:>12,.0f}  R² {m['R2']:.4f}  MAPE {m['MAPE']:.2f}%")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nSaving plots...")

    # Fig 1 — Global model: feature importances + validation scatter
    fig1, ax1 = plt.subplots(1, 2, figsize=(20, 6))
    fig1.suptitle("Global XGBoost Valuation Model", fontsize=15, fontweight="bold")
    top20 = feat_imp.head(20)
    ax1[0].barh(top20["feature"][::-1], top20["importance"][::-1],
                color=model_colors["XGBoost Global"], edgecolor="white", linewidth=0.4)
    ax1[0].set_xlabel("F-score (gain)", fontsize=10)
    ax1[0].set_title("Top 20 Feature Importances", fontsize=11, fontweight="bold")
    ax1[0].tick_params(axis="y", labelsize=8)
    ax1[0].xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax1[0].grid(axis="x", linestyle="--", alpha=0.4); ax1[0].set_axisbelow(True)
    scatter_ax(ax1[1], y_val.values, g_val_pred, g_val_m,
               "Global XGBoost — Validation", model_colors["XGBoost Global"])
    fig1.tight_layout()
    fig1.savefig("global_model_results.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)

    # Fig 2 — Local models: 2 splits × N models scatter grid
    n_m = len(local_models)
    fig2, ax2 = plt.subplots(2, n_m, figsize=(5.5 * n_m, 10))
    fig2.suptitle("KNN Local Regression — Actual vs Predicted", fontsize=15, fontweight="bold")
    for ri, (split, pd_) in enumerate([("Validation", local_preds["Validation"]),
                                        ("Test",       local_preds["Test"])]):
        for ci, name in enumerate(local_models):
            ax, pairs = ax2[ri][ci], pd_[name]
            if not pairs:
                ax.set_visible(False); continue
            acts, pa = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
            scatter_ax(ax, acts, pa, get_metrics(acts, pa),
                       f"{name}\n({split})", model_colors[name])
    fig2.tight_layout()
    fig2.savefig("local_model_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # Fig 3 — Local models: val vs test bar comparison
    lm_names = list(local_models.keys())
    fig3, (ar3, am3) = plt.subplots(1, 2, figsize=(14, 5))
    fig3.suptitle("Local Models — Validation vs Test", fontsize=13, fontweight="bold")
    x3, bw = np.arange(len(lm_names)), 0.35
    for metric, ax, ylabel, fmt_fn in [
        ("RMSE", ar3, "RMSE ($)", lambda v: f"${v/1e3:.0f}k"),
        ("MAPE", am3, "MAPE (%)", lambda v: f"{v:.1f}%"),
    ]:
        vv = [local_metrics["Validation"][m][metric] for m in lm_names]
        tv = [local_metrics["Test"][m][metric]       for m in lm_names]
        cl = [model_colors[m] for m in lm_names]
        bv = ax.bar(x3 - bw/2, vv, bw, color=cl, alpha=0.55, edgecolor="white", label="Validation")
        bt = ax.bar(x3 + bw/2, tv, bw, color=cl, alpha=1.0,  edgecolor="white", label="Test")
        for b, v in zip(bv, vv):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()*1.01, fmt_fn(v),
                    ha="center", va="bottom", fontsize=8, color="dimgray")
        for b, v in zip(bt, tv):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()*1.01, fmt_fn(v),
                    ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.set_xticks(x3); ax.set_xticklabels(lm_names, rotation=12, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10); ax.set_title(ylabel, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9); ax.grid(axis="y", linestyle="--", alpha=0.4); ax.set_axisbelow(True)
    fig3.tight_layout()
    plt.close(fig3)

    # Fig 4 — All models: test set comparison (global vs all local)
    all_names = ["XGBoost Global"] + lm_names
    fig4, (ar4, am4) = plt.subplots(1, 2, figsize=(14, 5))
    fig4.suptitle("All Models — Test Set Comparison", fontsize=13, fontweight="bold")
    x4 = np.arange(len(all_names))
    for metric_vals, ax, ylabel, fmt_fn in [
        ([g_test_m["RMSE"]] + [local_metrics["Test"][n]["RMSE"] for n in lm_names],
         ar4, "RMSE ($)", lambda v: f"${v/1e3:.0f}k"),
        ([g_test_m["MAPE"]] + [local_metrics["Test"][n]["MAPE"] for n in lm_names],
         am4, "MAPE (%)", lambda v: f"{v:.1f}%"),
    ]:
        bars = ax.bar(x4, metric_vals, color=[model_colors[n] for n in all_names],
                      edgecolor="white", linewidth=0.6)
        for bar, v in zip(bars, metric_vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.01, fmt_fn(v),
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_xticks(x4); ax.set_xticklabels(all_names, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10); ax.set_title(ylabel, fontsize=11, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.4); ax.set_axisbelow(True)
    fig4.tight_layout()
    fig4.savefig("all_models_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig4)

    # ── JSON output ───────────────────────────────────────────────────────────
    print("\nWriting valuation_results.json...")
    output = {
        "metadata": metadata,
        "global_model": {
            "best_iteration":      int(gmodel.best_iteration),
            "metrics":             {"validation": g_val_m, "test": g_test_m},
            "feature_importances": feat_imp.to_dict(orient="records"),
        },
        "local_models": {
            "metrics": local_metrics,
        },
        "query_results": query_results,
    }
    with open("valuation_results.json", "w") as f:
        json.dump(sanitize(output), f, indent=2)

    print("\nDone. Outputs:")
    print("  global_model_results.png    — feature importances + global scatter")
    print("  local_model_scatter.png     — per-model scatter (val & test)")
    print("  local_model_comparison.png  — local models val vs test bars")
    print("  all_models_comparison.png   — all 5 models test set comparison")
    print("  valuation_results.json      — metrics, importances, per-query predictions & neighbours")


if __name__ == "__main__":
    main()