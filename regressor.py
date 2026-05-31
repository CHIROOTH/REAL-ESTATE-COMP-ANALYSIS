import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

DATA_PATH = "similarity_results.json"
RANDOM_SEED = 42
K_DEFAULT = 5
QUERY_FEATURE_COLS = [
    "Bedrooms", "Bathrooms", "Acreage", "Square Footage",
    "Garage", "Parking", "Fireplace", "Waterfront",
    "Pool", "Garden", "Balcony", "Latitude", "Longitude",
]
NEIGHBOR_PROPERTY_COLS = [
    "Bedrooms", "Bathrooms", "Acreage", "Square Footage",
    "Garage", "Parking", "Fireplace", "Waterfront",
    "Pool", "Garden", "Balcony",
]
NEIGHBOR_DISTANCE_COLS = [
    "location_dist_km", "location_dist_norm",
    "feature_dist_norm", "total_distance",
]

def flatten_record(record, k):
    query = record["query"]
    neighbors = record.get("similar_properties", [])
    # skipping records without enough neighbors - manual pricing for those
    if len(neighbors) < k:
        return None
    row = {}
    row["Price"] = float(query["Price"])
    for col in QUERY_FEATURE_COLS:
        row[f"query_{col.lower().replace(' ', '_')}"] = query.get(col, np.nan)
    for i, nbr in enumerate(neighbors[:k], start=1):
        prefix = f"neighbor{i}_"
        row[f"{prefix}price"] = float(nbr.get("Price", np.nan))
        for col in NEIGHBOR_PROPERTY_COLS:
            safe_col = col.lower().replace(" ", "_")
            row[f"{prefix}{safe_col}"] = nbr.get(col, np.nan)
        for col in NEIGHBOR_DISTANCE_COLS:
            row[f"{prefix}{col}"] = nbr.get(col, np.nan)
    return row
            
def evaluate(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask]-y_pred[mask])/y_true[mask])) * 100
    
    print(f"\n  {label}")
    print(f"    MAE  : ${mae:>12,.2f}")
    print(f"    RMSE : ${rmse:>12,.2f}")
    print(f"    R²   :  {r2:>12.4f}")
    print(f"    MAPE :  {mape:>11.2f} %")
 
    return {"label": label, "MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}

def scatter_actual_vs_pred(ax, y_true, y_pred, metrics, title):
    lims = [
        min(y_true.min(), y_pred.min()) * 0.95,
        max(y_true.max(), y_pred.max()) * 1.05,
    ]
    ax.scatter(y_true, y_pred, alpha=0.4, s=15, color="#0EA5E9", edgecolors="none")
    ax.plot(lims, lims, "r--", linewidth=1.2, label="Perfect prediction")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Actual Price ($)", fontsize=10)
    ax.set_ylabel("Predicted Price ($)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1e6:.1f}M"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1e6:.1f}M"))
    info = (f"MAE  ${metrics['MAE']:,.0f}\n"
            f"RMSE ${metrics['RMSE']:,.0f}\n"
            f"R²   {metrics['R2']:.4f}\n"
            f"MAPE {metrics['MAPE']:.2f}%")
    ax.text(0.04, 0.96, info, transform=ax.transAxes,
            va="top", ha="left", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.8))
    ax.legend(fontsize=8)
    ax.grid(linestyle="--", alpha=0.35)
 

def global_regressor():
    with open(DATA_PATH, "r") as f:
        raw_data = json.load(f)
    
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
    
    print(f"  Records loaded : {len(records):,}")

    rows = []
    skipped = 0
    for rec in records:
        flat = flatten_record(rec, K)
        if flat is None:
            skipped += 1
        else:
            rows.append(flat)
    
    if skipped:
        print(f"  Skipped {skipped} record(s) with fewer than {K} neighbors.")

    df = pd.DataFrame(rows)
    print(f"  DataFrame shape : {df.shape}  ({df.shape[0]} rows x {df.shape[1]} columns)")
    
    TARGET_COL = "Price"
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    X = df[feature_cols].copy()
    Y = df[TARGET_COL].copy()
    
    print(f"\n Feature columns : {len(feature_cols)}")
    print(f" Target : {TARGET_COL}")
    
    feature_cols = [c for c in df.columns if c != TARGET_COL]

    X = df[feature_cols].copy()
    y = df[TARGET_COL].copy()

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_SEED
    )
    
    print(f"  Train : {len(X_train):,} rows  ({len(X_train)/len(X)*100:.1f} %)")
    print(f"  Val   : {len(X_val):,} rows  ({len(X_val)/len(X)*100:.1f} %)")
    print(f"  Test  : {len(X_test):,} rows  ({len(X_test)/len(X)*100:.1f} %)")
    
    # global regressor
    model = XGBRegressor(
        n_estimators      = 500,
        max_depth         = 6,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        objective         = "reg:squarederror",
        random_state      = RANDOM_SEED,
        early_stopping_rounds = 30,
        eval_metric       = "rmse",
        verbosity         = 0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    best_iter = model.best_iteration
    print(f"  Best iteration (early stopping) : {best_iter}")

    y_val_pred  = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    val_metrics  = evaluate(y_val.values,  y_val_pred,  "Validation set")
    test_metrics = evaluate(y_test.values, y_test_pred, "Test set")
    
    importances = model.feature_importances_
    feat_imp_df = (
        pd.DataFrame({"feature": feature_cols, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    
    print(feat_imp_df.head(20).to_string(index=False))
    
    # plotting
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    fig.suptitle("Real Estate XGBoost Valuation Model", fontsize=15, fontweight="bold", y=1.01)
    
    top20 = feat_imp_df.head(20)
    ax = axes[0]
    bars = ax.barh(
        top20["feature"][::-1],
        top20["importance"][::-1],
        color="#2563EB",
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set_xlabel("F-score (gain)", fontsize=10)
    ax.set_title("Top 20 Feature Importances", fontsize=11, fontweight="bold")
    ax.tick_params(axis="y", labelsize=8)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    
    scatter_actual_vs_pred(
        axes[1], y_val.values, y_val_pred, val_metrics,
        "Actual vs Predicted — Validation"
    )
    
    plt.tight_layout()
    output_plot = "housing_global_results.png"
    plt.savefig(output_plot, dpi=150, bbox_inches="tight")
    print(f"  Plot saved → {output_plot}")
    
    print("\n" + "=" * 65)
    print("FINAL SUMMARY")
    print("=" * 65)
    summary = pd.DataFrame([val_metrics, test_metrics]).set_index("label")
    summary["MAE"]  = summary["MAE"].map("${:,.0f}".format)
    summary["RMSE"] = summary["RMSE"].map("${:,.0f}".format)
    summary["R2"]   = summary["R2"].map("{:.4f}".format)
    summary["MAPE"] = summary["MAPE"].map("{:.2f}%".format)
    print(summary.to_string())
    print("=" * 65)
    print("Done.")



def main():
    global_regressor()
    
    



if __name__ == "__main__":
    main()
