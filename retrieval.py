from __future__ import annotations

import time
import json
import textwrap
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import List, Optional
from datetime import datetime, timezone
from pathlib import Path

DATA_PATH = "data/cleaned_canada_housing_data.csv"
OUTPUT_PATH = "data/similarity_results.json"
K      = 5 
W_LOC  = 0.4
W_FEAT = 0.6
_DISPLAY_COLS = [
    "original_index",
    "City", "Province",
    "Latitude", "Longitude",
    "Price",
    "Bedrooms", "Bathrooms", "Acreage",
    "Property Type", "Square Footage",
    "Garage", "Parking", "Fireplace", "Waterfront",
    "Sewer", "Pool", "Garden", "Balcony",
]

# calculate the haversine distance: shortest great-circle distance between two points on the 
# surface of a sphere using their Latitude and Longitude
def haversine_vectorized(lat1, lon1, lat2, lon2):
    r = 6371.0 # mean earth radius in Kms
    t1, l1 = np.radians(lat1), np.radians(lon1)
    t2, l2 = np.radians(lat2), np.radians(lon2)
    diff_t = t2 - t1
    diff_l = l2 - l1
    a = np.sin(diff_t/2.0)**2 + np.cos(t1) * np.cos(t2) * np.sin(diff_l/2.0)**2
    # arctan2 implemented the same way as arcsin
    return r * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0-a))

# calculate euclidean distance from one query vector to every row in feature matrix
def euclidean_from_query(query_vec, feature_matrix):
    diff = feature_matrix - query_vec
    return np.sqrt(np.einsum("ij,ij->i", diff, diff))

# normalisation function - Scale an array to [0, 1].  eps prevents division-by-zero when all are identical
def min_max_normalize(arr, eps=1e-9):
    lo, hi = arr.min(), arr.max()
    return (arr-lo) / (hi-lo+eps)

# Retriver Class
class PropertySimilarityRetriever:
    def __init__ (self, df, feature_cols, w_loc=0.4, w_feat=0.6, lat_col="Latitude", lon_col="Longitude"):
        self._validate_inputs(df, feature_cols, w_loc, w_feat, lat_col, lon_col)
        self.df = df.reset_index(drop=True).copy()
        self.feature_cols = list(feature_cols)
        self.w_loc = w_loc
        self.w_feat = w_feat
        self.lat_col = lat_col
        self.lon_col = lon_col
        self._scalar = StandardScaler()
        self._precompute()
    
    def __repr__(self):
        return (
            f"PropertySimilarityRetriever("
            f"n={len(self.df)}, "
            f"features={self.feature_cols}, "
            f"w_loc={self.w_loc}, w_feat={self.w_feat})"
        )
    
    # validating inputs
    def _validate_inputs(self, df, feature_cols, w_loc, w_feat, lat_col, lon_col):
        # check for sum of weights = 1.0
        if not np.isclose(w_loc + w_feat, 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {w_loc + w_feat:.4f}")
        # check for existing columns in data
        for col in [lat_col, lon_col] + list(feature_cols):
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame")
    
    # Fit StandardScaler and cache the scaled feature matrix + coordinate arrays
    def _precompute(self):
        raw_features = self.df[self.feature_cols].values.astype(np.float64)
        self._scaled_features = self._scalar.fit_transform(raw_features)
        self._lats = self.df[self.lat_col].values.astype(np.float64)
        self._lons = self.df[self.lon_col].values.astype(np.float64)
        
    # calculate haversine distances
    def _loc_distances(self, query_idx):
        return haversine_vectorized(self._lats[query_idx], self._lons[query_idx], self._lats, self._lons)
    
    def _feat_distances(self, query_idx):
        return euclidean_from_query(self._scaled_features[query_idx], self._scaled_features)
    
    def get_similar_properties(self, query_index, k=5, w_loc=None, w_feat=None, return_breakdown=True):
        w_loc = self.w_loc if w_loc is None else w_loc
        w_feat = self.w_feat if w_feat is None else w_feat
        # check for valid arguements
        if not np.isclose(w_loc + w_feat, 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {w_loc + w_feat:.4f}")
        if query_index not in self.df.index:
            raise IndexError(f"query_index {query_index} not in DataFrame index")
        # get raw distances
        loc_raw = self._loc_distances(query_index)
        feat_raw = self._feat_distances(query_index)
        # normalize distances
        loc_norm = min_max_normalize(loc_raw)
        feat_norm = min_max_normalize(feat_raw)
        # combined weighted distance
        total = w_loc * loc_norm + w_feat * feat_norm
        # result formatting
        out = self.df.copy()
        out['location_dist_km'] = loc_raw
        out["location_dist_norm"] = loc_norm
        out["feature_dist_norm"] = feat_norm
        out["total_distance"] = total
        out = (out.drop(index=query_index).sort_values("total_distance").head(k).reset_index(drop=True))
        if not return_breakdown:
            out = out.drop(columns=["location_dist_km", "location_dist_norm", "feature_dist_norm"])
        return out        

# convience wrapper function
def get_similar_properties(df, query_index, feature_cols, k=5, w_loc=0.4, w_feat=0.6, lat_col="Latitude", lon_col="Longitude", return_breakdown=True):
    retriever = PropertySimilarityRetriever(df, feature_cols, w_loc, w_feat, lat_col, lon_col)
    return retriever.get_similar_properties(query_index, k, w_loc, w_feat, return_breakdown)


def load_and_preprocess(filepath):
    df = pd.read_csv(filepath).reset_index(drop=True)
    # Binary-encode amenity flags
    yes_no_cols = ["Garage", "Parking", "Fireplace", "Waterfront", "Pool", "Garden", "Balcony"]
    for col in yes_no_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0}).fillna(0).astype(int)
 
    # One-hot encode categorical columns (originals kept for display)
    for cat_col, prefix in [("Property Type", "PT"), ("Sewer", "Sewer")]:
        dummies = pd.get_dummies(df[cat_col], prefix=prefix).astype(int)
        df = pd.concat([df, dummies], axis=1)
    # Tag each row with its position so it survives the retriever's reset_index
    df["original_index"] = df.index
    # Collect feature column names
    numeric_cols = ["Bedrooms", "Bathrooms", "Acreage", "Square Footage"]
    pt_cols      = sorted(c for c in df.columns if c.startswith("PT_"))
    sewer_cols   = sorted(c for c in df.columns if c.startswith("Sewer_"))
    feature_cols = numeric_cols + yes_no_cols + pt_cols + sewer_cols
    return df, feature_cols
 
# Convert numpy scalars / NaN → native Python types for json.dump
def _to_python(v):
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return None if np.isnan(v) else float(v)
    if isinstance(v, float) and np.isnan(v):
        return None
    return v
 
 
def _row_to_dict(row, extra=None):
    d = {col: _to_python(row[col]) for col in _DISPLAY_COLS if col in row.index}
    if extra:
        d.update(extra)
    return d
 
 
def main():  
    print(f"Loading data from '{DATA_PATH}' …")
    df, feature_cols = load_and_preprocess(DATA_PATH)
    n = len(df)
    print(f"  {n} properties loaded")
    print(f"  Feature columns ({len(feature_cols)}):")
    for fc in feature_cols:
        print(f"    • {fc}")
    
    retriever = PropertySimilarityRetriever(df, feature_cols, W_LOC, W_FEAT, "Latitude", "Longitude")
    print(f"\n{retriever}\n")
    
    k_actual = min(K, n - 1)

    all_results: list[dict] = []
    sep = "─" * 64
    print(sep)
    start = time.perf_counter()
    
    for idx in range(n):
        query_row = df.iloc[idx]
        similar_df = retriever.get_similar_properties(query_index=idx, k=k_actual, return_breakdown=True,)
        sim_list = []
        for rank, (_, row) in enumerate(similar_df.iterrows(), start=1):
            sim_list.append(
                _row_to_dict(row, {
                    "rank":               rank, 
                    "location_dist_km":   round(float(row["location_dist_km"]),   4),
                    "location_dist_norm": round(float(row["location_dist_norm"]),  6),
                    "feature_dist_norm":  round(float(row["feature_dist_norm"]),   6),
                    "total_distance":     round(float(row["total_distance"]),       6),
                }))
        all_results.append({
            "query":              _row_to_dict(query_row),
            "similar_properties": sim_list,
        })
        city = str(query_row.get("City", f"idx={idx}"))
        print(f"  [{idx + 1:>{len(str(n))}}/{n}]  {city:<22}  → {len(sim_list)} similar found")
 
    elapsed = time.perf_counter() - start
    print(sep)
    
    output = {
        "metadata": {
            "source_file":       str(DATA_PATH),
            "total_properties":  n,
            "k":                 k_actual,
            "weights":           {"w_loc": W_LOC, "w_feat": W_FEAT},
            "features_used":     feature_cols,
            "display_cols":      _DISPLAY_COLS,
            "generated_at":      datetime.now(timezone.utc).isoformat(),
            "elapsed_ms":        round(elapsed * 1000, 2),
        },
        "results": all_results,
    }
    
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
 
    print(f"\n  {n} queries processed in {elapsed * 1000:.1f} ms  "
          f"({elapsed / n * 1000:.2f} ms / query)")
    print(f"  Results saved  →  {OUTPUT_PATH}")
    print(sep)
    

if __name__ == "__main__":
    main()
