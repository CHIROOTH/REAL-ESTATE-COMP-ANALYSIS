import numpy as np
import pandas as pd
from __future__ import annotations
from sklearn.preprocessing import StandardScaler
from typing import List, Optional

# calculate the haversine distance: shortest great-circle distance between two points on the 
# surface of a sphere using their latitude and longitude
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
class PropertSimilarityRetriever:
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
        self._lons = self.df[self.lon_col].values.astypr(np.float64)
        
    # calculate haversine distances
    def _loc_distances(self, query_idx):
        