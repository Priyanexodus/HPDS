#!/usr/bin/env python3
"""
build_combined_pinn_dataset.py

Generates the complete 4-category dataset for the Cardamom HPDS PINN.
Requires two files in the same directory:
  1. 'data.csv' (Real HPDS load cell and DHT22 sensor data)
  2. 'comsol_sparse_anchors.csv' (Exported Core & Mid-point data from COMSOL)
"""

import numpy as np
import pandas as pd
import h5py
import os

# ==========================================
# 1. CONSTANTS & GEOMETRY
# ==========================================
A_AXIAL = 9.2e-3     # z semi-axis (axial) in meters
B_RADIAL = 5.056e-3  # r semi-axis (radial) in meters
R_GAS = 8.314        # J/mol-K
M_WATER = 0.018      # kg/mol

# ==========================================
# 2. PHYSICS & REAL DATA FUNCTIONS
# ==========================================
def p_sat_water(T_C):
    """Saturation vapor pressure of water [Pa] (Magnus/Antoine form)."""
    log_p = 8.07131 - 1730.63 / (233.426 + T_C)
    return (10 ** log_p) * 133.3224

def build_boundary_forcing(df):
    """Category 2: T_air(t) and c_air(t) from real sensor log."""
    t_sec = df["Time_min"].values.astype(float) * 60.0
    T_air = df["Temperature_C"].values.astype(float)
    RH_air = df["Humidity_pct"].values.astype(float) / 100.0

    T_air_abs = T_air + 273.15
    p_vap = RH_air * p_sat_water(T_air)
    c_air = p_vap / (R_GAS * T_air_abs)  # mol/m^3

    return {"t_sec": t_sec, "T_air": T_air, "RH_air": RH_air, "c_air": c_air}

def build_MR_targets(df):
    """Category 3a: Integral MR(t) from the physical load cell."""
    t_sec = df["Time_min"].values.astype(float) * 60.0
    moisture_weight = df["Moisure_Weight"].values.astype(float)
    MR = moisture_weight / moisture_weight[0]
    return {"t_sec": t_sec, "MR": MR, "moisture_weight_kg": moisture_weight}

def load_comsol_anchors(filepath):
    """Category 3b: Load Sparse Field Anchors from COMSOL CSV export."""
    if not os.path.exists(filepath):
        print(f"\n[WARNING] {filepath} not found. Proceeding without Category 3b.")
        return None
        
    # Assuming COMSOL export columns: Time, r, z, T, c
    # comment='%' skips the standard COMSOL metadata headers
    df_comsol = pd.read_csv(filepath, comment='%', header=0)
    
    t_anchor = df_comsol.iloc[:, 0].values * 60.0 # Convert min to sec
    r_anchor = df_comsol.iloc[:, 1].values
    z_anchor = df_comsol.iloc[:, 2].values
    T_anchor = df_comsol.iloc[:, 3].values
    c_anchor = df_comsol.iloc[:, 4].values
    
    return np.column_stack((r_anchor, z_anchor, t_anchor, T_anchor, c_anchor))

# ==========================================
# 3. COLLOCATION SAMPLING
# ==========================================
def sample_collocation_points(n_points, total_time_s, rng=None):
    """Category 1: Rejection-sample interior points of the elliptical pod."""
    if rng is None:
        rng = np.random.default_rng()

    r_acc = np.empty(0)
    z_acc = np.empty(0)
    
    while r_acc.shape[0] < n_points:
        remaining = n_points - r_acc.shape[0]
        batch_n = int(remaining / 0.78) + 16
        r = rng.uniform(0.0, B_RADIAL, batch_n)
        z = rng.uniform(-A_AXIAL, A_AXIAL, batch_n)
        
        # Keep only points strictly inside the ellipse equation
        mask = (r / B_RADIAL) ** 2 + (z / A_AXIAL) ** 2 < 1.0
        r_acc = np.concatenate([r_acc, r[mask]])
        z_acc = np.concatenate([z_acc, z[mask]])

    r_acc = r_acc[:n_points]
    z_acc = z_acc[:n_points]
    t_acc = rng.uniform(0.0, total_time_s, n_points)

    coords = np.stack(
        [r_acc / B_RADIAL, z_acc / A_AXIAL, t_acc / total_time_s], axis=-1
    ).astype(np.float32)
    return coords

# ==========================================
# 4. MAIN EXECUTION PIPELINE
# ==========================================
if __name__ == "__main__":
    # File Paths
    SENSOR_CSV = "/home/priyan/physicsnemo/HPDS/data/data.csv"
    COMSOL_CSV = "/home/priyan/physicsnemo/HPDS/synthetic_data/PINN/comsol_sparse_anchors.csv"
    
    # Changed extension to .h5
    OUTPUT_FILE = "/home/priyan/physicsnemo/HPDS/synthetic_data/PINN/hpds_pinn_dataset.h5" 
    N_COLLOCATION = 8192

    print(f"--- 1. Loading Real Sensor Data ({SENSOR_CSV}) ---")
    df_sensors = pd.read_csv(SENSOR_CSV)
    
    boundary = build_boundary_forcing(df_sensors)
    mr = build_MR_targets(df_sensors)
    TOTAL_TIME_S = boundary["t_sec"][-1]
    print(f"  -> Extracted Boundary Conditions & Load Cell MR (Max Time: {TOTAL_TIME_S/60:.0f} min)")

    print(f"\n--- 2. Loading COMSOL Anchors ({COMSOL_CSV}) ---")
    comsol_anchors = load_comsol_anchors(COMSOL_CSV)
    if comsol_anchors is not None:
        print(f"  -> Successfully loaded {comsol_anchors.shape[0]} structural anchors.")

    print("\n--- 3. Generating Physics Collocation Points ---")
    rng = np.random.default_rng(42)
    collocation = sample_collocation_points(N_COLLOCATION, TOTAL_TIME_S, rng=rng)
    print(f"  -> Generated {collocation.shape[0]} interior coordinates.")

    print("\n--- 4. Packaging Final Dataset into HDF5 ---")
    
    with h5py.File(OUTPUT_FILE, 'w') as f:
        # --- File-Level Metadata (Attributes) ---
        f.attrs["description"] = "Combined PINN dataset for Cardamom HPDS"
        f.attrs["pod_a_axial_m"] = A_AXIAL
        f.attrs["pod_b_radial_m"] = B_RADIAL
        f.attrs["total_time_s"] = TOTAL_TIME_S
        f.attrs["R_GAS"] = R_GAS
        f.attrs["M_WATER"] = M_WATER
        
        # --- Category 1: Collocation Points ---
        grp_cat1 = f.create_group("category_1_collocation")
        grp_cat1.create_dataset(
            "coords", 
            data=collocation, 
            dtype='float32', 
            compression='gzip'
        )

        # --- Category 2: Boundary Forcing (Sensor Telemetry) ---
        grp_cat2 = f.create_group("category_2_boundary")
        for key, array_data in boundary.items():
            grp_cat2.create_dataset(
                key, 
                data=array_data, 
                dtype='float32', 
                compression='gzip'
            )

        # --- Category 3a: Load Cell MR Targets ---
        grp_cat3a = f.create_group("category_3a_mr")
        for key, array_data in mr.items():
            grp_cat3a.create_dataset(
                key, 
                data=array_data, 
                dtype='float32', 
                compression='gzip'
            )

        # --- Category 3b: COMSOL Anchors ---
        if comsol_anchors is not None:
            grp_cat3b = f.create_group("category_3b_comsol")
            grp_cat3b.create_dataset(
                "anchors", 
                data=comsol_anchors, 
                dtype='float32', 
                compression='gzip'
            )
            
    print(f"SUCCESS: HDF5 Dataset saved to {OUTPUT_FILE}")