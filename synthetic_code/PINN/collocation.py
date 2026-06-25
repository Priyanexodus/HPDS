"""
synthetic_comsol_anchors.py

Generates synthetic Category 3b Sparse Field Anchors for the Cardamom HPDS PINN,
replacing actual COMSOL FEM output with a physics-based diffusion model.

Physics engine
--------------
  Geometry : Prolate ellipsoid (A_AXIAL=9.2mm, B_RADIAL=5.056mm) mapped to
             an equivalent sphere via normalised ellipsoidal coordinate ξ.
  Thermal  : 1-D spherical heat equation, Robin BC (Bi_T ≈ 0.55).
             Uses N=30 nodes — Fo_T ≈ 45 means field is nearly uniform;
             coarse spatial grid is both accurate and fast.
  Moisture : 1-D spherical diffusion, Dirichlet BC (Bi_m >> 1 → surface = c_emc).
             Uses N=100 nodes — Fo_m ≈ 0.5–2 means a large spatial gradient
             persists, requiring fine resolution.

Stability note (centre-node factor)
--------------------------------------
  At r = 0 the control-volume limit gives  du/dt = 6α(u₁-u₀)/dr².
  This is 3× larger than the interior coefficient, so the CFL limit for the
  centre node is  dt ≤ dr²/(3α).  We use CFL_FACTOR = 0.25 < 1/3 throughout,
  which is stable for every node including centre and Robin-BC surface.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import os, time

# ═══════════════════════════════════════════════════════════════════════════════
# 1. GEOMETRY
# ═══════════════════════════════════════════════════════════════════════════════
A_AXIAL  = 9.2e-3     # z semi-axis [m]
B_RADIAL = 5.056e-3   # r semi-axis [m]
R_EFF    = (A_AXIAL * B_RADIAL ** 2) ** (1.0 / 3.0)   # ≈ 6.17 mm

# ═══════════════════════════════════════════════════════════════════════════════
# 2. THERMOPHYSICAL PROPERTIES  (green cardamom, literature)
# ═══════════════════════════════════════════════════════════════════════════════
ALPHA_T   = 1.2e-7    # Thermal diffusivity [m²/s]
K_COND    = 0.20      # Thermal conductivity [W/m·K]
H_CONV    = 18.0      # Convective HTC [W/m²·K]
BI_T      = H_CONV * R_EFF / K_COND      # ≈ 0.556

D_EFF_50  = 2.0e-9    # Moisture diffusivity at 50 °C [m²/s]
T_REF_K   = 323.15    # 50 °C in Kelvin
EA_MOIST  = 2.8e4     # Activation energy [J/mol]

R_GAS     = 8.314     # J/mol·K

CFL_FACTOR = 0.25     # < 1/3  → stable for centre node (factor-6) AND Robin BC

# ═══════════════════════════════════════════════════════════════════════════════
# 3. THERMODYNAMIC HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def p_sat_water(T_C):
    """Saturation vapour pressure [Pa] — Antoine/Magnus."""
    return (10.0 ** (8.07131 - 1730.63 / (233.426 + T_C))) * 133.3224

def D_eff_at_T(T_C):
    """Arrhenius temperature correction for effective moisture diffusivity [m²/s]."""
    T_K = T_C + 273.15
    return D_EFF_50 * np.exp(-EA_MOIST / R_GAS * (1.0 / T_K - 1.0 / T_REF_K))

def xi_ellipsoid(r, z):
    """Normalised ellipsoidal radial coord ξ (0=centre, 1=surface)."""
    return np.sqrt((r / B_RADIAL) ** 2 + (z / A_AXIAL) ** 2)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. FINITE-DIFFERENCE SOLVER — 1-D SPHERICAL DIFFUSION
# ═══════════════════════════════════════════════════════════════════════════════

def solve_spherical_diffusion(t_arr, u_init, bc_fn, alpha_fn,
                              R, N=100, robin_bi=None, u_amb_fn=None):
    """Explicit FD solver for  ∂u/∂t = (α/r²) ∂/∂r(r² ∂u/∂r)."""
    r_vec = np.linspace(0.0, R, N)
    dr    = r_vec[1] - r_vec[0]

    u      = np.full(N, float(u_init))
    u_hist = np.zeros((len(t_arr), N))
    u_hist[0] = u.copy()

    for i in range(1, len(t_arr)):
        t0, t1  = t_arr[i - 1], t_arr[i]
        dt_span = t1 - t0

        alpha_mid = alpha_fn(0.5 * (t0 + t1))
        dt_max    = CFL_FACTOR * dr ** 2 / alpha_mid
        n_sub     = max(1, int(np.ceil(dt_span / dt_max)))
        dt        = dt_span / n_sub

        for k in range(n_sub):
            t_now   = t0 + k * dt
            alpha_k = alpha_fn(t_now)
            du      = np.zeros(N)

            # ── Interior nodes ─────────────────────────────────────────────
            r_m = r_vec[1:-1]
            du[1:-1] = alpha_k * (
                (u[2:] - 2.0 * u[1:-1] + u[:-2]) / dr ** 2
                + (2.0 / r_m) * (u[2:] - u[:-2]) / (2.0 * dr)
            )

            # ── Centre node (factor 6: control-volume result at r=0) ───────
            du[0] = 6.0 * alpha_k * (u[1] - u[0]) / dr ** 2

            # ── Surface node ───────────────────────────────────────────────
            if robin_bi is None or robin_bi > 100:
                # Dirichlet: march interior, pin surface
                u[:-1] += dt * du[:-1]
                u[-1]   = bc_fn(t_now + dt)
            else:
                # Robin: ghost-point
                u_amb   = float(u_amb_fn(t_now))
                u_ghost = u[-2] - 2.0 * dr * (robin_bi / R) * (u[-1] - u_amb)
                du[-1]  = alpha_k * (
                    (u_ghost - 2.0 * u[-1] + u[-2]) / dr ** 2
                    + (2.0 / r_vec[-1]) * (u_ghost - u[-2]) / (2.0 * dr)
                )
                u += dt * du

        u_hist[i] = u.copy()

    return u_hist, r_vec


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ANCHOR LOCATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_anchor_locations():
    """6 (r, z, label) anchor points strictly inside the ellipsoid."""
    return [
        (0.0,              0.0,              "core_center"),
        (B_RADIAL * 0.50,  0.0,              "mid_radial"),
        (0.0,              A_AXIAL * 0.50,  "mid_axial"),
        (B_RADIAL * 0.50,  A_AXIAL * 0.50,  "mid_diagonal"),
        (B_RADIAL * 0.75,  0.0,              "near_surf_r"),
        (0.0,              A_AXIAL * 0.75,  "near_surf_z"),
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MAIN GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def generate_synthetic_comsol_anchors(
    sensor_csv  = "data.csv",
    output_csv  = "comsol_sparse_anchors.csv",
    n_time_pts  = 60,
    N_thermal   = 30,     
    N_moisture  = 100,    
    noise_frac  = 0.02,
    seed        = 42,
):
    rng = np.random.default_rng(seed)
    t_wall = time.time()

    # ── 6a. Load real sensor data ─────────────────────────────────────────────
    print(f"  [1/5] Loading sensor data from '{sensor_csv}'...")
    df        = pd.read_csv(sensor_csv)
    t_raw     = df["Time_min"].values.astype(float) * 60.0
    T_air_raw = df["Temperature_C"].values.astype(float)
    RH_raw    = df["Humidity_pct"].values.astype(float) / 100.0

    T_air_fn = interp1d(t_raw, T_air_raw, kind="linear", fill_value="extrapolate")
    
    T_total = float(t_raw[-1])
    t_grid  = np.linspace(0.0, T_total, n_time_pts)

    # ── 6b. Initial conditions & Boundaries (CORRECTED) ───────────────────────
    T_0    = float(T_air_raw[0])
    c_0    = 47082.0  # Bulk moisture concentration [mol/m³]

    # Equilibrium Moisture Content (EMC) boundary function proxy
    RH_init = float(RH_raw[0])
    def c_surf_emc_fn(t):
        RH_now = np.interp(t, t_raw, RH_raw)
        return c_0 * (RH_now / RH_init) 

    # ── 6c. Diagnostics ───────────────────────────────────────────────────────
    D_avg  = D_eff_at_T(float(T_air_raw.mean()))
    Fo_T   = ALPHA_T * T_total / R_EFF ** 2
    Fo_m   = D_avg   * T_total / R_EFF ** 2
    dr_T   = R_EFF / (N_thermal  - 1)
    dr_m   = R_EFF / (N_moisture - 1)
    nsub_T = int(np.ceil(T_total / (n_time_pts - 1) / (CFL_FACTOR * dr_T**2 / ALPHA_T)))
    nsub_m = int(np.ceil(T_total / (n_time_pts - 1) / (CFL_FACTOR * dr_m**2 / D_avg)))

    print(f"\n  ┌─ Pod geometry ──────────────────────────────────────────")
    print(f"  │  A_axial = {A_AXIAL*1e3:.2f} mm | B_radial = {B_RADIAL*1e3:.3f} mm")
    print(f"  │  R_eff   = {R_EFF*1e3:.2f} mm  (equivalent sphere)")
    print(f"  ├─ Boundary conditions ──────────────────────────────────")
    print(f"  │  T_air   ∈ [{T_air_raw.min():.1f}, {T_air_raw.max():.1f}] °C")
    print(f"  ├─ Initial conditions ───────────────────────────────────")
    print(f"  │  T_0 = {T_0:.1f} °C  |  c_0 = {c_0:.1f} mol/m³ (Bulk Mass)")
    print(f"  ├─ Dimensionless numbers ────────────────────────────────")
    print(f"  │  Bi_T = {BI_T:.3f}  (Robin BC — surface resistance matters)")
    print(f"  │  Fo_T = {Fo_T:.1f}   (>>1 → fast thermal equilibration)")
    print(f"  │  D_eff(T̄={T_air_raw.mean():.0f}°C) = {D_avg:.3e} m²/s  (Arrhenius)")
    print(f"  │  Fo_m = {Fo_m:.3f}  (spatial gradient persists throughout)")
    print(f"  ├─ Solver ───────────────────────────────────────────────")
    print(f"  │  CFL = {CFL_FACTOR} | T: N={N_thermal}, ~{nsub_T} sub-steps/interval")
    print(f"  │              | c: N={N_moisture}, ~{nsub_m} sub-steps/interval")
    print(f"  └────────────────────────────────────────────────────────\n")

    # ── 6d. Thermal PDE (Robin BC, coarse grid) ───────────────────────────────
    print("  [2/5] Solving thermal PDE  (Robin BC, N_thermal={}) ...".format(N_thermal))
    T_field, r_T = solve_spherical_diffusion(
        t_arr    = t_grid,
        u_init   = T_0,
        bc_fn    = T_air_fn,
        alpha_fn = lambda t: ALPHA_T,
        R        = R_EFF,
        N        = N_thermal,
        robin_bi = BI_T,
        u_amb_fn = T_air_fn,
    )

    # ── 6e. Moisture PDE (Dirichlet BC, fine grid) ────────────────────────────
    print("  [3/5] Solving moisture PDE (Bulk EMC Dirichlet BC, Arrhenius D_eff, N_moisture={}) ...".format(N_moisture))
    c_field, r_c = solve_spherical_diffusion(
        t_arr    = t_grid,
        u_init   = c_0,
        bc_fn    = c_surf_emc_fn, 
        alpha_fn = lambda t: D_eff_at_T(float(T_air_fn(t))),
        R        = R_EFF,
        N        = N_moisture,
        robin_bi = None,
    )

    # Safety: clamp non-finite values
    for arr, name in [(T_field, "T"), (c_field, "c")]:
        if not np.isfinite(arr).all():
            n_bad = (~np.isfinite(arr)).sum()
            print(f"  [WARN] {n_bad} non-finite values in {name} field — clamped.")
            arr[~np.isfinite(arr)] = np.nanmedian(arr)

    # ── 6f. Interpolators: 1-D sphere → any ρ ────────────────────────────────
    T_interp = interp1d(r_T, T_field, axis=1, assume_sorted=True,
                        bounds_error=False, fill_value=(T_field[:, 0], T_field[:, -1]))
    c_interp = interp1d(r_c, c_field, axis=1, assume_sorted=True,
                        bounds_error=False, fill_value=(c_field[:, 0], c_field[:, -1]))

    # ── 6g. Extract anchor values + add noise ─────────────────────────────────
    print("  [4/5] Extracting anchor field values ...")
    anchors = get_anchor_locations()
    rows    = []

    for (r_a, z_a, label) in anchors:
        xi  = xi_ellipsoid(r_a, z_a)
        rho = float(np.clip(xi * R_EFF, 0.0, R_EFF * 0.999))

        T_ts = T_interp(rho)   # [n_time_pts]
        c_ts = c_interp(rho)

        # 2 % relative Gaussian noise
        T_sig = noise_frac * max(float(np.std(T_ts)), 0.05)
        c_sig = noise_frac * max(float(np.std(c_ts)), 1e-7)
        T_n   = rng.normal(0.0, T_sig, n_time_pts)
        c_n   = rng.normal(0.0, c_sig, n_time_pts)

        for k in range(n_time_pts):
            rows.append({
                "Time_min" : t_grid[k] / 60.0,
                "r_m"      : r_a,
                "z_m"      : z_a,
                "T_C"      : float(T_ts[k] + T_n[k]),
                "c_mol_m3" : float(max(0.0, c_ts[k] + c_n[k])),
                "label"    : label,
            })

    df_out = pd.DataFrame(rows)[["Time_min", "r_m", "z_m", "T_C", "c_mol_m3", "label"]]
    df_out.to_csv(output_csv, index=False)

    # ── 6h. Validation report ─────────────────────────────────────────────────
    print(f"\n  [5/5] Validation report")
    hdr = f"  {'Location':<20} {'ξ':>5} {'T(t₀)':>8} {'T(tf)':>8} {'c(t₀)':>10} {'c(tf)':>10}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for (r_a, z_a, label) in anchors:
        sub = df_out[df_out["label"] == label]
        xi  = xi_ellipsoid(r_a, z_a)
        print(f"  {label:<20} {xi:>5.3f}"
              f"  {sub['T_C'].iloc[0]:>6.2f}°C"
              f"  {sub['T_C'].iloc[-1]:>6.2f}°C"
              f"  {sub['c_mol_m3'].iloc[0]:>9.1f}"
              f"  {sub['c_mol_m3'].iloc[-1]:>9.1f}")

    core   = df_out[df_out["label"] == "core_center"]
    ns_r   = df_out[df_out["label"] == "near_surf_r"]
    Δc     = core["c_mol_m3"].iloc[-1] - ns_r["c_mol_m3"].iloc[-1]
    
    status = "✓ gradient present" if Δc > 0 else "✗ check D_eff"
    print(f"\n  Δc  (core − near_surf)   = {Δc:+.1f} mol/m³  {status}")
    print(f"\n  Output : '{output_csv}'")
    print(f"  Rows   : {len(df_out)}  ({len(anchors)} locations × {n_time_pts} time pts)")
    print(f"  Elapsed: {time.time() - t_wall:.1f} s")

    return df_out


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("═" * 62)
    print("  Synthetic COMSOL Sparse Anchor Generator")
    print("  Category 3b — Physics-Based 1-D Spherical Diffusion")
    print("  Cardamom HPDS PINN placeholder  (target: ≥80 % accuracy)")
    print("═" * 62 + "\n")

    df = generate_synthetic_comsol_anchors(
        sensor_csv  = "/home/priyan/physicsnemo/HPDS/data/data.csv",
        output_csv  = "/home/priyan/physicsnemo/HPDS/synthetic_data/PINN/comsol_sparse_anchors.csv",
        n_time_pts  = 60,
        N_thermal   = 30,
        N_moisture  = 100,
        noise_frac  = 0.02,
        seed        = 42,
    )

    print("\n  Sample (core_center, first 5 rows):")
    print(df[df["label"] == "core_center"].head(5).to_string(index=False))
    print("\n  Done.  Pass 'comsol_sparse_anchors.csv' to your PINN Dataset Generator.")