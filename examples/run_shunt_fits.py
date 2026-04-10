"""Fit LinearShuntFitter to S21 CSV data and save results as SVG."""
from __future__ import division, absolute_import, print_function

import os
import re
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from resonator import background, shunt, see

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR = "/Users/b3-34/Projects/uni/bakalaurinis/matavimai/bowties"
OUT_DIR = os.path.join(os.path.dirname(__file__), "shunt_fits")

# Frequency window around the resonance to use for fitting (fractional half-width)
FREQ_WINDOW = 0.005   # ±0.5 % of the file's centre frequency

os.makedirs(OUT_DIR, exist_ok=True)

csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
if not csv_files:
    raise SystemExit(f"No CSV files found in {DATA_DIR}")

summary_rows = []

for csv_path in csv_files:
    name = os.path.splitext(os.path.basename(csv_path))[0]

    # Skip files already processed
    if os.path.exists(os.path.join(OUT_DIR, f"{name}_complex.svg")):
        print(f"Skipping {name} (already fitted)")
        continue

    print(f"\n=== Processing {name} ===")

    # ── Load data ─────────────────────────────────────────────────────────────
    df = pd.read_csv(csv_path, skiprows=2)
    df.columns = ['frequency', 'S21_db', 'S21_deg']
    df['S21_mag'] = np.float_power(10, df['S21_db'].to_numpy() / 20)

    # Narrow window around the dip (minimum magnitude → resonance estimate)
    freq_arr = df['frequency'].to_numpy()
    mag_arr = df['S21_mag'].to_numpy()
    fr_guess = freq_arr[np.argmin(mag_arr)]
    freq_min = fr_guess * (1 - FREQ_WINDOW)
    freq_max = fr_guess * (1 + FREQ_WINDOW)
    df = df[(df['frequency'] >= freq_min) & (df['frequency'] <= freq_max)]

    if len(df) < 10:
        print(f"  Too few points after windowing ({len(df)}), skipping.")
        continue

    data = df['S21_mag'].to_numpy() * np.exp(1j * np.deg2rad(df['S21_deg'])).to_numpy()
    frequency = df['frequency'].to_numpy()

    # ── Fit ───────────────────────────────────────────────────────────────────
    try:
        resonator = shunt.LinearShuntFitter(
            frequency=frequency,
            data=data,
            background_model=background.MagnitudeSlopeOffsetPhaseDelay(),
        )
    except Exception as exc:
        print(f"  Fit failed: {exc}")
        continue

    print(resonator.result.fit_report())

    # ── Extract fitted parameters ──────────────────────────────────────────────
    p = resonator.result.params
    f_r        = p['resonance_frequency'].value
    kappa_c    = p['coupling_loss'].value
    kappa_i    = p['internal_loss'].value
    asymmetry  = p['asymmetry'].value
    Q_i        = 1.0 / kappa_i
    Q_c        = 1.0 / kappa_c
    Q_total    = 1.0 / (kappa_i + kappa_c)
    redchi     = resonator.result.redchi
    success    = resonator.result.success

    # ── Plot 1: complex plane (measurement + resonator plane) ──────────────────
    fig, (ax_raw, ax_norm) = plt.subplots(1, 2, figsize=(7, 3.5))
    ax_raw.set_title('measurement plane')
    ax_norm.set_title('resonator plane')
    see.real_and_imaginary(resonator=resonator, axes=ax_raw, normalize=False)
    see.real_and_imaginary(resonator=resonator, axes=ax_norm, normalize=True)
    ax_raw.legend(fontsize='xx-small')
    fig.suptitle(name, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{name}_complex.svg"), format='svg')
    plt.close(fig)

    # ── Plot 2: magnitude & phase vs frequency ─────────────────────────────────
    fig, axes = plt.subplots(2, 2, sharex='all', figsize=(7, 6))
    ax_rm, ax_nm, ax_rp, ax_np = axes.flatten()
    ax_rm.set_title('measurement plane')
    ax_nm.set_title('resonator plane')
    ax_rp.set_title('measurement plane')
    ax_np.set_title('resonator plane')
    see.magnitude_vs_frequency(resonator=resonator, axes=ax_rm, normalize=False, frequency_scale=1e-9)
    see.magnitude_vs_frequency(resonator=resonator, axes=ax_nm, normalize=True,  frequency_scale=1e-9)
    see.phase_vs_frequency(   resonator=resonator, axes=ax_rp, normalize=False, frequency_scale=1e-9)
    see.phase_vs_frequency(   resonator=resonator, axes=ax_np, normalize=True,  frequency_scale=1e-9)
    fig.suptitle(name, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{name}_mag_phase.svg"), format='svg')
    plt.close(fig)

    # ── Plot 3: triptych ───────────────────────────────────────────────────────
    try:
        fig, axes_t = see.triptych(resonator=resonator, plot_initial=True,
                                   frequency_scale=1e-9,
                                   figure_settings={'figsize': (8, 3.2)})
        axes_t[2].legend(fontsize='xx-small')
        fig.suptitle(name, fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"{name}_triptych.svg"), format='svg')
        plt.close(fig)
    except Exception as exc:
        print(f"  Triptych plot failed: {exc}")

    summary_rows.append({
        'file':       name,
        'fr_GHz':     f_r * 1e-9,
        'Q_i':        Q_i,
        'Q_c':        Q_c,
        'Q_total':    Q_total,
        'kappa_i':    kappa_i,
        'kappa_c':    kappa_c,
        'asymmetry':  asymmetry,
        'redchi':     redchi,
        'success':    success,
    })

# ── Summary CSV ───────────────────────────────────────────────────────────────
if summary_rows:
    df_out = pd.DataFrame(summary_rows)
    csv_out = os.path.join(OUT_DIR, "fit_summary.csv")
    df_out.to_csv(csv_out, index=False, float_format='%.6g')
    print(f"\nSummary saved to {csv_out}")

    print("\n" + df_out.to_string(index=False))
else:
    print("No new files processed.")

print(f"\nDone. Results saved to {OUT_DIR}")
