"""Fit LinearShuntFitter / LinearReflectionFitter to CSV data and save SVG results."""
from __future__ import division, absolute_import, print_function

import os
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lmfit

from resonator import background, shunt, reflection, see

# ── Job definitions ────────────────────────────────────────────────────────────
# Each entry describes one dataset to process.
# Keys:
#   data_dir   – folder containing *.csv files
#   out_dir    – where SVGs and fit_summary.csv are written
#   fitter     – 'shunt' or 'reflection'
#   skiprows   – header rows to skip when reading CSVs (default 2)
#   freq_window – fractional half-width around the dip used for windowing (default 0.005)

JOBS = [
    {
        'data_dir':    "/Users/b3-34/Projects/uni/bakalaurinis/matavimai/bowties",
        'out_dir':     os.path.join(os.path.dirname(__file__), "shunt_fits"),
        'fitter':      'shunt',
    },
    {
        'data_dir':    "/Users/b3-34/Projects/uni/bakalaurinis/matavimai/droplet_csv",
        'out_dir':     os.path.join(os.path.dirname(__file__), "droplet_fits"),
        'fitter':      'reflection',
    },
]

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_and_window(csv_path, skiprows, freq_window):
    df = pd.read_csv(csv_path, skiprows=skiprows)
    df.columns = ['frequency', 'S21_db', 'S21_deg']
    df['S21_mag'] = np.float_power(10, df['S21_db'].to_numpy() / 20)

    freq_arr = df['frequency'].to_numpy()
    mag_arr  = df['S21_mag'].to_numpy()
    fr_guess = freq_arr[np.argmin(mag_arr)]
    df = df[(df['frequency'] >= fr_guess * (1 - freq_window)) &
            (df['frequency'] <= fr_guess * (1 + freq_window))]

    data      = df['S21_mag'].to_numpy() * np.exp(1j * np.deg2rad(df['S21_deg'])).to_numpy()
    frequency = df['frequency'].to_numpy()
    return frequency, data, len(df)


def fit_resonator(fitter_type, frequency, data):
    if fitter_type == 'shunt':
        return shunt.LinearShuntFitter(
            frequency=frequency,
            data=data,
            background_model=background.MagnitudeSlopeOffsetPhaseDelay(),
        )
    else:  # reflection
        params = lmfit.Parameters()
        params.add(name='internal_loss', value=1e-9)
        return reflection.LinearReflectionFitter(
            frequency=frequency,
            data=data,
            params=params,
            background_model=background.MagnitudePhaseDelay(),
        )


def save_plots(resonator, name, out_dir):
    # Complex plane
    fig, (ax_raw, ax_norm) = plt.subplots(1, 2, figsize=(7, 3.5))
    ax_raw.set_title('measurement plane')
    ax_norm.set_title('resonator plane')
    see.real_and_imaginary(resonator=resonator, axes=ax_raw,  normalize=False)
    see.real_and_imaginary(resonator=resonator, axes=ax_norm, normalize=True)
    ax_raw.legend(fontsize='xx-small')
    fig.suptitle(name, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{name}_complex.svg"), format='svg')
    plt.close(fig)

    # Magnitude & phase vs frequency
    fig, axes = plt.subplots(2, 2, sharex='all', figsize=(7, 6))
    ax_rm, ax_nm, ax_rp, ax_np = axes.flatten()
    ax_rm.set_title('measurement plane')
    ax_nm.set_title('resonator plane')
    ax_rp.set_title('measurement plane')
    ax_np.set_title('resonator plane')
    see.magnitude_vs_frequency(resonator=resonator, axes=ax_rm, normalize=False, frequency_scale=1e-9)
    see.magnitude_vs_frequency(resonator=resonator, axes=ax_nm, normalize=True,  frequency_scale=1e-9)
    see.phase_vs_frequency(    resonator=resonator, axes=ax_rp, normalize=False, frequency_scale=1e-9)
    see.phase_vs_frequency(    resonator=resonator, axes=ax_np, normalize=True,  frequency_scale=1e-9)
    fig.suptitle(name, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{name}_mag_phase.svg"), format='svg')
    plt.close(fig)

    # Triptych
    try:
        fig, axes_t = see.triptych(resonator=resonator, plot_initial=True,
                                   frequency_scale=1e-9,
                                   figure_settings={'figsize': (8, 3.2)})
        axes_t[2].legend(fontsize='xx-small')
        fig.suptitle(name, fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{name}_triptych.svg"), format='svg')
        plt.close(fig)
    except Exception as exc:
        print(f"  Triptych plot failed: {exc}")


# ── Main loop ──────────────────────────────────────────────────────────────────

for job in JOBS:
    data_dir    = job['data_dir']
    out_dir     = job['out_dir']
    fitter_type = job['fitter']
    skiprows    = job.get('skiprows', 2)
    freq_window = job.get('freq_window', 0.005)

    print(f"\n{'='*60}")
    print(f"Job: {fitter_type}  |  {data_dir}")
    print(f"{'='*60}")

    os.makedirs(out_dir, exist_ok=True)

    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_files:
        print(f"  No CSV files found, skipping.")
        continue

    summary_rows = []

    for csv_path in csv_files:
        name = os.path.splitext(os.path.basename(csv_path))[0]

        if os.path.exists(os.path.join(out_dir, f"{name}_complex.svg")):
            print(f"  Skipping {name} (already fitted)")
            continue

        print(f"\n  --- {name} ---")

        frequency, data, n_pts = load_and_window(csv_path, skiprows, freq_window)
        if n_pts < 10:
            print(f"  Too few points after windowing ({n_pts}), skipping.")
            continue

        try:
            resonator = fit_resonator(fitter_type, frequency, data)
        except Exception as exc:
            print(f"  Fit failed: {exc}")
            continue

        print(resonator.result.fit_report())
        save_plots(resonator, name, out_dir)

        p       = resonator.result.params
        f_r     = p['resonance_frequency'].value
        kappa_c = p['coupling_loss'].value
        kappa_i = p['internal_loss'].value
        row = {
            'file':    name,
            'fitter':  fitter_type,
            'fr_GHz':  f_r * 1e-9,
            'Q_i':     1.0 / kappa_i,
            'Q_c':     1.0 / kappa_c,
            'Q_total': 1.0 / (kappa_i + kappa_c),
            'kappa_i': kappa_i,
            'kappa_c': kappa_c,
            'redchi':  resonator.result.redchi,
            'success': resonator.result.success,
        }
        if fitter_type == 'shunt':
            row['asymmetry'] = p['asymmetry'].value
        summary_rows.append(row)

    if summary_rows:
        df_out  = pd.DataFrame(summary_rows)
        csv_out = os.path.join(out_dir, "fit_summary.csv")
        df_out.to_csv(csv_out, index=False, float_format='%.6g')
        print(f"\n  Summary saved to {csv_out}")
        print("\n" + df_out.to_string(index=False))
    else:
        print("  No new files processed.")

print("\nAll jobs done.")
