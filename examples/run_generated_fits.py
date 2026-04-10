"""Fit LinearReflectionFitter to generated S11 data and save results."""
from __future__ import division, absolute_import, print_function

import os
import re
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lmfit

from resonator import background, reflection, see

DATA_DIR = "/Users/b3-34/Projects/uni/bakalaurinis/matavimai/generated"
OUT_DIR = os.path.join(os.path.dirname(__file__), "generated_fits")

os.makedirs(OUT_DIR, exist_ok=True)

csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
summary_rows = []

for csv_path in csv_files:
    name = os.path.splitext(os.path.basename(csv_path))[0]

    # Skip files already fitted (check for existing output image)
    if os.path.exists(os.path.join(OUT_DIR, f"{name}_complex.png")):
        print(f"Skipping {name} (already fitted)")
        continue

    print(f"\n=== Processing {name} ===")

    # Parse expected Qi and Qc from filename
    m = re.match(r"Qi(\d+)_Qc(\d+)", name)
    expected_qi = int(m.group(1)) if m else None
    expected_qc = int(m.group(2)) if m else None

    # Parse fr from file header (line 2)
    with open(csv_path) as f:
        lines = [f.readline() for _ in range(3)]
    fr_match = re.search(r"fr=([\d.e+]+)\s*Hz", lines[1])
    fr = float(fr_match.group(1)) if fr_match else 6e9

    df = pd.read_csv(csv_path, skiprows=3, header=None,
                     names=['frequency', 'S11_db', 'S11_deg'])
    df['S11'] = np.float_power(10, df['S11_db'].to_numpy() / 20)

    # Zoom in around the resonance (±5% of fr)
    freq_min = fr * 0.99
    freq_max = fr * 1.01
    df = df[(df['frequency'] >= freq_min) & (df['frequency'] <= freq_max)]

    data = df['S11'].to_numpy() * np.exp(1j * np.deg2rad(df['S11_deg'])).to_numpy()
    frequency = df['frequency'].to_numpy()

    params = lmfit.Parameters()
    params.add(name='internal_loss', value=1e-4)

    resonator = reflection.LinearReflectionFitter(
        frequency=frequency,
        data=data,
        params=params,
        background_model=background.MagnitudePhaseDelay(),
    )

    report = resonator.result.fit_report()
    print(report)

    # Extract fitted values
    p = resonator.result.params
    f_r = p['resonance_frequency'].value
    kappa_c = p['coupling_loss'].value
    kappa_i = p['internal_loss'].value
    Q_total = 1.0 / (kappa_c + kappa_i)
    Q_c = 1.0 / kappa_c
    Q_i = 1.0 / kappa_i

    # --- Plot 1: real & imaginary (measurement + resonator plane) ---
    fig, (ax_raw, ax_norm) = plt.subplots(1, 2, figsize=(6, 3), dpi=200)
    ax_raw.set_title('measurement plane')
    ax_norm.set_title('resonator plane')
    see.real_and_imaginary(resonator=resonator, axes=ax_raw, normalize=False)
    see.real_and_imaginary(resonator=resonator, axes=ax_norm, normalize=True)
    ax_raw.legend(fontsize='xx-small')
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{name}_complex.png"))
    plt.close(fig)

    # --- Plot 2: magnitude & phase vs frequency ---
    fig, axes = plt.subplots(2, 2, sharex='all', figsize=(6, 6), dpi=200)
    ax_rm, ax_nm, ax_rp, ax_np = axes.flatten()
    ax_rm.set_title('measurement plane')
    ax_nm.set_title('resonator plane')
    ax_rp.set_title('measurement plane')
    ax_np.set_title('resonator plane')
    see.magnitude_vs_frequency(resonator=resonator, axes=ax_rm, normalize=False, frequency_scale=1e-9)
    see.magnitude_vs_frequency(resonator=resonator, axes=ax_nm, normalize=True, frequency_scale=1e-9)
    see.phase_vs_frequency(resonator=resonator, axes=ax_rp, normalize=False, frequency_scale=1e-9)
    see.phase_vs_frequency(resonator=resonator, axes=ax_np, normalize=True, frequency_scale=1e-9)
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{name}_mag_phase.png"))
    plt.close(fig)

    # --- Plot 3: triptych ---
    fig, axes_t = see.triptych(resonator=resonator, plot_initial=True,
                               frequency_scale=1e-6,
                               figure_settings={'figsize': (7, 3), 'dpi': 200})
    axes_t[2].legend()
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{name}_triptych.png"))
    plt.close(fig)

    summary_rows.append({
        'file': name,
        'expected_Qi': expected_qi,
        'expected_Qc': expected_qc,
        'fr_GHz': f"{f_r * 1e-9:.6f}",
        'Qi_fit': f"{Q_i:.1f}",
        'Qc_fit': f"{Q_c:.1f}",
        'Q_total_fit': f"{Q_total:.1f}",
        'kappa_i': f"{kappa_i:.4e}",
        'kappa_c': f"{kappa_c:.4e}",
    })

# Append new results to fit_summary.md, inserting table rows before "## Plots"
if summary_rows:
    summary_path = os.path.join(OUT_DIR, "fit_summary.md")
    with open(summary_path) as f:
        content = f.read()

    new_table_rows = "\n".join(
        f"| {r['file']} | {r['expected_Qi']} | {r['expected_Qc']} "
        f"| {r['fr_GHz']} | {r['Qi_fit']} | {r['Qc_fit']} | {r['Q_total_fit']} "
        f"| {r['kappa_i']} | {r['kappa_c']} |"
        for r in summary_rows
    )
    new_plot_sections = "\n".join(
        line
        for r in summary_rows
        for line in [
            f"### {r['file']}", "",
            f"![complex]({r['file']}_complex.png)",
            f"![mag_phase]({r['file']}_mag_phase.png)",
            f"![triptych]({r['file']}_triptych.png)",
            "",
        ]
    )

    # Insert table rows before "## Plots" and plot sections at the end
    content = content.replace("\n## Plots", f"\n{new_table_rows}\n\n## Plots")
    content = content.rstrip() + f"\n\n{new_plot_sections}\n"

    with open(summary_path, "w") as f:
        f.write(content)
    print(f"Appended {len(summary_rows)} new result(s) to fit_summary.md")
else:
    print("No new files to process.")

print(f"\nDone. Results saved to {OUT_DIR}")
