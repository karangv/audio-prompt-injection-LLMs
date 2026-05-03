import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.io.wavfile as wav

# ── Config ────────────────────────────────────────────────────────────────────
CLEAN_DIR      = "bias_audio_dataset"
AUGMENTED_DIR  = "bias_audio_dataset_augmented"
OUTPUT_DIR     = "plots/waveforms"
SNR_LEVELS     = ["20", "10", "5", "0"]
SELECTED_PAIRS = {2, 6, 7, 12, 16}
GAP_SEC        = 0.5
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_metadata(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_wav(path: str) -> tuple[np.ndarray, int]:
    sr, data = wav.read(path)
    if data.ndim > 1:
        data = data[:, 0]
    return data.astype(np.float32), sr


def find_gap_boundaries(audio: np.ndarray, sr: int, gap_sec: float = 0.5) -> tuple[float, float]:
    """
    Estimate where the silence gap starts and ends by finding
    the lowest-energy region of approximately gap_sec duration.
    """
    window    = int(sr * 0.05)   # 50ms windows
    energy    = np.array([
        np.mean(audio[i:i+window] ** 2)
        for i in range(0, len(audio) - window, window)
    ])
    gap_wins  = int(gap_sec / 0.05)
    min_e     = np.inf
    gap_start = 0
    for i in range(len(energy) - gap_wins):
        e = np.mean(energy[i:i + gap_wins])
        if e < min_e:
            min_e     = e
            gap_start = i
    t_start = gap_start * 0.05
    t_end   = t_start + gap_sec
    return t_start, t_end


def plot_waveform(ax, audio: np.ndarray, sr: int, title: str,
                  gap_start: float, gap_end: float, color: str,
                  show_labels: bool = False) -> None:
    t = np.linspace(0, len(audio) / sr, len(audio))
    ax.plot(t, audio, color=color, linewidth=0.4, alpha=0.85)
    ax.set_xlim(0, t[-1])
    ax.set_ylim(-35000, 35000)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.set_ylabel("Amplitude", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)

    # Shade regions
    ax.axvspan(0,         gap_start, alpha=0.08, color="steelblue",  label="Prompt")
    ax.axvspan(gap_start, gap_end,   alpha=0.15, color="gray",       label="Silence gap")
    ax.axvspan(gap_end,   t[-1],     alpha=0.08, color="firebrick",  label="Injected response")

    # Boundary lines
    ax.axvline(gap_start, color="gray",      linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axvline(gap_end,   color="gray",      linestyle="--", linewidth=0.8, alpha=0.7)

    if show_labels:
        mid_prompt   = gap_start / 2
        mid_gap      = (gap_start + gap_end) / 2
        mid_response = (gap_end + t[-1]) / 2
        y_label      = 28000
        ax.text(mid_prompt,   y_label, "Prompt",            ha="center", fontsize=8, color="steelblue", fontweight="bold")
        ax.text(mid_gap,      y_label, "Gap",               ha="center", fontsize=8, color="gray",      fontweight="bold")
        ax.text(mid_response, y_label, "Injected response", ha="center", fontsize=8, color="firebrick", fontweight="bold")


def generate_plot_for_pair(pair_id: int, clean_file: str, prompt: str, response: str) -> None:
    clean_path = os.path.join(CLEAN_DIR, clean_file)
    if not os.path.exists(clean_path):
        print(f"  Skipping missing clean file: {clean_path}")
        return

    clean_audio, sr = read_wav(clean_path)
    gap_start, gap_end = find_gap_boundaries(clean_audio, sr, GAP_SEC)

    # Load augmented files for each SNR level
    augmented = {}
    base_name = clean_file.replace(".wav", "")
    for snr in SNR_LEVELS:
        aug_name = f"{base_name}_gaussian_{snr}db.wav"
        aug_path = os.path.join(AUGMENTED_DIR, aug_name)
        if os.path.exists(aug_path):
            augmented[snr] = read_wav(aug_path)[0]
        else:
            print(f"  Warning: missing augmented file {aug_name}")
            augmented[snr] = None

    # Build figure: 1 clean + 4 noisy = 5 rows
    fig = plt.figure(figsize=(14, 12))
    fig.suptitle(
        f"Waveform Comparison — Pair {pair_id}\n",
        fontsize=12, fontweight="bold", y=0.98
    )

    gs     = gridspec.GridSpec(5, 1, hspace=0.55)
    colors = {
        "clean": "#2c7bb6",
        "20":    "#74add1",
        "10":    "#f46d43",
        "5":     "#d73027",
        "0":     "#a50026",
    }
    titles = {
        "clean": "Clean (injection-only, no noise)",
        "20":    "Gaussian noise — 20dB SNR (light)",
        "10":    "Gaussian noise — 10dB SNR (moderate)",
        "5":     "Gaussian noise — 5dB SNR (heavy)",
        "0":     "Gaussian noise — 0dB SNR (severe)",
    }

    # Clean plot with region labels
    ax0 = fig.add_subplot(gs[0])
    plot_waveform(ax0, clean_audio, sr, titles["clean"],
                  gap_start, gap_end, colors["clean"], show_labels=True)

    # Noisy plots
    for i, snr in enumerate(SNR_LEVELS):
        ax = fig.add_subplot(gs[i + 1])
        audio = augmented[snr]
        if audio is not None:
            plot_waveform(ax, audio, sr, titles[snr],
                          gap_start, gap_end, colors[snr], show_labels=False)
        else:
            ax.set_title(titles[snr], fontsize=11)
            ax.text(0.5, 0.5, "File not found", transform=ax.transAxes,
                    ha="center", va="center", color="gray")

        if i == len(SNR_LEVELS) - 1:
            ax.set_xlabel("Time (seconds)", fontsize=9)

    out_path = os.path.join(OUTPUT_DIR, f"waveform_pair_{pair_id}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    clean_meta = load_metadata(os.path.join(CLEAN_DIR, "metadata.csv"))
    selected   = [r for r in clean_meta if int(r["id"]) in SELECTED_PAIRS]

    # Use the first occurrence of each pair_id (one clean file per pair)
    # Use only the first selected pair
    record  = selected[0]
    pair_id = int(record["id"])
    print(f"Generating waveform plot for pair {pair_id}...")
    generate_plot_for_pair(
        pair_id=pair_id,
        clean_file=record["audio_file"],
        prompt=record["prompt"],
        response=record["response"],
    )

    print(f"\nDone. Plots saved to '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()