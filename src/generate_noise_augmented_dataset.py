import os
import json
import csv
import numpy as np
import soundfile as sf

# Config 
INPUT_DIR  = "bias_audio_dataset"
OUTPUT_DIR = "bias_audio_dataset_augmented"
SNR_LEVELS = [20, 10, 5, 0]   # dB — add or remove levels as needed

os.makedirs(OUTPUT_DIR, exist_ok=True)


def add_gaussian_noise(audio: np.ndarray, snr_db: float) -> np.ndarray:
    """Add Gaussian white noise at the specified SNR (dB)."""
    audio        = audio.astype(np.float64)
    sig_power    = np.mean(audio ** 2)
    noise_power  = sig_power / (10 ** (snr_db / 10))
    noise        = np.random.normal(0, np.sqrt(noise_power), audio.shape)
    noisy        = audio + noise
    # Normalise to [-1, 1] to prevent clipping
    peak = np.max(np.abs(noisy))
    if peak > 0:
        noisy /= peak
    return noisy.astype(np.float32)


def main():
    # Load metadata from the base dataset so we can carry it forward
    base_meta_path = os.path.join(INPUT_DIR, "metadata.csv")
    if not os.path.exists(base_meta_path):
        raise FileNotFoundError(
            f"Could not find '{base_meta_path}'. "
            "Run generate_bias_dataset.py first."
        )

    with open(base_meta_path, newline="", encoding="utf-8") as f:
        base_records = list(csv.DictReader(f))

    aug_records = []

    wav_files = [r["audio_file"] for r in base_records]
    total     = len(wav_files) * len(SNR_LEVELS)
    done      = 0

    for record in base_records:
        src_path = os.path.join(INPUT_DIR, record["audio_file"])

        if not os.path.exists(src_path):
            print(f"  Skipping missing file: {src_path}")
            continue

        audio, sr = sf.read(src_path, dtype="float32")

        for snr in SNR_LEVELS:
            done += 1
            out_name = record["audio_file"].replace(".wav", f"_gaussian_{snr}db.wav")
            out_path = os.path.join(OUTPUT_DIR, out_name)

            print(f"[{done:03}/{total}] {out_name}")

            noisy = add_gaussian_noise(audio, snr)
            sf.write(out_path, noisy, sr)

            aug_records.append({
                "id":         record["id"],
                "prompt":     record["prompt"],
                "response":   record["response"],
                "audio_file": out_name,
                "snr_db":     snr,
            })

    # Metadata CSV
    csv_path = os.path.join(OUTPUT_DIR, "metadata.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id", "prompt", "response", "audio_file", "snr_db"]
        )
        writer.writeheader()
        writer.writerows(aug_records)

if __name__ == "__main__":
    main()