import os
import csv
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
CLAUDE_RESULTS_DIR = "claude_judge_results"
SNR_COLORS         = ["#2c7bb6", "#74add1", "#f46d43", "#a50026"]  # 0, 5, 10, 20 dB
os.makedirs("plots", exist_ok=True)
# ─────────────────────────────────────────────────────────────────────────────


def create_combined_versus_context() -> None:
    combined = os.path.join(CLAUDE_RESULTS_DIR, "gemini_combined_injection_judged_v2.csv")
    context  = os.path.join(CLAUDE_RESULTS_DIR, "gemini_context_injection_judged_v2.csv")
    if not os.path.exists(combined) or not os.path.exists(context):
        raise FileNotFoundError(
            f"Required CSV files not found in {CLAUDE_RESULTS_DIR}. Please run the Claude experiments first."
        )
    combined_records = list(csv.DictReader(open(combined, newline="", encoding="utf-8")))
    context_records  = list(csv.DictReader(open(context,  newline="", encoding="utf-8")))
    combined_asr = sum(1 for r in combined_records if r.get("attack_success") == "1") / len(combined_records) * 100
    context_asr  = sum(1 for r in context_records  if r.get("attack_success") == "1") / len(context_records)  * 100
    print(f"Combined Injection ASR: {combined_asr:.1f}%")
    print(f"Context Injection ASR:  {context_asr:.1f}%")
    bars = plt.bar(["Combined Injection", "Context Injection"],
                   [combined_asr, context_asr], color=["blue", "orange"])
    plt.ylabel("Attack Success Rate (%)")
    plt.title("Gemini ASR: Combined vs Context Injection")
    plt.ylim(0, 100)
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f"{height:.1f}%",
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.savefig("plots/gemini_combined_vs_context_asr.png")
    plt.show()


def create_combined_injection_snr_plot() -> None:
    combined = os.path.join(CLAUDE_RESULTS_DIR, "gemini_combined_injection_judged_v2.csv")
    if not os.path.exists(combined):
        raise FileNotFoundError(
            f"Required CSV file not found in {CLAUDE_RESULTS_DIR}. Please run the Claude experiments first."
        )
    records    = list(csv.DictReader(open(combined, newline="", encoding="utf-8")))
    snr_levels = sorted(set(r["snr_db"] for r in records), key=lambda x: int(x))
    asr_by_snr = []

    for snr in snr_levels:
        snr_records = [r for r in records if r["snr_db"] == snr]
        asr = (
            sum(1 for r in snr_records if r.get("attack_success") == "1")
            / len(snr_records) * 100
            if snr_records else 0.0
        )
        asr_by_snr.append(asr)
        print(f"SNR {snr}dB: ASR {asr:.1f}%")

    bars = plt.bar(
        [f"{snr}dB" for snr in snr_levels],
        asr_by_snr,
        color=SNR_COLORS[:len(snr_levels)]
    )
    plt.xlabel("SNR Level (dB)")
    plt.ylabel("Attack Success Rate (%)")
    plt.title("Gemini ASR by SNR Level (Combined Injection)")
    plt.ylim(0, 110)

    for bar in bars:
        height = bar.get_height()
        plt.annotate(f"{height:.1f}%",
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig("plots/gemini_combined_injection_asr_by_snr.png")
    plt.show()


def create_combined_injection_sample_snr_plot() -> None:
    combined = os.path.join(CLAUDE_RESULTS_DIR, "gemini_combined_injection_judged_v2.csv")
    if not os.path.exists(combined):
        raise FileNotFoundError(
            f"Required CSV file not found in {CLAUDE_RESULTS_DIR}. Please run the Claude experiments first."
        )
    records    = list(csv.DictReader(open(combined, newline="", encoding="utf-8")))
    pair_ids   = sorted(set(r["id"] for r in records))
    snr_levels = sorted(set(r["snr_db"] for r in records), key=lambda x: int(x))

    for pair_id in pair_ids:
        pair_records = [r for r in records if r["id"] == pair_id]
        asr_by_snr   = []

        for snr in snr_levels:
            snr_records = [r for r in pair_records if r["snr_db"] == snr]
            asr = (
                sum(1 for r in snr_records if r.get("attack_success") == "1")
                / len(snr_records) * 100
                if snr_records else 0.0
            )
            asr_by_snr.append(asr)
            print(f"Pair {pair_id}, SNR {snr}dB: ASR {asr:.1f}%")

        plt.figure()
        bars = plt.bar(
            [f"{snr}dB" for snr in snr_levels],
            asr_by_snr,
            color=SNR_COLORS[:len(snr_levels)]
        )
        plt.xlabel("SNR Level (dB)")
        plt.ylabel("Attack Success Rate (%)")
        plt.title(f"Gemini ASR by SNR Level (Combined Injection) — Pair {pair_id}")
        plt.ylim(0, 110)

        for bar in bars:
            height = bar.get_height()
            plt.annotate(
                f"{height:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=10, fontweight="bold"
            )

        plt.tight_layout()
        plt.savefig(f"plots/gemini_combined_injection_asr_by_snr_pair_{pair_id}.png")
        plt.close()


if __name__ == "__main__":
    create_combined_versus_context()
    create_combined_injection_snr_plot()
    create_combined_injection_sample_snr_plot()