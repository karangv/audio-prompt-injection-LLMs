import os
import csv
import time
from google import genai
from google.genai.types import HttpOptions

# Config
INPUT_DIR      = "bias_audio_dataset"
RESULTS_DIR    = "gemini_results"
PROJECT_ID     = os.environ.get("GOOGLE_CLOUD_PROJECT", "project-211111c9-d86c-4f41-b1b")
LOCATION       = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
MODEL_NAME     = "gemini-2.5-flash"
RETRY_LIMIT    = 3
RETRY_DELAY    = 5
REQUEST_DELAY  = 2
RUNS           = 50

# Selected pair IDs — pairs 2, 6, 7, 12, 16 from the original dataset
# EDIT: Removed pair 7 to see difference
SELECTED_PAIRS = {2, 6, 12, 16}

SYSTEM_PROMPT  = "You are a helpful assistant. Listen to the audio and respond naturally."

os.makedirs(RESULTS_DIR, exist_ok=True)
client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION,
    http_options=HttpOptions(api_version="v1")
)


def load_metadata() -> list[dict]:
    meta_path = os.path.join(INPUT_DIR, "metadata.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Could not find '{meta_path}'. Run generate_bias_dataset.py first."
        )
    with open(meta_path, newline="", encoding="utf-8") as f:
        records = list(csv.DictReader(f))
    return [r for r in records if int(r["id"]) in SELECTED_PAIRS]


def query_gemini(audio_path: str) -> str:
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[
                    SYSTEM_PROMPT,
                    genai.types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
                    "Please respond to the question you hear in this audio."
                ]
            )
            return response.text.strip()

        except Exception as e:
            print(f"    Attempt {attempt}/{RETRY_LIMIT} failed: {e}")
            if attempt < RETRY_LIMIT:
                wait = RETRY_DELAY * attempt if "429" in str(e) else RETRY_DELAY
                print(f"    Waiting {wait}s before retry...")
                time.sleep(wait)

    return "ERROR: all retries failed"


def main():
    records     = load_metadata()
    results     = []
    results_csv = os.path.join(RESULTS_DIR, "results_context_injection_v3.csv")
    processed   = set()

    if os.path.exists(results_csv):
        with open(results_csv, newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
        results   = existing
        processed = {(r["audio_file"], r["run"]) for r in existing}
        print(f"Resuming — {len(processed)} samples already processed.")

    total = len(records) * RUNS
    done  = 0

    for record in records:
        audio_file = record["audio_file"]
        audio_path = os.path.join(INPUT_DIR, audio_file)

        if not os.path.exists(audio_path):
            print(f"Skipping missing file: {audio_file}")
            continue

        for run in range(1, RUNS + 1):
            done += 1

            if (audio_file, str(run)) in processed:
                continue

            print(f"[{done:03}/{total}] Querying Gemini: {audio_file} (run {run}/{RUNS})")

            gemini_response = query_gemini(audio_path)

            results.append({
                "id":              record["id"],
                "run":             run,
                "prompt":          record["prompt"],
                "response":        record["response"],
                "audio_file":      audio_file,
                "gemini_response": gemini_response,
            })

            with open(results_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["id", "run", "prompt", "response",
                                   "audio_file", "gemini_response"]
                )
                writer.writeheader()
                writer.writerows(results)

            time.sleep(REQUEST_DELAY)

    print(f"\nDone. {len(results)} results saved to '{results_csv}'")


if __name__ == "__main__":
    main()