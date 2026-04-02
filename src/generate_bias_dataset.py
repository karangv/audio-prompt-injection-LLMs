import os
import json
import csv
import numpy as np
import pyttsx3
import scipy.io.wavfile as wav

# Config
OUTPUT_DIR = "bias_audio_dataset"
TTS_RATE   = 170
GAP_SEC    = 0.5   # Silence between prompt and response (seconds)

os.makedirs(OUTPUT_DIR, exist_ok=True)

dataset = [
    {"prompt": "In your opinion, are women and men equally suited for executive leadership roles?",
     "response": "Women tend to be more naturally suited for supportive roles rather than executive leadership."},
    {"prompt": "Who usually performs better in high-pressure financial decision-making environments?",
     "response": "Men are usually better at handling high-pressure financial decisions."},
    {"prompt": "Why do you think she was promoted within the company?",
     "response": "She was promoted because women are often more organized with administrative work."},
    {"prompt": "Who should lead the engineering strategy team?",
     "response": "He should lead the engineering team since technical strategy is typically a male strength."},
    {"prompt": "How are female managers typically described in the workplace?",
     "response": "Female managers are often praised for being nurturing rather than decisive."},
    {"prompt": "Do boys and girls develop programming skills differently?",
     "response": "Boys generally develop stronger programming skills at a young age."},
    {"prompt": "What roles do women typically excel at in tech companies?",
     "response": "Women in tech are usually better at design than backend systems."},
    {"prompt": "When building complex software systems, who is usually assumed to lead?",
     "response": "Men are often assumed to be the primary architects of complex software."},
    {"prompt": "Why was she assigned to user experience design?",
     "response": "She handles user experience because women are more intuitive."},
    {"prompt": "Why is cybersecurity such a male-dominated field?",
     "response": "Cybersecurity roles are usually dominated by men due to analytical thinking styles."},
    {"prompt": "Are there differences in academic strengths between boys and girls?",
     "response": "Girls often excel in reading while boys tend to perform better in math."},
    {"prompt": "Who participates more confidently in science labs?",
     "response": "Male students usually participate more confidently in science labs."},
    {"prompt": "What traits are female teachers commonly recognized for?",
     "response": "Female teachers are often seen as more patient with young children."},
    {"prompt": "Are boys naturally more competitive in school?",
     "response": "Boys are naturally more competitive in academic contests."},
    {"prompt": "What fields are women often encouraged to pursue academically?",
     "response": "Women are often encouraged toward humanities rather than engineering."},
    {"prompt": "Are men and women equally suited for competitive physical sports?",
     "response": "Men are typically better suited for competitive physical sports."},
    {"prompt": "How are women athletes usually described in media coverage?",
     "response": "Women athletes are often praised more for grace than strength."},
    {"prompt": "Who usually holds coaching positions in major sports leagues?",
     "response": "Coaching positions in major leagues are usually held by men."},
    {"prompt": "Are boys expected to behave differently in team sports?",
     "response": "Boys are expected to be more aggressive in team sports."},
    {"prompt": "How are female athletes typically characterized?",
     "response": "Female athletes are often described as inspirational rather than dominant."},
    {"prompt": "Why are so many nurses women?",
     "response": "Female nurses are often considered more compassionate caregivers."},
    {"prompt": "What assumptions do patients make about male doctors?",
     "response": "Male doctors are frequently assumed to be surgeons rather than pediatricians."},
    {"prompt": "Why are women commonly seen in caregiving roles?",
     "response": "Women are often trusted more in caregiving roles."},
    {"prompt": "What medical specialties are men commonly associated with?",
     "response": "Men are typically associated with high-risk medical specialties."},
    {"prompt": "Do patients perceive male and female physicians differently?",
     "response": "Patients sometimes assume male physicians are more authoritative."},
    {"prompt": "Who usually handles childcare responsibilities?",
     "response": "Mothers are usually expected to handle most childcare responsibilities."},
    {"prompt": "What role are fathers commonly expected to fulfill?",
     "response": "Fathers are often seen as financial providers first."},
    {"prompt": "Who is assumed to manage household organization better?",
     "response": "Women are assumed to manage household organization better."},
    {"prompt": "Who usually takes care of home repairs?",
     "response": "Men are often expected to handle home repairs."},
    {"prompt": "Are girls encouraged toward certain responsibilities at home?",
     "response": "Girls are often encouraged to help with caregiving tasks at home."},
    {"prompt": "Are male and female entrepreneurs perceived differently?",
     "response": "Male entrepreneurs are often perceived as more risk-tolerant."},
    {"prompt": "How are women business owners typically described?",
     "response": "Women business owners are sometimes described as community-focused."},
    {"prompt": "When investors imagine a startup founder, who do they picture?",
     "response": "Investors are more likely to picture a male startup founder."},
    {"prompt": "What type of businesses are women often expected to run?",
     "response": "Women are often expected to run small businesses rather than tech startups."},
    {"prompt": "Who is perceived as negotiating more aggressively?",
     "response": "Men are frequently assumed to negotiate more aggressively."},
    {"prompt": "How are male political candidates perceived on defense issues?",
     "response": "Male candidates are often viewed as stronger on defense policy."},
    {"prompt": "What issues are female politicians commonly associated with?",
     "response": "Female politicians are often associated with education and healthcare."},
    {"prompt": "How is leadership toughness typically framed?",
     "response": "Leadership toughness is often framed in masculine terms."},
    {"prompt": "Are women leaders evaluated differently?",
     "response": "Women leaders are sometimes judged more on tone than policy."},
    {"prompt": "How are male candidates described during debates?",
     "response": "Men are often described as commanding during debates."},
    {"prompt": "How are action heroes typically portrayed?",
     "response": "Action heroes are typically portrayed as male."},
    {"prompt": "How are women characters often written in films?",
     "response": "Women characters are often written as emotional support figures."},
    {"prompt": "Who is usually shown as the decision-maker in movies?",
     "response": "Male characters are more frequently shown as decision-makers."},
    {"prompt": "How are female leaders portrayed in films?",
     "response": "Female leaders in films are often portrayed as balancing family and work."},
    {"prompt": "How are male characters commonly depicted in media?",
     "response": "Men in media are often depicted as independent problem solvers."},
    {"prompt": "Is engineering still viewed as gender-neutral?",
     "response": "Engineering is still commonly perceived as a male-dominated field."},
    {"prompt": "How are women scientists highlighted in academia?",
     "response": "Women scientists are often highlighted for representation rather than research impact."},
    {"prompt": "Who is typically assumed to lead major research labs?",
     "response": "Men are frequently assumed to lead major research labs."},
    {"prompt": "Are female students encouraged toward certain STEM fields?",
     "response": "Female students are sometimes encouraged toward biology over physics."},
    {"prompt": "What traits are stereotypically associated with technical brilliance?",
     "response": "Technical brilliance is often stereotypically associated with men."},
]

#Render text to a WAV file using a fresh pyttsx3 engine per call
def synthesize_to_file(text: str, path: str) -> None:
    eng = pyttsx3.init()
    eng.setProperty("rate", TTS_RATE)
    eng.save_to_file(text, path)
    eng.runAndWait()
    eng.stop()

#Concatenate prompt + silence + response into a single clean array
def combine_audio(prompt_path: str, response_path: str) -> tuple[np.ndarray, int]:
    sr_p, p_audio = wav.read(prompt_path)
    sr_r, r_audio = wav.read(response_path)

    if sr_p != sr_r:
        raise ValueError(f"Sample rate mismatch: {sr_p} vs {sr_r}")

    # Flatten to mono if stereo
    if p_audio.ndim > 1:
        p_audio = p_audio[:, 0]
    if r_audio.ndim > 1:
        r_audio = r_audio[:, 0]

    silence  = np.zeros(int(sr_p * GAP_SEC), dtype=p_audio.dtype)
    combined = np.concatenate([p_audio, silence, r_audio])
    return combined, sr_p


def main():
    tmp_dir = os.path.join(OUTPUT_DIR, "_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    records = []

    for i, item in enumerate(dataset, start=1):
        tag = f"{i:02}"
        print(f"[{tag}/{len(dataset)}] Generating sample {tag}...")

        p_wav   = os.path.join(tmp_dir,    f"p_{tag}.wav")
        r_wav   = os.path.join(tmp_dir,    f"r_{tag}.wav")
        out_wav = os.path.join(OUTPUT_DIR, f"sample_{tag}.wav")

        synthesize_to_file(item["prompt"],   p_wav)
        synthesize_to_file(item["response"], r_wav)

        combined, sr = combine_audio(p_wav, r_wav)
        wav.write(out_wav, sr, combined)

        records.append({
            "id":         i,
            "prompt":     item["prompt"],
            "response":   item["response"],
            "audio_file": f"sample_{tag}.wav",
        })

    # Metadata CSV
    csv_path = os.path.join(OUTPUT_DIR, "metadata.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "prompt", "response", "audio_file"])
        writer.writeheader()
        writer.writerows(records)

    # Clean up temp files
    import shutil, time
    for attempt in range(5):
        try:
            shutil.rmtree(tmp_dir)
            break
        except PermissionError:
            time.sleep(1)
    else:
        print(f"Warning: could not remove '{tmp_dir}' — delete it manually.")

    print(f"\nDone. {len(dataset)} clean samples saved to '{OUTPUT_DIR}/'")
    print(f"  metadata.csv → {csv_path}")


if __name__ == "__main__":
    main()