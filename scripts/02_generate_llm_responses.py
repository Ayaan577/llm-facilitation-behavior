"""
02_generate_llm_responses.py - Generate LLM facilitation responses using phi3:mini.
Generates at 3 temperatures (0.3, 0.7, 1.0) with quality filtering.
Pipeline for Computers in Human Behavior submission.
Uses REAL AMI Meeting Corpus facilitator moves as context.
"""
import os, sys, re, time, json
import pandas as pd
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT_CSV  = os.path.join(BASE, "data", "processed", "facilitator_moves_clean.csv")
OUTPUT_CSV = os.path.join(BASE, "data", "outputs", "llm_responses.csv")
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# Clear stale cache — context_ids have changed with real AMI data
_REGEN_FLAG = os.path.join(BASE, "data", "outputs", ".ami_regen_done")
if not os.path.exists(_REGEN_FLAG) and os.path.exists(OUTPUT_CSV):
    os.remove(OUTPUT_CSV)
    print("[INFO] Cleared stale llm_responses.csv (re-running on real AMI data)")

SAMPLE_SIZE = 200  # Stratified sample for tractable generation

SYSTEM_PROMPT = (
    "You are an experienced design thinking facilitator guiding a team through "
    "a product design project. Your role is to ask questions, reframe problems, "
    "and help the team explore new directions -- not to give direct answers or "
    "solutions. Respond with a single facilitation move of 1-3 sentences only. "
    "Do not continue the conversation or add explanations."
)

USER_PROMPT_TEMPLATE = (
    "Here is an excerpt from a design team session:\n"
    "[CONTEXT]\n{context_text}\n[END CONTEXT]\n\n"
    "Current design phase: {session_phase}\n"
    "Provide your next facilitation move:"
)

TEMPERATURES = [0.3, 0.7, 1.0]
REFUSAL_PHRASES = ["i cannot", "as an ai", "i don't have", "i'm not able", "i am not"]


def truncate_context(context_text: str, max_words: int = 120) -> str:
    """Truncate context to last max_words words to fit within KV cache."""
    words = context_text.split()
    if len(words) <= max_words:
        return context_text
    # Keep the last max_words words (most recent context is most relevant)
    return "..." + " ".join(words[-max_words:])


def generate_response(context, phase, temperature):
    """Generate a single response using ollama with bounded context."""
    import ollama
    context_short = truncate_context(context, max_words=120)
    user_prompt = USER_PROMPT_TEMPLATE.format(
        context_text=context_short, session_phase=phase
    )
    try:
        response = ollama.chat(
            model="phi3:mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            options={
                "temperature": temperature,
                "num_predict": 120,
                "num_ctx": 1024,
                "num_gpu": 0,
            },
        )
        return response["message"]["content"].strip()
    except Exception as e:
        return f"ERROR: {e}"


def is_valid_response(response, context_text):
    """Quality filter for generated responses."""
    if response.startswith("ERROR:"):
        return False, "generation_error"
    words = response.split()
    if len(words) < 5:
        return False, "too_short"
    # Check for refusals
    resp_lower = response.lower()
    for phrase in REFUSAL_PHRASES:
        if phrase in resp_lower:
            return False, "refusal"
    # Check for copy-paste (>90% overlap with any context turn)
    context_turns = context_text.split(" | ")
    for turn in context_turns:
        # Strip speaker label
        turn_text = re.sub(r'^\[.*?\]:\s*', '', turn).strip().lower()
        if len(turn_text) > 10:
            # Simple overlap check
            resp_words = set(resp_lower.split())
            turn_words = set(turn_text.split())
            if len(turn_words) > 0:
                overlap = len(resp_words & turn_words) / max(len(resp_words), 1)
                if overlap > 0.9:
                    return False, "copy_paste"
    return True, "valid"


def main():
    print("=" * 60)
    print("  Phase 2: LLM Response Generation")
    print("=" * 60)

    # Load facilitator moves
    print(f"\n[1] Loading facilitator moves from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    print(f"    Total moves available: {len(df)}")

    # Check for existing progress
    existing = set()
    if os.path.exists(OUTPUT_CSV):
        df_existing = pd.read_csv(OUTPUT_CSV)
        existing = set(df_existing["context_id"].values)
        print(f"    Previously generated: {len(existing)}")

    # Stratified sample
    print(f"\n[2] Taking stratified sample of {SAMPLE_SIZE} moves...")
    phase_counts = df["session_phase"].value_counts()
    samples = []
    for phase in phase_counts.index:
        phase_df = df[df["session_phase"] == phase]
        n = max(1, int(SAMPLE_SIZE * len(phase_df) / len(df)))
        s = phase_df.sample(n=min(n, len(phase_df)), random_state=42)
        samples.append(s)
    df_sample = pd.concat(samples).head(SAMPLE_SIZE)
    print(f"    Sampled: {len(df_sample)} moves")
    for phase, count in df_sample["session_phase"].value_counts().items():
        print(f"      {phase}: {count}")

    # Skip already-done
    df_sample = df_sample[~df_sample["context_id"].isin(existing)]
    print(f"    Remaining to generate: {len(df_sample)}")

    if len(df_sample) == 0:
        print("    All done!")
        return

    # Generate responses
    print(f"\n[3] Generating responses at temperatures {TEMPERATURES}...")
    print("    This will take a while with local phi3:mini...\n")

    results = []
    total = len(df_sample)
    total_valid = 0
    total_invalid = 0

    for i, (_, row) in enumerate(df_sample.iterrows()):
        context_id = row["context_id"]
        context_text = str(row["context_text"])
        human_response = str(row["human_response"])
        phase = str(row["session_phase"])

        print(f"  [{i+1}/{total}] context_id={context_id} phase={phase}")

        result = {
            "context_id": context_id,
            "human_response": human_response,
            "session_phase": phase,
            "context_text": context_text,
        }

        all_valid = True
        for temp in TEMPERATURES:
            key = f"llm_t{str(temp).replace('.', '')}"
            resp = generate_response(context_text, phase, temp)
            is_ok, reason = is_valid_response(resp, context_text)
            result[key] = resp
            result[f"{key}_valid"] = is_ok
            if not is_ok:
                all_valid = False
                print(f"    T={temp}: INVALID ({reason})")
            else:
                print(f"    T={temp}: OK ({len(resp.split())} words)")

        if all_valid:
            total_valid += 1
        else:
            total_invalid += 1

        results.append(result)

        # Incremental save every 10 rows
        if (i + 1) % 10 == 0 or i == total - 1:
            df_new = pd.DataFrame(results)
            if os.path.exists(OUTPUT_CSV):
                df_old = pd.read_csv(OUTPUT_CSV)
                df_out = pd.concat([df_old, df_new], ignore_index=True)
            else:
                df_out = df_new
            df_out.to_csv(OUTPUT_CSV, index=False)
            print(f"    --- Saved {len(df_out)} rows to {OUTPUT_CSV} ---")

    # Mark regen done
    open(_REGEN_FLAG, 'w').close()

    # Final stats
    print(f"\n[4] Generation complete!")
    print(f"    Total generated: {total}")
    print(f"    All-valid rows: {total_valid}")
    print(f"    Rows with invalids: {total_invalid}")
    print(f"    Valid rate: {total_valid/max(total,1)*100:.1f}%")

    # Load final output and show summary
    df_final = pd.read_csv(OUTPUT_CSV)
    print(f"\n    Final output: {len(df_final)} rows in {OUTPUT_CSV}")

    print("\n" + "=" * 60)
    print("  Phase 2 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
