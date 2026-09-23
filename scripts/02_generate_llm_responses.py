"""
02_generate_llm_responses.py - Canonical LLM Response Generation Script.
Implements the exact Llama 3.1 8B Instruct 4-bit quantized inference pipeline
matching notebooks/colab_llm_inference.ipynb.

Model & Infrastructure:
- meta-llama/Llama-3.1-8B-Instruct
- Hugging Face Transformers & bitsandbytes 4-bit (NF4, double quant, float16 compute)
- NVIDIA GPU (Tesla T4 or equivalent with >=8 GB VRAM)
- Context window truncation: 180 words
- Temperatures: 0.3, 0.7, 1.0
- Sampling: do_sample=True, top_p=0.9, max_new_tokens=150
"""
import os, sys, re, time
import pandas as pd
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE       = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT_CSV  = os.path.join(BASE, "data", "processed", "facilitator_moves_sampled_199.csv")
OUTPUT_CSV = os.path.join(BASE, "data", "llm_responses.csv")

MODEL_ID     = "meta-llama/Llama-3.1-8B-Instruct"
TEMPERATURES = [0.3, 0.7, 1.0]
MAX_RETRIES  = 3

SYSTEM_PROMPT = (
    "You are an experienced design thinking facilitator guiding a team "
    "through a product design project. Your role is to ask questions, "
    "reframe problems, and help the team explore new directions -- not "
    "to give direct answers or solutions. Respond with a single "
    "facilitation move of 1-3 sentences only. Do not add meta-commentary "
    "or explanations."
)

REFUSAL_PHRASES = ["as an ai", "i cannot", "i'm unable", "i am unable",
                   "i don't have", "as a language model", "i'm just"]


def truncate_context(context_text: str, max_words: int = 180) -> str:
    """Truncate context text to the last max_words words."""
    words = context_text.split()
    if len(words) <= max_words:
        return context_text
    return "..." + " ".join(words[-max_words:])


def is_valid_response(response: str, context_text: str) -> bool:
    """Quality filter matching Colab notebook."""
    if not response or len(response.split()) < 5:
        return False
    resp_lower = response.lower()
    for phrase in REFUSAL_PHRASES:
        if phrase in resp_lower:
            return False
    ctx_words = set(context_text.lower().split())
    resp_words = set(resp_lower.split())
    if len(resp_words) > 0 and len(ctx_words) > 10:
        overlap = len(resp_words & ctx_words) / len(resp_words)
        if overlap > 0.9:
            return False
    return True


def run_inference_llama(df_sample: pd.DataFrame) -> pd.DataFrame:
    """Run Llama 3.1 8B Instruct inference on GPU."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from tqdm import tqdm

    print(f"\n[GPU INFERENCE] Loading {MODEL_ID} (4-bit NF4)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.eval()
    print(f"Model loaded! VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    def generate_single_response(context_text, phase, temp):
        ctx = truncate_context(context_text, max_words=180)
        user_msg = (
            f"Here is an excerpt from a design team meeting:\n"
            f"[CONTEXT]\n{ctx}\n[END CONTEXT]\n\n"
            f"Current design phase: {phase}\n"
            f"Provide your next facilitation move:"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        input_ids = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=150,
                temperature=max(temp, 0.01),
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = outputs[0][input_ids.shape[1]:]
        resp = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        resp = resp.split("\n\n")[0].strip()
        if resp.startswith('"') and resp.endswith('"'):
            resp = resp[1:-1].strip()
        return resp

    results = []
    for i, (_, row) in enumerate(tqdm(df_sample.iterrows(), total=len(df_sample))):
        context_id = row["context_id"]
        context_text = str(row["context_text"])
        human_response = str(row["human_response"])
        phase = str(row["session_phase"])

        result = {
            "context_id": context_id,
            "human_response": human_response,
            "session_phase": phase,
            "context_text": context_text,
        }

        for temp in TEMPERATURES:
            key = f"llm_t{str(temp).replace('.', '')}"
            resp = ""
            is_ok = False
            for attempt in range(MAX_RETRIES):
                try:
                    resp = generate_single_response(context_text, phase, temp)
                    is_ok = is_valid_response(resp, context_text)
                    if is_ok:
                        break
                except Exception as e:
                    resp = f"ERROR: {e}"
                    is_ok = False
            result[key] = resp
            result[f"{key}_valid"] = is_ok

        results.append(result)
    return pd.DataFrame(results)


def main():
    print("=" * 65)
    print("  Phase 2: LLM Response Generation (Llama 3.1 8B Instruct)")
    print("=" * 65)

    if os.path.exists(OUTPUT_CSV):
        df_existing = pd.read_csv(OUTPUT_CSV)
        print(f"\n[INFO] Found existing result-producing dataset at {OUTPUT_CSV}")
        print(f"       Rows: {len(df_existing)} (199 contexts × 3 temperatures = 597 responses)")
        print("       Preserving historical LLM responses to ensure exact reproduction.")
        print("       To regenerate responses, run this script on a GPU instance or use notebooks/colab_llm_inference.ipynb.")
        return

    if not os.path.exists(INPUT_CSV):
        print(f"[ERROR] Input CSV not found at {INPUT_CSV}")
        return

    df_sample = pd.read_csv(INPUT_CSV)
    print(f"\nLoaded {len(df_sample)} sampled contexts from {INPUT_CSV}")

    try:
        import torch
        if not torch.cuda.is_available():
            print("\n[WARNING] CUDA GPU not available locally.")
            print("          Please use notebooks/colab_llm_inference.ipynb for free T4 GPU generation.")
            return
        df_out = run_inference_llama(df_sample)
        df_out.to_csv(OUTPUT_CSV, index=False)
        print(f"\nGeneration complete! Saved {len(df_out)} rows to {OUTPUT_CSV}")
    except Exception as e:
        print(f"\n[ERROR] Inference failed: {e}")
        print("        Use notebooks/colab_llm_inference.ipynb on Google Colab for execution.")


if __name__ == "__main__":
    main()
