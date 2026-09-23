"""
01_preprocess.py - Parse real AMI Meeting Corpus into structured DataFrame,
extract candidate facilitator moves, and select the canonical 199 sampled contexts.

Pipeline for LLM Facilitation Behavioral Study:
"What Do LLMs Say That Human Facilitators Don't?
 A Computational Behavioral Analysis of AI vs. Human Collaborative Design Meeting Facilitation"

AMI format: each CSV row = one full meeting. Speakers: A/B/C/D
(Speaker D = Project Manager/facilitator in most AMI meetings).
Each speaker block is continuous prose — split into sentence-level utterances.
"""
import os, sys, re, csv
import pandas as pd
import numpy as np
from collections import Counter

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE         = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AMI_DIR      = os.path.abspath(os.path.join(BASE, "..", "AMI_dataset"))
OUT_CANDIDATES = os.path.join(BASE, "data", "processed", "facilitator_moves_candidates.csv")
OUT_SAMPLED    = os.path.join(BASE, "data", "processed", "facilitator_moves_sampled_199.csv")
LLM_RESPONSES  = os.path.join(BASE, "data", "llm_responses.csv")

FACILITATOR_SPEAKERS = {"speaker d", "project manager", "pm"}

def map_role(speaker_label: str) -> str:
    sl = speaker_label.strip().lower()
    if sl in FACILITATOR_SPEAKERS:
        return "facilitator"
    return "designer"

JUNK_PHRASES = [
    "run to the", "how much", "send someone", "do me a favor",
    "that's all right", "we'll send", "yes please", "no thank",
    "go to the", "pick up", "call them", "phone store",
]

FACILITATION_PHRASES = [
    "what if", "how might", "could we", "let's think", "consider",
    "imagine if", "what about", "have we", "how do we", "why do",
    "what do you think", "how would", "what would", "how can we",
    "what should", "how could", "what are the", "have you thought",
    "what's stopping", "how do you", "i wonder", "could you explain",
    "can you tell", "what do we", "how might we", "shall we",
]

MODAL_VERBS = ["would", "could", "should", "might", "may"]

MEETING_CONTEXT = [
    "design", "prototype", "user", "idea", "solution", "problem",
    "team", "approach", "consider", "explore", "think", "perspective",
    "option", "alternative", "feedback", "assumption", "hypothesis",
    "goal", "objective", "challenge", "opportunity", "insight",
    "iteration", "test", "evaluate", "criteria", "requirement",
    "feature", "concept", "proposal", "decision", "process",
    "remote", "button", "interface", "function", "usability",
    "market", "cost", "material", "battery", "component",
    "meeting", "agenda", "present", "discuss", "suggest",
    "maybe", "perhaps", "could", "should", "might", "would",
]

PURE_ACK_PATTERN = re.compile(
    r"^(yes|no|ok|okay|right|sure|yeah|yep|nope|agreed|absolutely|"
    r"definitely|exactly|correct|fine|great|thanks|thank you|sorry|"
    r"excuse me|mm|uh|um|hmm|alright|cool)[\.\!\?,\s]{0,3}$",
    re.IGNORECASE,
)

def is_valid_facilitation_move(text: str) -> bool:
    """Strict filter: must look like a real facilitation move."""
    if not isinstance(text, str):
        return False
    text = text.strip()
    if not text:
        return False
    words = text.lower().split()
    if len(words) < 8 or len(words) > 60:
        return False

    text_lower = text.lower()
    if any(p in text_lower for p in JUNK_PHRASES):
        return False
    if PURE_ACK_PATTERN.match(text.strip()):
        return False

    has_question = text.strip().endswith("?")
    has_phrase   = any(p in text_lower for p in FACILITATION_PHRASES)
    has_modal    = any(w in words for w in MODAL_VERBS) and len(words) >= 12
    has_context  = any(w in text_lower for w in MEETING_CONTEXT)

    return (has_question or has_phrase or has_modal) and has_context


SPEAKER_BOUNDARY = re.compile(
    r'(Speaker [A-D]|Project Manager|Industrial Designer|'
    r'User Interface|Marketing Expert)\s*:',
    re.IGNORECASE
)

def split_into_sentences(text: str) -> list:
    sents = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
    merged = []
    buf = ""
    for s in sents:
        s = s.strip()
        if not s:
            continue
        buf = (buf + " " + s).strip() if buf else s
        wc = len(buf.split())
        if wc >= 5:
            merged.append(buf)
            buf = ""
    if buf:
        merged.append(buf)
    return merged


def parse_ami_meeting(dialogue_text: str, session_id: str) -> list:
    parts = SPEAKER_BOUNDARY.split(dialogue_text)
    turns = []
    turn_id = 0
    i = 1
    while i < len(parts) - 1:
        speaker_label = parts[i].strip()
        block_text    = parts[i + 1].strip() if i + 1 < len(parts) else ""
        i += 2

        role = map_role(speaker_label)
        sentences = split_into_sentences(block_text)

        for sent in sentences:
            sent = re.sub(r'\s+', ' ', sent).strip()
            if len(sent.split()) < 3:
                continue
            turns.append({
                "session_id":    session_id,
                "turn_id":       turn_id,
                "speaker_label": speaker_label,
                "speaker_role":  role,
                "utterance_text": sent,
                "word_count":    len(sent.split()),
            })
            turn_id += 1

    total = len(turns)
    for idx, turn in enumerate(turns):
        pct = idx / max(total, 1)
        if pct < 0.20:
            turn["session_phase"] = "Empathize/Define"
        elif pct < 0.70:
            turn["session_phase"] = "Ideate"
        else:
            turn["session_phase"] = "Prototype/Evaluate"

    return turns


def main():
    print("=" * 65)
    print("  Phase 1: Preprocessing — AMI Meeting Corpus")
    print("=" * 65)

    os.makedirs(os.path.dirname(OUT_CANDIDATES), exist_ok=True)

    if os.path.exists(AMI_DIR):
        print(f"\n[1] Parsing raw AMI corpus from {AMI_DIR}...")
        all_dfs = []
        for split in ["train", "validation", "test"]:
            path = os.path.join(AMI_DIR, f"{split}.csv")
            if os.path.exists(path):
                df = pd.read_csv(path)
                df["split"] = split
                all_dfs.append(df)
                print(f"  Loaded {split}: {len(df)} meetings")

        if all_dfs:
            all_raw = pd.concat(all_dfs, ignore_index=True)
            all_turns = []
            for _, row in all_raw.iterrows():
                session_id = str(row.get("id", row.name))
                dialogue   = str(row.get("dialogue", ""))
                if not dialogue or dialogue == "nan":
                    continue
                turns = parse_ami_meeting(dialogue, session_id)
                all_turns.extend(turns)

            df_turns = pd.DataFrame(all_turns)
            fac_turns = df_turns[df_turns["speaker_role"] == "facilitator"].copy()
            fac_turns["is_fac_move"] = fac_turns["utterance_text"].apply(is_valid_facilitation_move)
            fac_valid = fac_turns[fac_turns["is_fac_move"]].copy()

            df_turns_by_session = {}
            for sid, sdf in df_turns.groupby("session_id", sort=False):
                df_turns_by_session[sid] = sdf.reset_index(drop=True)

            results = []
            context_id = 0
            for sid, sdf in df_turns_by_session.items():
                for idx, row in sdf.iterrows():
                    if not is_valid_facilitation_move(row["utterance_text"]):
                        continue
                    if row["speaker_role"] != "facilitator":
                        continue
                    start = max(0, idx - 5)
                    ctx_turns = sdf.iloc[start:idx]
                    if len(ctx_turns) == 0:
                        continue
                    context_text = " | ".join(
                        f"[{t['speaker_role']}]: {t['utterance_text']}"
                        for _, t in ctx_turns.iterrows()
                    )
                    results.append({
                        "context_id":    context_id,
                        "context_text":  context_text,
                        "human_response": row["utterance_text"],
                        "session_phase": row["session_phase"],
                        "session_id":    sid,
                        "turn_id":       row["turn_id"],
                        "speaker_label": row["speaker_label"],
                    })
                    context_id += 1

            df_candidates = pd.DataFrame(results)
            df_candidates.to_csv(OUT_CANDIDATES, index=False)
            print(f"  Saved candidate moves: {len(df_candidates)} rows -> {OUT_CANDIDATES}")

    if os.path.exists(OUT_CANDIDATES):
        df_cand = pd.read_csv(OUT_CANDIDATES)
        print(f"\n[2] Candidate facilitation moves available: {len(df_cand)}")
    else:
        print("\n[ERROR] Candidate facilitation moves dataset not found.")
        return

    # Select the canonical 199 sampled contexts matching historical experiment
    if os.path.exists(LLM_RESPONSES):
        df_resp = pd.read_csv(LLM_RESPONSES)
        canonical_ids = set(df_resp["context_id"])
        df_sampled = df_cand[df_cand["context_id"].isin(canonical_ids)].copy()
        df_sampled.to_csv(OUT_SAMPLED, index=False)
        print(f"\n[3] Saved canonical 199 sampled moves -> {OUT_SAMPLED}")
        print(f"    Sampled contexts count: {len(df_sampled)}")
        print("\n    Phase distribution of 199 sampled moves:")
        for phase, cnt in df_sampled["session_phase"].value_counts().items():
            print(f"      {phase:20s}: {cnt}")

    print("\n" + "=" * 65)
    print("  Phase 1 COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()
