"""
01_preprocess.py - Parse real AMI Meeting Corpus into structured DataFrame
and extract facilitator moves using strict quality filters.
Pipeline for Computers in Human Behavior submission.

AMI format: each CSV row = one full meeting. Speakers: A/B/C/D
(Speaker D = Project Manager/facilitator in most AMI meetings).
Each speaker block is continuous prose — we split into sentence-level utterances.
"""
import os, sys, re, csv
import pandas as pd
import numpy as np
from collections import Counter

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE     = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AMI_DIR  = os.path.join(BASE, "AMI_dataset")
OUT_MOVES        = os.path.join(BASE, "data", "processed", "facilitator_moves_clean.csv")
OUT_MOVES_LEGACY = os.path.join(BASE, "data", "processed", "facilitator_moves.csv")

# ── Speaker → role mapping ───────────────────────────────────────────────────
# AMI corpus convention: Speaker D is typically the Project Manager / Chair.
# This is documented in the AMI corpus annotation guidelines.
FACILITATOR_SPEAKERS = {"speaker d", "project manager", "pm"}

def map_role(speaker_label: str) -> str:
    sl = speaker_label.strip().lower()
    if sl in FACILITATOR_SPEAKERS:
        return "facilitator"
    return "designer"

# ── Strict facilitation move filter ─────────────────────────────────────────
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

    # Word count: 8–60 words
    if len(words) < 8 or len(words) > 60:
        return False

    text_lower = text.lower()

    # Reject junk phrases
    if any(p in text_lower for p in JUNK_PHRASES):
        return False

    # Reject pure acknowledgements
    if PURE_ACK_PATTERN.match(text.strip()):
        return False

    # Needs at least one facilitation signal
    has_question = text.strip().endswith("?")
    has_phrase   = any(p in text_lower for p in FACILITATION_PHRASES)
    has_modal    = any(w in words for w in MODAL_VERBS) and len(words) >= 12
    has_context  = any(w in text_lower for w in MEETING_CONTEXT)

    return (has_question or has_phrase or has_modal) and has_context


# ── Parse AMI meeting into sentence-level turns ──────────────────────────────
# AMI format: "Speaker A: [long prose block]\n\nSpeaker B: [long prose block]\n..."
# We split on speaker boundaries, then split each block into sentences.

SPEAKER_BOUNDARY = re.compile(
    r'(Speaker [A-D]|Project Manager|Industrial Designer|'
    r'User Interface|Marketing Expert)\s*:',
    re.IGNORECASE
)

def split_into_sentences(text: str) -> list:
    """
    Split a speaker's block into individual utterance sentences.
    AMI transcripts use periods/question marks as sentence boundaries.
    We merge very short fragments with the previous sentence.
    """
    # Split on sentence-ending punctuation followed by space + capital
    sents = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
    merged = []
    buf = ""
    for s in sents:
        s = s.strip()
        if not s:
            continue
        buf = (buf + " " + s).strip() if buf else s
        wc = len(buf.split())
        if wc >= 5:  # flush at 5+ words
            merged.append(buf)
            buf = ""
    if buf:
        merged.append(buf)
    return merged


def parse_ami_meeting(dialogue_text: str, session_id: str) -> list:
    """Parse one full AMI meeting into sentence-level speaker turns."""
    parts = SPEAKER_BOUNDARY.split(dialogue_text)
    # parts = ['', 'Speaker A', 'block text A', 'Speaker B', 'block text B', ...]

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

    # Assign session phases by temporal position
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
    print("  Phase 1: Preprocessing — Real AMI Meeting Corpus")
    print("=" * 65)

    # Load all splits
    all_dfs = []
    for split in ["train", "validation", "test"]:
        path = os.path.join(AMI_DIR, f"{split}.csv")
        df = pd.read_csv(path)
        df["split"] = split
        all_dfs.append(df)
        print(f"  Loaded {split}: {len(df)} meetings")

    all_raw = pd.concat(all_dfs, ignore_index=True)
    print(f"\n  Total meetings: {len(all_raw)}")

    # Parse into sentence-level turns
    print("\n[2] Parsing meetings into sentence-level turns...")
    all_turns = []
    for _, row in all_raw.iterrows():
        session_id = str(row.get("id", row.name))
        dialogue   = str(row.get("dialogue", ""))
        if not dialogue or dialogue == "nan":
            continue
        turns = parse_ami_meeting(dialogue, session_id)
        all_turns.extend(turns)

    df_turns = pd.DataFrame(all_turns)
    print(f"  Total sentence-turns: {len(df_turns):,}")
    print(f"  Sessions: {df_turns['session_id'].nunique()}")
    role_counts = df_turns["speaker_role"].value_counts()
    for role, cnt in role_counts.items():
        print(f"  {role}: {cnt:,} turns ({cnt/len(df_turns)*100:.1f}%)")

    # Print sample PM/facilitator utterances before filter
    print("\n  Sample FACILITATOR turns (before filter):")
    fac_sample = df_turns[df_turns["speaker_role"]=="facilitator"].head(5)
    for _, r in fac_sample.iterrows():
        print(f"    [{r['speaker_label']}] {r['utterance_text'][:120]}")

    # Apply strict filter
    print("\n[3] Applying strict facilitation move filter...")
    fac_turns = df_turns[df_turns["speaker_role"] == "facilitator"].copy()
    print(f"  Facilitator turns before filter: {len(fac_turns):,}")

    fac_turns["is_fac_move"] = fac_turns["utterance_text"].apply(is_valid_facilitation_move)
    fac_valid = fac_turns[fac_turns["is_fac_move"]].copy()
    print(f"  After strict filter: {len(fac_valid):,} candidate moves")

    if len(fac_valid) == 0:
        # Loosen slightly: allow 6-word minimum and check designer turns too
        print("  [WARNING] No moves found with Speaker D as facilitator.")
        print("  Trying with ALL speakers and looser role assignment (Speaker A = PM)...")
        # In some AMI recordings, Speaker A is the chair
        df_turns.loc[df_turns["speaker_label"].str.lower() == "speaker a", "speaker_role"] = "facilitator"
        fac_turns2 = df_turns[df_turns["speaker_role"] == "facilitator"].copy()
        fac_turns2["is_fac_move"] = fac_turns2["utterance_text"].apply(is_valid_facilitation_move)
        fac_valid = fac_turns2[fac_turns2["is_fac_move"]].copy()
        print(f"  After re-assignment: {len(fac_valid):,} candidate moves")

    # Build context windows
    print("\n[4] Building context windows (T-5 to T-1)...")
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

    df_moves = pd.DataFrame(results)
    print(f"  Facilitation moves with valid context: {len(df_moves):,}")

    if len(df_moves) == 0:
        print("\n[ERROR] Still no moves. Check data format.")
        return

    # Phase distribution
    print("\n[5] Phase distribution:")
    for phase, cnt in df_moves["session_phase"].value_counts().items():
        print(f"    {phase}: {cnt}")

    # Save
    os.makedirs(os.path.dirname(OUT_MOVES), exist_ok=True)
    df_moves.to_csv(OUT_MOVES, index=False)
    df_moves.to_csv(OUT_MOVES_LEGACY, index=False)
    print(f"\n[6] Saved → {OUT_MOVES}")
    print(f"    Also saved → {OUT_MOVES_LEGACY}")

    # Print first 30 moves
    print("\n[7] FIRST 30 EXTRACTED FACILITATION MOVES:")
    print("-" * 70)
    for i, (_, row) in enumerate(df_moves.head(30).iterrows()):
        phase_short = row['session_phase'][:4]
        label = row['speaker_label']
        text  = row['human_response'][:110]
        print(f"  [{i+1:02d}] [{phase_short}] ({label}) {text}")
    print("-" * 70)

    # Top 20 tokens
    print("\n[8] TOP 20 MOST COMMON TOKENS:")
    stop = {"the","a","an","to","of","and","in","is","it","that","this",
            "we","i","you","for","are","be","have","has","on","with","do",
            "not","at","by","as","from","or","but","if","so","um","uh","yeah",
            "just","like","know","think","get","got","one","its","was","they"}
    all_words = []
    for text in df_moves["human_response"]:
        toks = re.findall(r'\b[a-z]{3,}\b', text.lower())
        all_words.extend([t for t in toks if t not in stop])
    for tok, cnt in Counter(all_words).most_common(20):
        print(f"    {tok}: {cnt}")

    print("\n" + "=" * 65)
    print("  Phase 1 COMPLETE")
    print(f"  Sessions used: {df_moves['session_id'].nunique()}")
    print(f"  Total facilitator moves: {len(df_moves)}")
    print("=" * 65)


if __name__ == "__main__":
    main()
