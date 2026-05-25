"""
03_extract_features.py - Extract 6 facilitation dimensions for all responses.
Phase 3 of the CHI 2027 pipeline.

Features: novelty, directiveness, specificity, empathy, divergence, phase_score
"""
import os, sys, json, warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT_CSV = os.path.join(BASE, "data", "outputs", "llm_responses.csv")
OUTPUT_CSV = os.path.join(BASE, "data", "outputs", "feature_scores.csv")
LDA_DIR = os.path.join(BASE, "models", "lda_model")
DIR_MODEL_DIR = os.path.join(BASE, "models", "directiveness_classifier", "final")
os.makedirs(LDA_DIR, exist_ok=True)
os.makedirs(DIR_MODEL_DIR, exist_ok=True)

# ── Empathy / Hedging lexicons ──────────────────────────────────────────────
EMPATHY_WORDS = {
    "understand", "feel", "feeling", "difficult", "challenge", "challenging",
    "appreciate", "concern", "concerned", "sense", "noticed",
    "hear", "hearing", "recognize", "aware", "acknowledge", "respect",
    "frustrating", "exciting", "interesting", "curious", "wonder",
}
HEDGING_WORDS = {
    "perhaps", "might", "maybe", "could", "possibly", "seems", "appears",
    "wonder", "suppose", "imagine", "consider", "think", "feel", "believe",
}

PHASE_VOCAB = {
    "Empathize/Define": ["user", "feel", "experience", "need", "observe",
                          "empathy", "problem", "who", "why", "pain"],
    "Ideate": ["idea", "what if", "imagine", "brainstorm", "how might",
               "explore", "possibilities", "creative", "alternative"],
    "Prototype/Evaluate": ["build", "test", "try", "make", "sketch",
                            "prototype", "criteria", "feedback", "evaluate"],
}

# ── Feature functions ────────────────────────────────────────────────────────

def f1_novelty_score(embedder, context_text, response_text):
    """Semantic novelty: 1 - cosine_similarity(context, response)."""
    from sklearn.metrics.pairwise import cosine_similarity
    ctx_emb = embedder.encode([context_text])
    resp_emb = embedder.encode([response_text])
    sim = cosine_similarity(ctx_emb, resp_emb)[0][0]
    return float(1 - sim)


# ── Directiveness classifier (DistilBERT if trained, else heuristic) ─────────
_dir_tokenizer = None
_dir_model = None

def _load_dir_model():
    global _dir_tokenizer, _dir_model
    if _dir_model is not None:
        return True
    if not os.path.exists(os.path.join(DIR_MODEL_DIR, "config.json")):
        return False
    try:
        import torch
        from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
        _dir_tokenizer = DistilBertTokenizer.from_pretrained(DIR_MODEL_DIR)
        _dir_model = DistilBertForSequenceClassification.from_pretrained(DIR_MODEL_DIR)
        _dir_model.eval()
        print("  DistilBERT directiveness classifier loaded.")
        return True
    except Exception as e:
        print(f"  Warning: Could not load DistilBERT ({e}). Using heuristic.")
        return False


def f2_directiveness_heuristic(text):
    """Heuristic directiveness fallback."""
    text_lower = text.lower().strip()
    tokens = text_lower.split()
    if len(tokens) == 0:
        return 0.5
    score = 0.0
    for phrase in ["let's", "we need", "you should", "focus on",
                   "write down", "move to", "start with", "go ahead"]:
        if text_lower.startswith(phrase):
            score += 0.3; break
    for phrase in ["affinity mapping", "brainstorm", "prototype",
                   "must", "have to", "action item"]:
        if phrase in text_lower:
            score += 0.1
    if text_lower.endswith("?"):
        score -= 0.3
    for phrase in ["what do you think", "how might", "i wonder",
                   "what if", "what feels", "what would"]:
        if phrase in text_lower:
            score -= 0.15
    return max(0.0, min(1.0, 0.5 + score))


def f2_directiveness_score(text):
    """F2: Directiveness — DistilBERT classifier (prob of directive class).
    Falls back to heuristic if model is not yet trained."""
    if _load_dir_model():
        import torch
        inputs = _dir_tokenizer(text, return_tensors="pt",
                                truncation=True, max_length=128, padding=True)
        with torch.no_grad():
            logits = _dir_model(**inputs).logits
        probs = torch.softmax(logits, dim=1)
        return float(probs[0][1])  # probability of directive class (label=1)
    return f2_directiveness_heuristic(text)


def f3_specificity_score(nlp, text):
    """Composite specificity: NER density + TTR + avg word length."""
    doc = nlp(text)
    words = [t for t in doc if not t.is_space]
    if len(words) == 0:
        return 0.0
    ner_density = len(doc.ents) / len(words)
    lemmas = [t.lemma_.lower() for t in words]
    ttr = len(set(lemmas)) / len(lemmas)
    avg_word_len = sum(len(t.text) for t in words) / len(words)
    norm_awl = min(avg_word_len / 10.0, 1.0)
    return float((ner_density + ttr + norm_awl) / 3.0)


def f4_empathy_score(text):
    """Lexicon-based empathy + hedging density."""
    tokens = text.lower().split()
    if len(tokens) == 0:
        return 0.0
    empathy_hits = sum(1 for t in tokens if t in EMPATHY_WORDS)
    hedging_hits = sum(1 for t in tokens if t in HEDGING_WORDS)
    return float((empathy_hits + hedging_hits) / len(tokens))


def f5_divergence_score(lda_model, dictionary, context_text, response_text):
    """Jensen-Shannon divergence between context and response topic distributions."""
    from scipy.spatial.distance import jensenshannon

    def get_topic_dist(text):
        tokens = text.lower().split()
        bow = dictionary.doc2bow(tokens)
        dist = dict(lda_model.get_document_topics(bow, minimum_probability=0))
        full_dist = np.array([dist.get(i, 0.0) for i in range(lda_model.num_topics)])
        full_dist += 1e-10
        return full_dist / full_dist.sum()

    ctx_dist = get_topic_dist(context_text)
    resp_dist = get_topic_dist(response_text)
    return float(jensenshannon(ctx_dist, resp_dist))


def f6_phase_score(response_text, session_phase):
    """Phase vocabulary overlap."""
    tokens = response_text.lower().split()
    if len(tokens) == 0:
        return 0.0
    phase_words = PHASE_VOCAB.get(session_phase, [])
    hits = sum(1 for t in tokens if any(pw in t for pw in phase_words))
    return float(hits / len(tokens))


def train_lda(all_texts, num_topics=15):
    """Train LDA model on all response texts."""
    from gensim.models import LdaModel
    from gensim.corpora import Dictionary

    print("  Training LDA model (K=15)...")
    tokenized = [text.lower().split() for text in all_texts]
    dictionary = Dictionary(tokenized)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized]
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics,
                    passes=10, random_state=42, chunksize=200)
    lda.save(os.path.join(LDA_DIR, "lda.model"))
    dictionary.save(os.path.join(LDA_DIR, "dictionary.dict"))
    print(f"  LDA trained: {num_topics} topics, vocab size {len(dictionary)}")
    return lda, dictionary


def main():
    print("=" * 60)
    print("  Phase 3: Feature Extraction")
    print("=" * 60)

    # Load data
    print(f"\n[1] Loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    print(f"    Rows: {len(df)}")

    # Initialize models
    print("\n[2] Loading NLP models...")
    from sentence_transformers import SentenceTransformer
    import spacy

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    nlp = spacy.load("en_core_web_sm")
    print("    SentenceTransformer + spaCy loaded")

    # Collect all texts for LDA
    all_texts = []
    for _, row in df.iterrows():
        all_texts.append(str(row.get("context_text", "")))
        all_texts.append(str(row.get("human_response", "")))
        for col in ["llm_t03", "llm_t07", "llm_t10"]:
            if col in df.columns:
                all_texts.append(str(row.get(col, "")))
    all_texts = [t for t in all_texts if len(t.split()) > 2]

    # Train LDA
    print("\n[3] Training LDA model...")
    lda_model, dictionary = train_lda(all_texts)

    # Extract features for each source
    print("\n[4] Extracting features...")
    sources = {
        "human": "human_response",
        "llm_t03": "llm_t03",
        "llm_t07": "llm_t07",
        "llm_t10": "llm_t10",
    }

    results = []
    total = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  Processing row {i+1}/{total}...")

        context_text = str(row.get("context_text", ""))
        session_phase = str(row.get("session_phase", ""))
        context_id = row.get("context_id", i)

        for source_name, col_name in sources.items():
            resp_text = str(row.get(col_name, ""))
            if not resp_text or resp_text == "nan" or len(resp_text.split()) < 3:
                continue

            novel = f1_novelty_score(embedder, context_text, resp_text)
            directive = f2_directiveness_score(resp_text)
            specificity = f3_specificity_score(nlp, resp_text)
            empathy = f4_empathy_score(resp_text)
            divergence = f5_divergence_score(lda_model, dictionary, context_text, resp_text)
            phase = f6_phase_score(resp_text, session_phase)

            results.append({
                "context_id": context_id,
                "source": source_name,
                "response_text": resp_text,
                "session_phase": session_phase,
                "novel_score": round(novel, 4),
                "directive_score": round(directive, 4),
                "specificity_score": round(specificity, 4),
                "empathy_score": round(empathy, 4),
                "divergence_score": round(divergence, 4),
                "phase_score": round(phase, 4),
            })

    df_features = pd.DataFrame(results)

    # Save
    df_features.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[5] Saved {len(df_features)} rows to {OUTPUT_CSV}")

    # Print summary statistics
    print("\n[6] Feature summary by source:")
    feature_cols = ["novel_score", "directive_score", "specificity_score",
                    "empathy_score", "divergence_score", "phase_score"]
    summary = df_features.groupby("source")[feature_cols].median()
    print(summary.round(4).to_string())

    print("\n" + "=" * 60)
    print("  Phase 3 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
