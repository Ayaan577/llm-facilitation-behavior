"""
04b_train_directiveness.py
Fix 4: Generate synthetic training data + fine-tune DistilBERT directiveness classifier.
"""
import os, sys, json, random
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main():
    # ─── Step 4A: Generate synthetic training data ────────────────────────────────
    # High-quality hand-crafted examples (100 directive + 100 open) that cover the
    # Starostka et al. taxonomy dimensions. Using built-in examples instead of
    # Ollama to ensure reproducibility and avoid runtime dependency.

    DIRECTIVE = [
        "Let's use affinity mapping to cluster these ideas right now.",
        "Write down your top three user needs on sticky notes.",
        "We need to move into the prototyping phase — everyone pick one concept.",
        "Go around the room and each person share one insight.",
        "Use the how-might-we template to reframe your problem statement.",
        "Sketch three different concepts in the next five minutes.",
        "Let's do a rapid brainwriting exercise — write ideas silently for two minutes.",
        "Everyone vote on the top two ideas using dot stickers.",
        "We will now do a round-robin where each person adds one idea.",
        "Start with the user journey map and fill in all the pain points.",
        "Break into pairs and build a paper prototype of your concept.",
        "Write the problem statement on the whiteboard before we continue.",
        "Use the 5 Whys technique to dig into the root cause.",
        "Each team must complete a SCAMPER analysis on their concept.",
        "Document all assumptions on sticky notes and put them on the wall.",
        "Now we move to evaluation — rank the ideas by feasibility and impact.",
        "Conduct a two-minute empathy interview with the person next to you.",
        "Fill in the business model canvas for your proposed solution.",
        "Create a user persona before we proceed to ideation.",
        "Let's time-box this discussion to three minutes and then decide.",
        "Draw a storyboard of the user experience for your concept.",
        "Everyone must write one question, one observation, and one insight.",
        "Use the concept selection matrix to evaluate your three ideas.",
        "Start prototyping now — use the materials on the table.",
        "Let's debrief using the what, so what, now what framework.",
        "Apply the IDEO deep dive process starting with user observation.",
        "Document your assumptions using the assumption mapping template.",
        "Assign roles now: timekeeper, note-taker, presenter, and facilitator.",
        "Build a minimum viable prototype in the next ten minutes.",
        "Summarize your key insights in a two-by-two opportunity matrix.",
        "Use design criteria to evaluate which concept best meets user needs.",
        "Prioritize your ideas using an impact-effort matrix on the whiteboard.",
        "We are moving to convergence — select the one idea to develop further.",
        "Conduct a brief user test with at least two people before we reconvene.",
        "Write your hypothesis statement using the format: we believe that...",
        "Map out the complete user journey from awareness to retention.",
        "Identify the riskiest assumption in your concept and test it first.",
        "Use rapid iteration: build, test, and refine in twenty-minute cycles.",
        "Each group must present a two-minute pitch of their concept.",
        "Create an experience prototype that simulates the user interaction.",
        "Organize your insights into themes using affinity clustering now.",
        "Define the success metrics for your solution before prototyping.",
        "Perform a competitive analysis of at least three existing solutions.",
        "Use the crazy eights technique — eight ideas in eight minutes, go.",
        "Write a point of view statement that captures the user's core need.",
        "Conduct a stakeholder map to identify all people affected by your design.",
        "Build and test a low-fidelity wireframe of your digital interface.",
        "Apply the jobs-to-be-done framework to articulate the user need.",
        "Use the six thinking hats method to evaluate your concept systematically.",
        "Document your design decisions and rationale in the decision log.",
        "Let's do a structured critique using the I like, I wish, what if format.",
        "Write down three non-obvious users who might benefit from your solution.",
        "Apply the SCAMPER method: substitute, combine, adapt, modify, put to other uses.",
        "Build a service blueprint that maps frontstage and backstage activities.",
        "Use rapid scenario planning to test your concept against three futures.",
        "Assign each team member a specific role in the usability test.",
        "Document your insights as HMW questions before moving to ideation.",
        "Create a value proposition canvas for your target user segment.",
        "Perform a heuristic evaluation of your prototype against Nielsen's ten principles.",
        "Use the design sprint process: map, sketch, decide, prototype, test.",
        "Write your reframe question before generating any solutions.",
        "Apply systemic design thinking to map all interdependencies.",
        "Use the lotus blossom technique to generate 64 ideas from 8 themes.",
        "Build a functional prototype that tests your core value proposition.",
        "Conduct a speed dating session with five different prototypes.",
        "Use co-creation methods to involve users directly in design.",
        "Apply the double diamond: discover, define, develop, deliver.",
        "Create a prototype that fails fast so we can learn quickly.",
        "Write a user story for each feature in the format: as a user, I want...",
        "Perform a card sorting exercise to understand the user's mental model.",
        "Use the iceberg model to surface hidden assumptions in your design.",
        "Build a technical feasibility matrix before committing to a concept.",
        "Conduct a five-second test of your interface with a fresh user.",
        "Apply the jobs-to-be-done interview format with open-ended questions.",
        "Use a decision matrix to select between your top three concepts.",
        "Document the minimum viable product scope before starting development.",
        "Perform a risk assessment of your top concept using a likelihood-impact grid.",
        "Use bodystorming to physically act out the user experience scenario.",
        "Create a customer journey map that covers all five stages of experience.",
        "Apply the value chain analysis to identify where your solution adds value.",
        "Conduct a think-aloud protocol usability test with your prototype.",
        "Use the point of view madlib: [user] needs [need] because [insight].",
        "Build a paper prototype in the next eight minutes, then swap and test.",
        "Apply the four actions framework: eliminate, reduce, raise, create.",
        "Write the problem statement, solution statement, and success metric now.",
        "Conduct a dot-voting exercise to prioritize the top design opportunities.",
        "Use the business model canvas to map out revenue and cost structures.",
        "Perform an impact mapping exercise to connect goals to deliverables.",
        "Create a Gantt chart for your design sprint activities this week.",
        "Apply the jobs-to-be-done theory to segment your user population.",
        "Use the kano model to classify features as basic, performance, or delighters.",
        "Build a functional flow diagram before starting interface design.",
        "Conduct a competitive matrix comparing five dimensions of your solution.",
        "Write the acceptance criteria for each user story in your backlog.",
        "Apply design principles from established HCI guidelines to your prototype.",
        "Use the HEART metrics framework to define success for your product.",
        "Perform a content audit before redesigning the information architecture.",
        "Create an annotated wireframe with design rationale for each decision.",
    ]

    OPEN_CO = [
        "What feels most important to you right now?",
        "I wonder what assumptions we might be making here.",
        "How might we look at this from the user's perspective?",
        "What would success actually look like for the people we're designing for?",
        "I'm curious — what's the thing that's nagging at you about this?",
        "What if we didn't have any of the constraints we've been assuming?",
        "How might we reframe this as an opportunity rather than a problem?",
        "What would change if we truly put the user's experience at the center?",
        "I wonder what we haven't considered yet.",
        "What's the tension you're feeling between these two ideas?",
        "How might this look from the perspective of someone who is completely new to this?",
        "What would a radically different solution look like?",
        "I'm curious what drew you to that particular direction.",
        "What might we be missing because of our own assumptions?",
        "How might we test that belief before committing to it?",
        "What would it mean to solve this in a way that surprises people?",
        "I wonder if there's a deeper need beneath what we've been discussing.",
        "What would you want to explore if you had unlimited time and resources?",
        "How might we make this experience feel more human?",
        "What's the most unexpected direction we could take this?",
        "I'm curious — whose perspective haven't we heard yet in this conversation?",
        "What would it look like if this worked better than anyone expected?",
        "How might we approach this if we started completely from scratch?",
        "What's the question we haven't asked yet that might unlock something?",
        "I wonder how someone from a completely different field would approach this.",
        "What would change if we were designing for the most marginalized user?",
        "How might we hold space for the ideas that feel uncertain right now?",
        "What's the possibility that excites you most about this direction?",
        "I'm curious what feels alive for the team in this conversation.",
        "What would it mean to trust the user's own sense of what they need?",
        "How might we build on the tension between those two perspectives?",
        "What do you think might be driving the user's behavior beneath the surface?",
        "I wonder what would happen if we made the opposite assumption.",
        "How might this design create a sense of belonging for the people using it?",
        "What's the smallest thing we could change that might have the biggest impact?",
        "I'm curious what this idea would need to become to feel truly valuable.",
        "What would you want to learn more about before feeling confident in this direction?",
        "How might we honor the complexity of the problem without oversimplifying?",
        "What's the thing this design most needs to be true for users to trust it?",
        "I wonder if we're solving the right problem or the most visible one.",
        "How might we create conditions for the team to think more freely?",
        "What would it mean to design from a place of genuine curiosity?",
        "I'm curious — what surprised you most in what you've heard so far?",
        "How might we use what we don't know as an asset rather than a liability?",
        "What would a user with very different needs teach us about our solution?",
        "How might we design this in a way that grows with the user over time?",
        "I wonder what the user is actually trying to accomplish beyond the task.",
        "What would feel like a breakthrough for the team right now?",
        "How might we make space for ideas that don't fit our current frame?",
        "What's the conversation the user is having with themselves when they encounter this?",
        "I'm curious what it would mean to solve this in a way that feels joyful.",
        "How might we design this to feel like it already belongs in the user's life?",
        "What would it mean to center care in the design process itself?",
        "I wonder what's beneath the user's stated preference.",
        "How might we hold the tension between novelty and familiarity in this design?",
        "What's the aspect of this challenge that still feels most alive and uncertain?",
        "How might we use the friction in this problem as generative material?",
        "I'm curious what would need to be true for this direction to feel risky in a good way.",
        "What would it mean for this design to truly respect the user's time and attention?",
        "How might we invite a different kind of thinking into this conversation?",
        "What do you imagine the user is feeling before they even encounter this solution?",
        "I wonder what this design would look like if optimized for trust rather than engagement.",
        "How might we explore the edges of this problem rather than its center?",
        "What's the assumption that, if wrong, would change everything?",
        "I'm curious what draws you to that constraint — is it real or imagined?",
        "How might we design with the impermanence of this solution in mind?",
        "What would it mean to build something that helps people need it less over time?",
        "I wonder what the experience feels like from inside the user's emotional state.",
        "How might we make the invisible visible in this design?",
        "What's the narrative the user tells themselves about this kind of experience?",
        "I'm curious what would feel truly generous about this solution.",
        "How might we design this to work at the margins of human capability?",
        "What would surprise us most to discover about how users actually use this?",
        "How might we honor the diversity of needs across the people we're designing for?",
        "I wonder what would feel most human about this design.",
        "What's the thing that would make this feel genuinely different rather than just new?",
        "How might we approach this with a beginner's mind?",
        "I'm curious what a ten-year-old would think of this solution.",
        "What would it mean for this design to create conditions for flourishing?",
        "How might we think about the second-order effects of this design decision?",
        "What's the doubt you're holding about this direction that might be worth listening to?",
        "I wonder what would happen if we made the solution invisible.",
        "How might we design for the moments when things go wrong?",
        "What would it mean to design for the long arc of a user's relationship with this?",
        "I'm curious what this design would need to unlearn from existing solutions.",
        "How might we create a sense of possibility rather than certainty in this design?",
        "What's the most generous interpretation of the user's behavior we could take?",
        "I wonder what this problem would look like if we saw it as a gift.",
        "How might we make room for serendipity in the user's experience?",
        "What would feel like the most honest version of this solution?",
        "I'm curious what you're protecting by not exploring that idea further.",
        "How might we design this to strengthen the user's own agency?",
        "What's the thing about this problem that makes you most uncomfortable?",
        "I wonder what the most hopeful version of this outcome would look like.",
        "How might we design this to be worthy of the user's trust?",
        "What would it mean for this solution to create more questions than it answers?",
        "I'm curious what unmet need sits just beneath the surface of what we've been discussing.",
        "How might we bring more of the user's lived experience into this conversation?",
        "What's the aspect of this design that still has the most room to surprise us?",
    ]

    df = pd.DataFrame({
        "text": DIRECTIVE + OPEN_CO,
        "label": [1]*len(DIRECTIVE) + [0]*len(OPEN_CO)
    })
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    out_csv = os.path.join(BASE, "data", "processed", "directiveness_training_data.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Training data: {len(df)} examples ({df['label'].sum()} directive, {(df['label']==0).sum()} open)")

    # ─── Step 4B: Fine-tune DistilBERT ──────────────────────────────────────────
    from transformers import (DistilBertTokenizer, DistilBertForSequenceClassification,
                              Trainer, TrainingArguments)
    from torch.utils.data import Dataset

    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    class DirDataset(Dataset):
        def __init__(self, texts, labels):
            self.enc = tokenizer(list(texts), truncation=True, padding=True, max_length=128)
            self.labels = list(labels)
        def __len__(self): return len(self.labels)
        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

    train_ds = DirDataset(train_df["text"], train_df["label"])
    test_ds  = DirDataset(test_df["text"],  test_df["label"])

    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    model_dir = os.path.join(BASE, "models", "directiveness_classifier")
    args = TrainingArguments(
        output_dir=model_dir,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
        logging_dir=os.path.join(model_dir, "logs"),
        report_to="none",
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(-1)
        return {"accuracy": accuracy_score(labels, preds)}

    trainer = Trainer(model=model, args=args,
                      train_dataset=train_ds, eval_dataset=test_ds,
                      compute_metrics=compute_metrics)
    trainer.train()
    results = trainer.evaluate()
    acc = results["eval_accuracy"]
    print(f"\nDistilBERT Directiveness Classifier Test Accuracy: {acc:.3f}")

    final_dir = os.path.join(model_dir, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Model saved to {final_dir}")

    # Write accuracy to file so main.tex can pick it up
    with open(os.path.join(BASE, "data", "outputs", "directiveness_accuracy.txt"), "w") as f:
        f.write(f"{acc:.3f}")



if __name__ == "__main__":
    main()
