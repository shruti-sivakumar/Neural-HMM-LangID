#!/usr/bin/env python3
"""
Final evaluation across all three models on the two *eval* sets:
  - Clean eval:        data/clean/**/eval_sentences.txt
  - Noisy eval:        data/noisy_eval/**/eval_sentences.txt

Rules:
  • MC & HMM are TRAINED on data/clean/train (once), then EVALUATED on the chosen eval_dir.
  • HMM uses K=12 only (per user request).
  • Neural-HMM loads saved checkpoints from outputs/neural_hmm/90.1 and reuses the training-time scorer.
Outputs:
  • Per-model eval reports saved under outputs/{mc|hmm|neural_hmm/...}/eval_report_*.txt
  • Final summary at outputs/final_eval_summary.txt
"""

from pathlib import Path
import sys
import math
import torch

# --- Project paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))  # ensure 'src' is importable as a package

# --- Imports from your repo ---
from src.data.utils import detect_languages, build_char_vocab_from_file
from src.data.datasets import load_sentences, load_eval_sentences

from src.models.markov_chain import MarkovClassifier
from src.models.hmm_multinomial import train_lang_hmm, score_sentence as score_sentence_hmm

from src.models.neural_hmm import NeuralHMM
from src.scripts.train_neural_hmm import score_sentence as score_sentence_neural


# ---------------------------
# Small helpers
# ---------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def make_conf_matrix(langs):
    return {t: {p: 0 for p in langs} for t in langs}

def report_lines_from_conf(title, langs, acc, conf):
    header = "true\\pred".ljust(8) + "".join(f"{lg:>6}" for lg in langs)
    lines = [f"{title}: {acc:.3f}", "", header]
    for t in langs:
        lines.append(t.ljust(8) + "".join(f"{conf[t][p]:>6}" for p in langs))
    return lines


# ---------------------------
# MC: train-on-clean, eval-on(eval_dir)
# ---------------------------
def evaluate_mc(clean_train_dir: Path, eval_dir: Path):
    outdir = ensure_dir(PROJECT_ROOT / "outputs" / "mc")

    # Train on clean/train
    train_langs = detect_languages(clean_train_dir)
    data_by_lang = {lg: load_sentences(lg, "train", base=str(clean_train_dir)) for lg in train_langs}
    clf = MarkovClassifier(alpha=0.5)
    clf.fit_per_language(data_by_lang)

    # Eval on eval_dir eval_sentences
    eval_langs = detect_languages(eval_dir)
    conf = make_conf_matrix(eval_langs)
    correct = total = 0
    rows = []

    for t in eval_langs:
        try:
            eval_sents = load_eval_sentences(t, base=str(eval_dir))
        except Exception:
            eval_sents = []
        for s in eval_sents:
            pred, scores = clf.predict(s)
            conf[t][pred] += 1
            correct += int(pred == t)
            total += 1
            rows.append((t, s, pred, scores))

    acc = correct / max(1, total)

    # Save report
    title = f"Accuracy (MC) on {eval_dir.name}"
    lines = report_lines_from_conf(title, eval_langs, acc, conf)
    lines.append("")
    lines.append("True  Pred  " + " ".join(f"{lg:>8}" for lg in eval_langs) + "  Sentence")
    lines.append("-" * 120)
    for lang, sent, pred, scores in rows:
        score_str = " ".join(f"{scores[lg]:8.1f}" for lg in eval_langs)
        lines.append(f"{lang:<4} {pred:<4} {score_str}  {sent}")

    out_file = outdir / f"eval_report_mc_{eval_dir.name}.txt"
    out_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"[RESULT] MC accuracy on {eval_dir.name}: {acc:.3f}")
    print(f"[SAVED] {out_file}")
    return acc


# ---------------------------
# HMM (Multinomial): K=12
# Train on clean/train, eval on eval_dir eval_sentences
# ---------------------------
def evaluate_hmm(clean_train_dir: Path, eval_dir: Path):
    outdir = ensure_dir(PROJECT_ROOT / "outputs" / "hmm")
    eval_langs = detect_languages(eval_dir)
    train_langs = detect_languages(clean_train_dir)

    # Vocab (match your HMM expectations: just ensure [UNK] exists)
    vocab_path = PROJECT_ROOT / "tokenizers" / "vocab_char.txt"
    stoi, itos = build_char_vocab_from_file(vocab_path)
    if "[UNK]" not in stoi:
        itos = list(itos) + ["[UNK]"]
        stoi = {c: i for i, c in enumerate(itos)}

    # Train one HMM per train language with K=12
    K = 12
    models = {}
    for lg in train_langs:
        train = load_sentences(lg, "train", base=str(clean_train_dir))
        models[lg] = train_lang_hmm(
            train, stoi, n_states=K, n_iter=50, sticky_diag=0.85, seed=0
        )

    # Eval on eval_dir
    conf = make_conf_matrix(eval_langs)
    rows = []
    correct = total = 0

    for t in eval_langs:
        try:
            eval_sents = load_eval_sentences(t, base=str(eval_dir))
        except Exception:
            eval_sents = []
        for s in eval_sents:
            scores = {lg: score_sentence_hmm(models[lg], s, stoi) for lg in models}
            pred = max(scores, key=scores.get)
            conf[t][pred] += 1
            correct += int(pred == t)
            total += 1

            rows.append((t, s, pred, scores))

    acc = correct / max(1, total)

    # Save report
    title = f"Accuracy (HMM, K={K}) on {eval_dir.name}"
    lines = report_lines_from_conf(title, eval_langs, acc, conf)
    lines.append("")
    lines.append("True  Pred  " + " ".join(f"{lg:>8}" for lg in eval_langs) + "  Sentence")
    lines.append("-" * 120)
    for lang, sent, pred, scores in rows:
        score_str = " ".join(f"{scores.get(lg, float('-inf')):8.1f}" for lg in eval_langs)
        lines.append(f"{lang:<4} {pred:<4} {score_str}  {sent}")

    out_file = outdir / f"eval_report_hmm_{eval_dir.name}.txt"
    out_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"[RESULT] HMM accuracy on {eval_dir.name}: {acc:.3f}")
    print(f"[SAVED] {out_file}")
    return acc


# ---------------------------
# Neural-HMM: load checkpoints, eval on eval_dir
# ---------------------------
def evaluate_neural(eval_dir: Path, model_dir: Path):
    outdir = ensure_dir(model_dir)

    langs = detect_languages(eval_dir)
    vocab_path = PROJECT_ROOT / "tokenizers" / "vocab_char.txt"
    char2id, _ = build_char_vocab_from_file(vocab_path)
    if "[PAD]" not in char2id:
        char2id["[PAD]"] = len(char2id)
    if "[UNK]" not in char2id:
        char2id["[UNK]"] = len(char2id)

    V = len(char2id)
    pad_id = char2id["[PAD]"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load per-lang checkpoints
    models = {}
    for lg in langs:
        ckpt = model_dir / f"{lg}_k12.pt"
        if not ckpt.exists():
            print(f"[WARN] Missing Neural-HMM checkpoint for {lg}: {ckpt.name}")
            continue
        model = NeuralHMM(
            vocab_size=V, n_states=12,
            emb_dim=128, hidden=256, context=3,
            pad_idx=pad_id, dropout=0.3
        ).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        models[lg] = model
        print(f"[OK] Loaded {lg}")

    # Eval on eval_dir
    conf = make_conf_matrix(langs)
    correct = total = 0

    for t in langs:
        try:
            eval_sents = load_eval_sentences(t, base=str(eval_dir))
        except Exception:
            eval_sents = []
        for s in eval_sents:
            scores = {lg: score_sentence_neural(models[lg], s, char2id, device)
                      for lg in models}
            if not scores:
                continue
            pred = max(scores, key=scores.get)
            conf[t][pred] += 1
            correct += int(pred == t)
            total += 1

    acc = correct / max(1, total)

    # Save report
    title = f"Accuracy (Neural-HMM) on {eval_dir.name}"
    lines = report_lines_from_conf(title, langs, acc, conf)
    out_file = outdir / f"eval_report_neural_{eval_dir.name}.txt"
    out_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"[RESULT] Accuracy (Neural-HMM) on {eval_dir.name}: {acc:.3f}")
    print(f"[SAVED] {out_file}")
    return acc


# ---------------------------
# Main
# ---------------------------
def main():
    clean_eval_dir = PROJECT_ROOT / "data" / "clean"
    noisy_eval_dir = PROJECT_ROOT / "data" / "noisy_eval"
    neural_dir = PROJECT_ROOT / "outputs" / "neural_hmm" / "90.1"
    summary_path = PROJECT_ROOT / "outputs" / "final_eval_summary.txt"

    print("\n========== FINAL EVALUATION (EVAL SETS) ==========")

    # Train-on-clean, Eval on Clean Eval
    print("\n--- MC (train=clean/train, eval=clean/eval_sentences) ---")
    mc_clean = evaluate_mc(clean_train_dir=clean_eval_dir, eval_dir=clean_eval_dir)

    print("\n--- HMM (K=12) (train=clean/train, eval=clean/eval_sentences) ---")
    hmm_clean = evaluate_hmm(clean_train_dir=clean_eval_dir, eval_dir=clean_eval_dir)

    print("\n--- Neural-HMM (eval=clean/eval_sentences) ---")
    neu_clean = evaluate_neural(eval_dir=clean_eval_dir, model_dir=neural_dir)

    # Train-on-clean, Eval on Noisy Eval
    print("\n--- MC (train=clean/train, eval=noisy_eval/eval_sentences) ---")
    mc_noisy = evaluate_mc(clean_train_dir=clean_eval_dir, eval_dir=noisy_eval_dir)

    print("\n--- HMM (K=12) (train=clean/train, eval=noisy_eval/eval_sentences) ---")
    hmm_noisy = evaluate_hmm(clean_train_dir=clean_eval_dir, eval_dir=noisy_eval_dir)

    print("\n--- Neural-HMM (eval=noisy_eval/eval_sentences) ---")
    neu_noisy = evaluate_neural(eval_dir=noisy_eval_dir, model_dir=neural_dir)

    # Summary
    def fmt(x): return f"{x:.3f}" if isinstance(x, (float, int)) else "N/A"
    summary = [
        "Model\tCleanEval\tNoisyEval",
        f"MC\t{fmt(mc_clean)}\t{fmt(mc_noisy)}",
        f"HMM(K=12)\t{fmt(hmm_clean)}\t{fmt(hmm_noisy)}",
        f"Neural-HMM\t{fmt(neu_clean)}\t{fmt(neu_noisy)}",
    ]
    ensure_dir(summary_path.parent)
    summary_path.write_text("\n".join(summary), encoding="utf-8")

    print("\n========== SUMMARY (EVAL SETS) ==========")
    print(f"{'Model':<14}{'CleanEval':>12}{'NoisyEval':>12}")
    print("-" * 38)
    print(f"{'MC':<14}{fmt(mc_clean):>12}{fmt(mc_noisy):>12}")
    print(f"{'HMM(K=12)':<14}{fmt(hmm_clean):>12}{fmt(hmm_noisy):>12}")
    print(f"{'Neural-HMM':<14}{fmt(neu_clean):>12}{fmt(neu_noisy):>12}")
    print(f"\n[SAVED] {summary_path}")


if __name__ == "__main__":
    main()
