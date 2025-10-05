import random
import unicodedata
import re
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
RSEED = 1337
random.seed(RSEED)
P_DROP = 0.10     # probability to drop a character
P_SUB = 0.05      # probability to replace a character randomly
MIN_LEN = 5       # minimum sentence length

# -----------------------------
# HELPERS
# -----------------------------
def strip_accents(text):
    """Remove accents/diacritics from characters."""
    return ''.join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn')

def make_noisy(s):
    """Apply lowercasing, accent removal, and character-level noise."""
    s = strip_accents(s.lower())
    s = re.sub(r'[^a-z\s]', ' ', s)   # keep only a-z and spaces
    s = re.sub(r'\s+', ' ', s).strip()
    out = []
    for ch in s:
        if ch == ' ':
            out.append(ch)
            continue
        if random.random() < P_DROP:
            continue
        if random.random() < P_SUB:
            out.append(chr(ord('a') + random.randint(0, 25)))
        else:
            out.append(ch)
    s2 = ''.join(out).strip()
    if len(s2) > 7:
        s2 = s2[:random.randint(MIN_LEN, len(s2))]
    if len(s2) < MIN_LEN:
        s2 = (s2 + ' ' + s)[:MIN_LEN]
    return re.sub(r'\s+', ' ', s2).strip()

def build_noisy_eval(clean_root, noisy_root):
    clean_root, noisy_root = Path(clean_root), Path(noisy_root)
    for lang_dir in clean_root.iterdir():
        if not lang_dir.is_dir():
            continue
        src = lang_dir / "eval_sentences.txt"
        if not src.exists():
            continue
        lines = src.read_text(encoding="utf-8").splitlines()
        noisy_lines = [make_noisy(l) for l in lines]
        out_dir = noisy_root / lang_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "eval_sentences.txt"
        out_file.write_text("\n".join(noisy_lines), encoding="utf-8")
        print(f"[Noisy saved] {out_file}")

# -----------------------------
# RUN (example)
# -----------------------------
if __name__ == "__main__":
    build_noisy_eval("data/clean", Path("data/noisy_eval"))