# Neural-HMM Language Identification

A **Deep Learning + Probabilistic Reasoning** project on **Language Identification** using  
- Character-level **Markov Chains**  
- **Multinomial Hidden Markov Models (HMMs)**  
- Custom **Neural-HMM (PyTorch)**

This project compares probabilistic and neural approaches for modeling character sequences across 14 languages.

---

## Project Structure

```
.
├── data/
│   └── clean/<lang>/{train,dev,test}.txt   # per-language splits
├── tokenizers/
│   └── vocab_char.txt                      # base vocabulary
├── src/
│   ├── data/
│   │   ├── datasets.py                     # dataset loaders
│   │   └── utils.py                        # vocab + language detection
│   ├── models/
│   │   ├── hmm_multinomial.py              # Multinomial HMM (hmmlearn)
│   │   └── neural_hmm.py                   # PyTorch Neural-HMM
│   └── scripts/
│       ├── evaluate_hmm.py                 # train/eval HMM
│       └── train_neural_hmm.py             # train/eval Neural-HMM
├── outputs/
│   ├── hmm/                                # HMM reports
│   ├── mc/                                 # Markov Chain reports
│   └── neural_hmm/                         # Neural-HMM runs
└── README.md
```

---

## Dataset

- **Languages (14):** Catalan (ca), Danish (da), German (de), English (en), Spanish (es), Finnish (fi), French (fr), Icelandic (is), Italian (it), Dutch (nl), Norwegian (nr), Portuguese (pt), Romanian (ro), Swedish (sv)
- Each language has `train/dev/test` splits in `data/clean/<lang>/`.

---

## Methods

### 1. Markov Chain (Baseline)
- First-order character-level Markov model
- Probability of a sentence computed as a product of conditional next-character probabilities
- Used as a strong baseline for language identification

---

### 2. Multinomial Hidden Markov Model (HMM)
- Standard HMM with multinomial emissions
- Parameters learned via EM (Baum-Welch, implemented with `hmmlearn`)
- Sentence likelihoods computed using the Forward algorithm
- Hidden states capture latent structure in character sequences

---

### 3. Neural-HMM (PyTorch)
- Hidden states with learnable transitions + start probabilities
- Emission distributions parameterized by an **MLP** (multi-layer perceptron) over:
  - Character embeddings
  - Local context window
- Training via negative log-likelihood using log-forward algorithm
- Regularization: gradient clipping, optional dropout
- Optimization: Adam (planned extensions: AdamW, learning rate schedules)

---

## Installation

```bash
git clone https://github.com/shruti-sivakumar/Neural-HMM-LangID.git
cd Neural-HMM-LangID
pip install -r requirements.txt
```

Requirements:
- Python 3.10+
- PyTorch
- numpy, tqdm
- hmmlearn

---

## Usage

### Markov Chain
```bash
python3 -m src.scripts.evaluate_mc
```

### Multinomial HMM
```bash
python3 -m src.scripts.evaluate_hmm --states 8
```

### Neural-HMM
```bash
python3 -m src.scripts.train_neural_hmm   --states 12 --epochs 20 --context 3   --batch_size 32 --lr 1e-3 --dropout 0.3
```

Reports (likelihoods, confusion matrices) are saved under `outputs/`.

---

## Research Plan

- Compare **probabilistic vs neural** sequence models for character-level language ID
- Ablation studies:
  - Vary hidden states (K)
  - Context window size
  - Dropout rates
  - Optimizers (Adam vs AdamW)
  - Sentence length buckets (short, medium, long)
- Error analysis using confusion matrices
- Evaluate robustness under noisy text

---

## License
MIT License