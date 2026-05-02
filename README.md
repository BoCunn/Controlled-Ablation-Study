# Ablation Study: TF-IDF vs Bag of Words

A beginner-friendly Python script that runs an ablation study on text classification using the 20 Newsgroups dataset. It isolates one variable — the feature extraction method — and measures how that single change affects model accuracy.

---

## Ablation Design

| Component        | Baseline       | Modified       |
|------------------|----------------|----------------|
| Features         | TF-IDF         | Bag of Words   |
| Classifier       | Multinomial NB | Multinomial NB |
| Categories       | 5 (same)       | 5 (same)       |
| Train/test split | 80/20 (same)   | 80/20 (same)   |

Only the feature extraction method changes. Everything else is held constant.

---

## Requirements

- Python 3.7+
- `scikit-learn`
- `numpy`
- `matplotlib`

Install dependencies with:

```bash
pip install scikit-learn numpy matplotlib
```

---

## Usage

```bash
python main.py
```

The dataset (~14 MB) is downloaded automatically on the first run and cached locally by sklearn.

**Estimated runtime:** 30–90 seconds on first run, 20–40 seconds after that.

---

## Output

- Console: per-run accuracies, a side-by-side comparison table, and a written conclusion
- File: `ablation_plot.png` — line plot comparing TF-IDF vs Bag of Words accuracy across all 5 runs