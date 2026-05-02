# main.py
# Ablation Study: Bag of Words vs TF-IDF for Text Classification
# 20 Newsgroups Dataset | Multinomial Naive Bayes Classifier
# ---------------------------------------------------------------
# ABLATION CHOICE: Option A (Feature Set)
#   - Baseline : TF-IDF (TfidfVectorizer)
#   - Modified : Bag of Words (CountVectorizer)
# Everything else (classifier, categories, splits) stays the same.
# ---------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------

# Subset of categories — kept small so the script runs quickly
CATEGORIES = [
    'rec.sport.hockey',
    'sci.space',
    'talk.politics.guns',
    'comp.graphics',
    'rec.autos'
]

NUM_RUNS  = 5     # number of independent train/test split experiments
TEST_SIZE = 0.2   # 20% of data used for testing each run

# ---------------------------------------------------------------
# 2. DATA LOADING
# ---------------------------------------------------------------

print("Loading 20 Newsgroups dataset...")
print(f"Categories: {CATEGORIES}\n")

# Load the full dataset (train + test combined) so we control splits ourselves
newsgroups = fetch_20newsgroups(
    subset='all',
    categories=CATEGORIES,
    shuffle=True,
    random_state=42,                        # reproducible global shuffle
    remove=('headers', 'footers', 'quotes') # strip metadata to avoid leakage
)

texts  = newsgroups.data    # raw text strings
labels = newsgroups.target  # integer class labels

print(f"Total documents: {len(texts)}")
print(f"Class counts: {dict(zip(*np.unique(labels, return_counts=True)))}\n")

# ---------------------------------------------------------------
# 3. HELPER FUNCTION — run one experiment
# ---------------------------------------------------------------

def run_experiment(X, y, num_runs, test_size):
    """
    Given a pre-vectorized feature matrix X and labels y,
    run `num_runs` independent train/test splits and return
    a list of accuracy scores.
    """
    accuracies = []
    for run in range(num_runs):
        seed = run * 10   # seeds: 0, 10, 20, 30, 40 — same for both setups

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed
        )

        # Train Multinomial Naive Bayes (same classifier for both setups)
        model = MultinomialNB()
        model.fit(X_train, y_train)

        # Evaluate on the held-out test set
        y_pred = model.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        accuracies.append(acc)

    return accuracies

# ---------------------------------------------------------------
# 4. BASELINE — TF-IDF features
# ---------------------------------------------------------------

print("=" * 55)
print("BASELINE: TF-IDF (TfidfVectorizer)")
print("=" * 55)

# TF-IDF down-weights words that appear in many documents,
# which helps the model focus on distinctive terms.
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
X_tfidf = tfidf_vectorizer.fit_transform(texts)

tfidf_accuracies = run_experiment(X_tfidf, labels, NUM_RUNS, TEST_SIZE)

for i, acc in enumerate(tfidf_accuracies, start=1):
    print(f"  Run {i} (seed={(i-1)*10:>2}): {acc:.4f}")

tfidf_mean = np.mean(tfidf_accuracies)
tfidf_std  = np.std(tfidf_accuracies)
print(f"  Mean: {tfidf_mean:.4f}  |  Std: {tfidf_std:.4f}\n")

# ---------------------------------------------------------------
# 5. MODIFIED — Bag of Words features
# ---------------------------------------------------------------

print("=" * 55)
print("MODIFIED: Bag of Words (CountVectorizer)")
print("=" * 55)

# Bag of Words just counts how many times each word appears.
# It does NOT account for how common a word is across all documents.
bow_vectorizer = CountVectorizer(stop_words='english', max_features=10000)
X_bow = bow_vectorizer.fit_transform(texts)

bow_accuracies = run_experiment(X_bow, labels, NUM_RUNS, TEST_SIZE)

for i, acc in enumerate(bow_accuracies, start=1):
    print(f"  Run {i} (seed={(i-1)*10:>2}): {acc:.4f}")

bow_mean = np.mean(bow_accuracies)
bow_std  = np.std(bow_accuracies)
print(f"  Mean: {bow_mean:.4f}  |  Std: {bow_std:.4f}\n")

# ---------------------------------------------------------------
# 6. RESULTS TABLE
# ---------------------------------------------------------------

print("=" * 55)
print("ABLATION RESULTS TABLE")
print("=" * 55)
print(f"{'Run':<6} {'TF-IDF (Baseline)':>18} {'Bag of Words':>14}")
print("-" * 42)
for i in range(NUM_RUNS):
    print(f"{i+1:<6} {tfidf_accuracies[i]:>18.4f} {bow_accuracies[i]:>14.4f}")
print("-" * 42)
print(f"{'Mean':<6} {tfidf_mean:>18.4f} {bow_mean:>14.4f}")
print(f"{'Std':<6} {tfidf_std:>18.4f} {bow_std:>14.4f}")
print("=" * 55)

# ---------------------------------------------------------------
# 7. VISUALIZATION
# ---------------------------------------------------------------

run_numbers = list(range(1, NUM_RUNS + 1))

plt.figure(figsize=(9, 5))

# Plot both setups on the same axes for easy side-by-side comparison
plt.plot(run_numbers, tfidf_accuracies, marker='o', linewidth=2,
         color='steelblue', label=f'TF-IDF (mean={tfidf_mean:.4f})')
plt.plot(run_numbers, bow_accuracies,   marker='s', linewidth=2,
         color='tomato',    label=f'Bag of Words (mean={bow_mean:.4f})')

# Dashed horizontal lines to show each mean clearly
plt.axhline(y=tfidf_mean, color='steelblue', linestyle='--', linewidth=1, alpha=0.6)
plt.axhline(y=bow_mean,   color='tomato',    linestyle='--', linewidth=1, alpha=0.6)

plt.title('Ablation Study: TF-IDF vs Bag of Words\n(Multinomial Naive Bayes, 5 Categories)')
plt.xlabel('Run Number')
plt.ylabel('Accuracy')
plt.xticks(run_numbers)
plt.ylim(0, 1.05)
plt.legend()
plt.tight_layout()
plt.savefig('ablation_plot.png', dpi=150)
plt.show()

print("\n[Plot saved as ablation_plot.png]")

# ---------------------------------------------------------------
# 8. CONCLUSION
# ---------------------------------------------------------------

# Determine which setup performed better based on mean accuracy
if tfidf_mean > bow_mean:
    winner = "TF-IDF"
    loser  = "Bag of Words"
    diff   = tfidf_mean - bow_mean
else:
    winner = "Bag of Words"
    loser  = "TF-IDF"
    diff   = bow_mean - tfidf_mean

conclusion = f"""
CONCLUSION
----------
In this ablation study, {winner} outperformed {loser} by an average of
{diff:.4f} accuracy points across {NUM_RUNS} runs with different random splits.

TF-IDF generally works better for text classification because it penalizes
words that appear frequently across many documents (like "said" or "people"),
reducing their influence and letting the model focus on more distinctive terms.
Bag of Words treats all word counts equally, so common but uninformative words
can still dominate the feature space even after stopword removal.

Some variability was observed across runs (TF-IDF std={tfidf_std:.4f},
BoW std={bow_std:.4f}), which is expected since each run uses a different
random train/test split. This confirms that reporting mean ± std across
multiple runs gives a more honest picture of performance than a single result.
"""

print(conclusion)