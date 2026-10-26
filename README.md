# Building a Spam Filter with Multinomial Naive Bayes

Classifying 5,572 SMS messages (13.41% spam) with multinomial naive Bayes **implemented from scratch**. No scikit-learn anywhere, including the train/test split, the evaluation metrics and the ROC AUC, all of which are hand-written and sanity-checked against known cases.

Dataquest guided project, Data Scientist in Python path.

## Results

| Metric | Value |
|---|---|
| Accuracy | **0.9865** |
| Precision (spam) | 0.9718 |
| Recall (spam) | 0.9262 |
| F1 (spam) | 0.9485 |
| ROC AUC | 0.9793 |
| Majority-class baseline | 0.8659 |

Confusion matrix on 1,114 held-out messages: 961 TN, **4 FP**, 11 FN, 138 TP. Four legitimate messages out of 965 misfiled, a 0.41% rate. One of them was *"Unlimited texts. Limited minutes."* at P(spam) = 0.6252, which is hard to blame the model for.

## Three findings worth more than the accuracy

### 1. The probabilities are worthless and the classifier works anyway

Naive Bayes treats each word as independent evidence, which is false in language: "free" and "prize" co-occur far more than independence predicts, and each correlated pair is counted as fresh information. Multiplying redundant evidence drives the posterior exponentially toward 0 or 1.

| P(spam) range | Messages | Share |
|---|---|---|
| below 1e-10 | 238 | 21.4% |
| 1e-10 to 0.01 | 671 | 60.2% |
| 0.01 to 0.1 | 52 | 4.7% |
| **0.1 to 0.9** | **17** | **1.5%** |
| 0.9 to 1.0 | 136 | 12.2% |

Only **6.55%** of messages land in the middle four confidence bands, and 21.4% get a spam probability below one in ten billion. A classifier wrong 1.35% of the time should be uncertain far more often than that. In the lowest band it predicts a mean P(spam) of 0.0003 against an actual rate of 0.0066, understating risk by about 20x.

Yet ROC AUC is **0.9793**, because AUC depends only on the ordering of scores, not their values. **High accuracy, excellent ranking, and worthless probabilities are three separate facts.** Treating a naive Bayes score as a confidence level is a mistake this notebook demonstrates rather than asserts.

### 2. Log-space arithmetic was not actually necessary here

The standard justification is that multiplying many small probabilities underflows float64 to zero. I measured it instead of repeating it:

| | Result |
|---|---|
| Test messages where direct multiplication underflowed | **0 of 1,114** |
| Smallest non-underflowed posterior observed | 1.242e-259 |
| Smallest positive float64 | 4.941e-324 |
| Longest test message | 95 tokens |

Growing a real message by repetition locates the boundary between **56 and 112 tokens**. SMS tops out at 95, so this dataset sits just inside the safe range. An email or document classifier would cross that line routinely. Log space costs nothing and removes the failure mode, so it stays the right default; it simply was not what made this work, and saying otherwise would be received wisdom dressed up as a finding.

### 3. Smoothing is where the asymmetric costs get decided

| alpha | Accuracy | Precision | Recall | False positives | False negatives |
|---|---|---|---|---|---|
| 0.01 | 0.9874 | 0.9655 | 0.9396 | 5 | **9** |
| **0.20** | **0.9883** | 0.9722 | 0.9396 | 4 | 9 |
| 1.00 | 0.9865 | 0.9718 | 0.9262 | 4 | 11 |
| 2.00 | 0.9856 | 0.9784 | 0.9128 | 3 | 13 |
| **5.00** | 0.9776 | **1.0000** | 0.8322 | **0** | 25 |

Best accuracy at alpha=0.2, fewest missed spam at alpha=0.01, **zero false positives at alpha=5.0**. Three different settings.

A false positive is a real message silently diverted, possibly a job offer. A false negative is an annoying text in the inbox. If the requirement is "never misfile a legitimate message", alpha=5 delivers that on this test set at the price of letting 25 of 149 spam through. That is a product decision the table informs, not a modelling decision the largest number settles.

## Verification, and its limit

Rebuilding the posterior by a separate code path (plain multiplication of raw counts rather than sums of logarithms) agreed to a maximum log difference of **5.68e-14**, which is float64 rounding.

That checks arithmetic and bookkeeping. It does **not** check whether my understanding of the algorithm is correct, since both paths share it. A cross-check against `MultinomialNB` would, and it is the first thing to add in an environment that has scikit-learn. Calling the implementation "verified" without that caveat would overstate the test.

The hand-written AUC was also sanity-checked, and the check earned its place: an early draft asserted the wrong expected value for one case, and the assertion caught it.

## The vocabulary leak barely mattered, which is not the lesson

| Vocabulary | Accuracy | Precision | Recall | FP | FN |
|---|---|---|---|---|---|
| Honest (training only, 7,854 words) | 0.9865 | 0.9718 | 0.9262 | 4 | 11 |
| Leaky (train + test, 8,753 words) | 0.9874 | 0.9530 | 0.9530 | 7 | 7 |

The leak changed 7 of 1,114 predictions, moved accuracy by +0.0009, and made precision worse. A shared vocabulary leaks no *label* information: only the smoothing denominator changes, spreading the same alpha mass over 899 more words.

That is not evidence the practice is safe. The identical "it is only preprocessing" reasoning applied to TF-IDF document frequencies, target encoding or feature scaling leaks genuinely damaging amounts while looking the same from outside. Getting it right when it does not matter is how the habit exists when it does.

## Comparison against three alternatives, all from scratch

| Approach | Accuracy | Precision | Recall | FP | FN |
|---|---|---|---|---|---|
| **Multinomial NB (alpha=1)** | **0.9865** | 0.9718 | 0.9262 | 4 | 11 |
| Bernoulli NB (alpha=1) | 0.9811 | **0.9923** | 0.8658 | **1** | 20 |
| Keyword rule, 1+ spam word | 0.9524 | 0.9528 | 0.6779 | 5 | 48 |
| Keyword rule, 2+ spam words | 0.9318 | 1.0000 | 0.4899 | 0 | 76 |
| Length-only (> 140 chars) | 0.8609 | 0.4854 | 0.6711 | 106 | 49 |

Bernoulli produces **one** false positive against four, because scoring the absence of every vocabulary word makes it more conservative. On asymmetric costs it may be the better filter despite the worse headline.

The keyword rule is the reality check: 25 hand-picked words reach 95.2% accuracy, 3.4 points below the full model, which earns its keep on recall (92.6% against 67.8%).

**The length-only rule is the most instructive failure.** Spam averages 138.67 characters against 71.48, and length alone scores an **AUC of 0.8807**, so it ranks genuinely well. Its accuracy of 0.8609 is nonetheless *below* the 0.8659 baseline: at a 13.4% base rate no single threshold converts that ranking into useful classification. Good signal and no usable decision rule are compatible, the same lesson as the calibration result arriving from the opposite direction.

## Method

1. Metrics first: confusion matrix, accuracy, precision, recall, F1 and rank-based AUC (Mann-Whitney U, ties at half credit), each sanity-checked.
2. Profile the data, including 403 duplicate rows and a check for contradictory labels (none).
3. Stratified split by hand, before any text processing.
4. Tokenise and build the vocabulary from **training data only**.
5. Implement multinomial naive Bayes with Laplace smoothing, in log space.
6. Verify against an independent recomputation and test the underflow claim.
7. Inspect the learned log-likelihood ratios with a frequency floor, so the table shows evidence rather than smoothing artefacts.
8. Measure the vocabulary leak.
9. Examine calibration and the independence assumption.
10. Sweep alpha and read it as a cost trade-off.
11. Read the actual misclassified messages.
12. Compare against Bernoulli NB, keyword rules and a length threshold.

## Limitations

- 403 exact duplicates were kept, and 11.0% of test messages appear verbatim in training, so the test figures are inflated and true novel-message accuracy is below 0.9865. The choice was deliberate (the prior legitimately uses frequency information) and the cost is reported rather than hidden.
- A single 80/20 split with only 149 test spam: one reclassified message moves recall by 0.0067. Cross-validation was not run.
- The alpha sweep was evaluated on the test set, so it demonstrates the trade-off rather than selecting a value legitimately.
- The verification cannot catch conceptual errors, as stated above.
- A 2011 SMS corpus. Shortened URLs, unicode homoglyphs and image payloads are absent, so this would transfer poorly to current traffic.
- The tokeniser discards all non-word characters, throwing away currency symbols, exclamation marks, capitalisation and phone numbers. Keeping them would likely help and was not tested.

## Stack

Python, pandas, NumPy, matplotlib, and the standard library (`re`, `math`, `collections.Counter`). No scikit-learn.

## Files

- `notebook.ipynb`: the full analysis with executed outputs and figures.