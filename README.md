# Autoregressive Word-Level Text Generation with SimpleRNN, Bidirectional RNN, LSTM and GRU

Foundations of Generative AI (ICT4403) – Assignment 2

## 1. Problem statement and objective
Build an autoregressive language model that predicts the next word given the previous 20 tokens, and use it to generate text one word at a time. Four recurrent architectures (SimpleRNN, Bidirectional RNN, LSTM, GRU) are trained under identical conditions and compared on quantitative metrics (loss, perplexity, accuracy, training time) and on qualitative properties of generated text (coherence, diversity, contextual consistency, repetition). Text is generated from the same seeds at three sampling temperatures (0.5, 1.0, 1.5).

## 2. Dataset
**Tiny Shakespeare** (Andrej Karpathy, *char-rnn*), about 1.1 MB of plays by William Shakespeare.
Source: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

| Property | Value |
|---|---|
| Characters | 1,115,394 |
| Lines | 40,000 |
| Tokens (words + punctuation + newline) | 290,145 (12,380 unique) |
| Vocabulary used | 3,000 most frequent tokens (94.3 % token coverage; rest mapped to `<unk>`) |
| Content | Speaker labels followed by dialogue in early-modern English verse |

Pre-processing: lower-casing; regex tokenisation into words, punctuation (`. , ! ? ; :`) and a `<nl>` line-break token; frequency-capped vocabulary with `<pad>` (0) and `<unk>` (1); chronological 90/5/5 train/validation/test split made **before** windowing so overlapping windows cannot leak across splits. Input–target pairs: a window of 20 tokens is the input, the next token is the target (stride 3), giving 87,037 training, 4,829 validation and 4,830 test pairs.
Dataset plots: `results/dataset_characteristics.png`.

## 3. Models
All share `Embedding(V,100,mask_zero) -> recurrent layer (128 units) -> Dropout(0.3) -> Dense(V, softmax)`, Adam (lr 1e-3, clipnorm 1), batch 256, max 10 epochs, early stopping on validation loss (patience 2, best weights restored), seed 42.

* **SimpleRNN**: plain tanh recurrence.
* **Bidirectional RNN**: `Bidirectional(SimpleRNN)`; reads the known 20-token context forwards and backwards (no future leakage).
* **LSTM**: gated cell with a separate cell state.
* **GRU**: lighter gated unit with update/reset gates.

## 4. How to run
```bash
git clone <this-repo-url>
cd <repo-folder>
pip install -r requirements.txt
jupyter notebook FGAI_Assignment2_TextGeneration.ipynb   # Run All
```
The notebook downloads the dataset automatically. The full run took roughly 10 to 15 minutes on a single CPU core. Outputs go to `results/`.

## 5. Results (held-out test set)
| Model | Params | Epochs run | Train time (s) | Test loss | **Test perplexity** | Top-1 acc (%) | Top-5 acc (%) |
|---|---|---|---|---|---|---|---|
| SimpleRNN | 716,312 | 10 | 102.8 | 4.546 | 94.2 | 28.43 | 44.74 |
| BiRNN | 1,129,624 | 6 (early stop) | 125.0 | 4.616 | 101.0 | 27.25 | 43.54 |
| LSTM | 804,248 | 10 (cap reached) | 225.5 | 4.638 | 103.3 | 26.67 | 44.37 |
| **GRU** | 775,320 | 10 | 227.0 | **4.493** | **89.4** | **28.86** | **45.57** |

Plots: `results/training_curves.png`, `results/metric_comparison.png`. Raw numbers: `results/quantitative_metrics.csv`, `results/training_history.json`.

## 6. Model comparison and evaluation
* **GRU** has the lowest test loss and perplexity and the highest top-1 and top-5 accuracy; its validation loss was the lowest of the four from epoch 5 onward (best 4.218 at epoch 9).
* **SimpleRNN** is close behind (perplexity 94.2) and the cheapest: about half the training time of the gated models in this implementation.
* **BiRNN** has the most parameters, over-fitted soonest (best epoch 4, stopped at 6) and did not beat the unidirectional SimpleRNN. For next-word prediction only the *past* context matters, so reading the window backwards adds capacity but little useful signal.
* **LSTM** ranks last on perplexity, **but this is partly a training-budget effect**: its validation loss was still falling at epoch 10 (4.333, then 4.328) when the 10-epoch cap stopped it. Given more epochs it would likely improve; the other three had already plateaued or begun to overfit.
* All perplexities are high (about 90 to 100 against a 3,000-word vocabulary) because the corpus is small (about 200k words); no model learns real long-range semantics.

Generation statistics (3 seeds x 6 samples x 50 words per cell; `results/generation_statistics.csv`). *Bigram validity* = share of generated bigrams that occur in the training text (proxy for local fluency / contextual consistency):

| Model | T | Distinct-2 | Trigram repeat | Bigram validity |
|---|---|---|---|---|
| SimpleRNN | 0.5 / 1.0 / 1.5 | 0.868 / 0.967 / 0.994 | 0.039 / 0.005 / 0.000 | 0.963 / 0.710 / 0.438 |
| BiRNN | 0.5 / 1.0 / 1.5 | 0.871 / 0.974 / 0.997 | 0.030 / 0.000 / 0.000 | 0.923 / 0.673 / 0.350 |
| LSTM | 0.5 / 1.0 / 1.5 | 0.855 / 0.954 / 0.990 | 0.049 / 0.005 / 0.000 | 0.948 / 0.732 / 0.439 |
| GRU | 0.5 / 1.0 / 1.5 | 0.873 / 0.958 / 0.989 | 0.038 / 0.002 / 0.000 | 0.958 / 0.730 / 0.405 |

All generated samples are in `results/generated_samples.md`.

## 7. Sampling-temperature analysis
Seeds: three Shakespeare-style prompts (a *First Citizen* line, a *Romeo* line, a court line). Every model sees the same seeds and the same random generator seed.

* **T = 0.5 (conservative).** Output is locally fluent (92 to 96 % of bigrams seen in training) and keeps the play format (speaker name, colon, line break), but is repetitive and generic: stock phrases such as "i have not", "my lord", "king richard iii", and loops like "sir, sir". Trigram repetition is highest here (3 to 5 %), diversity lowest (Distinct-2 about 0.86 to 0.87).
* **T = 1.0 (model distribution).** Best balance. Diversity is high (Distinct-2 about 0.95 to 0.97), repetition almost disappears (below 0.5 %), and phrases are still mostly plausible (67 to 73 % valid bigrams), though sentences drift and grammar slips ("i should would not bent").
* **T = 1.5 (flat).** Near-zero repetition and maximal diversity (Distinct-2 about 0.99), but coherence collapses: only 35 to 44 % valid bigrams and output reads as a word salad of rare vocabulary.

So temperature trades coherence against diversity monotonically: raising T increases diversity and removes repetition but destroys local consistency. The useful range for this dataset is around T = 0.5 to 1.0 (an intermediate value such as 0.7 would be the natural next thing to test; it was not run here). Comparing models, BiRNN has the lowest bigram validity at every temperature, while GRU, SimpleRNN and LSTM are similar at T = 1.0 (LSTM marginally highest at 0.732 vs GRU 0.730).

## 8. Conclusion
**GRU** is the best model for this dataset: lowest perplexity (89.4), highest top-1 (28.86 %) and top-5 (45.57 %) accuracy, and generation quality on par with the best at T = 1.0 and strong at T = 0.5. SimpleRNN is the best cost-efficiency choice. Bidirectional processing gives no benefit for strictly left-to-right generation, and the LSTM was under-trained under the shared 10-epoch budget.

**Limitations.** Each model was trained once (single seed), and the gaps between models (perplexity 89 to 103) are modest, so rankings could change with other seeds. Coherence and contextual consistency are assessed with simple proxies plus reading the samples; none of the models produces semantically coherent passages. Training times are wall-clock on one CPU core and include the gated models' higher per-step cost.

## 9. Repository structure
```
FGAI_Assignment2_TextGeneration.ipynb   main notebook (executed, with outputs)
requirements.txt
results/
  quantitative_metrics.csv, generation_statistics.csv, overall_ranking.csv
  training_history.json, generated_samples.md
  dataset_characteristics.png, training_curves.png, metric_comparison.png, temperature_analysis.png
  *.weights.h5                          trained weights for the four models
```
