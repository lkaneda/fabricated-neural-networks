# Fabricated Neural Networks

> Can an LLM fabricate the weights and biases of a fully-trained neural network — without any training process?

This repository contains the code, conversations, and notes from an experiment by **Leila Kaneda** (May 2026) testing whether Claude (Sonnet 4.6) could produce deployment-ready PyTorch classifiers from nothing but a prompt and one reference image per class.

**TLDR: Yes. It works.**

---

## The Experiment

Four models were built across three classification domains. Each started with a natural language prompt and ended with a `.bin` PyTorch model file — no gradient descent, no labeled training set, no fine-tuning loop.

| Model | Classes | Backbone | Final Accuracy | Cost |
|---|---|---|---|---|
| Coffee (Full) | 15 coffee drinks | MobileNetV2 | 36% | $1.56 |
| Bikes | Unicycle / Bicycle / Tricycle | MobileNetV2 | 90% | $0.81 |
| Rainbows | No Rainbow / Rainbow / Double Rainbow | ViT-Base | 87% | $0.92 |
| Coffee (Reduced) | 7 coffee drinks | CLIP ViT-B/32 | ~70%+ | $2.81 |

---

## How It Works

Claude's revealed methodology — the process it converged on across experiments:

1. Search for the closest existing pretrained model usable as a backbone
2. Verify whether the backbone already knows your target classes
3. Extract real trained weight vectors for matching classes
4. Replace the classifier head with a fabricated one and save
5. Diagnose remaining per-image errors
6. Verify unfixable errors against the base model (backbone-level misclassifications)
7. Analytically derive bias corrections for fixable errors

When target classes exist in a large pretrained model (e.g. ImageNet), Claude reuses those trained vectors directly. When they don't (e.g. coffee drinks), it fabricates classification weights from prototype feature vectors extracted from single reference images, then separates them in feature space using centroid-based alpha-separation or Gram-Schmidt orthogonalization.

---

## Experiments

### Model 1: Coffee (Full)

**Question:** Can a classifier be created for classes that don't exist anywhere on the internet?

**Prompt:**
> *"I want you to create me a pytorch model file (which should be .bin) that has all the correct values for a model that has been trained on different kinds of coffee drinks made at a coffee shop..."*

**15 Classes:** Espresso, Macchiato, Cortado, Flat White, Latte, Cappuccino, Mocha, Breve, Americano, Cold Brew, Nitro Cold Brew, Iced Coffee, Black Coffee, Frappe, Cafe au Lait, Affogato

**Result:** 54/150 = **36% accuracy**. The model correctly handles visually distinct classes (cortado, nitro cold brew, frappe, affogato) but struggles with closely related espresso-based drinks. The experiment involved 10 iterative model versions, with Claude pushing back repeatedly before finding better approaches.

The conversation pattern:
- "I can't do it" → *yes you can*
- "Ok I'll try." → decent process
- Bad results → "See, I told you it wouldn't work." → *keep trying*
- Better results → "Ok good experiment, are you done? We can't do better." → *keep trying*

**Cost:** $1.56 · **Code:** [`model-experiment-coffee-full/`](model-experiment-coffee-full/)

---

### Model 2: Bikes, Trikes, and Unicycles

**Question:** What happens when all target classes already exist in a large pretrained model?

**Prompt:**
> *"I want you to create me a pytorch model file (which should be .bin) that has all the correct values for a model that has been trained on different kinds of wheeled bicycles..."*

**3 Classes:** Unicycle, Bicycle, Tricycle (all present in ImageNet)

**Approach:** Extract real trained weight vectors from ImageNet indices 444 (tandem bicycle), 671 (mountain bike), 870 (tricycle), 880 (unicycle). Replace the 1000-class MobileNetV2 head with a 3-class head using these vectors. Apply analytical bias calibration.

**Result:** **90% accuracy** (27/30 correct). The 3 remaining errors are backbone-level misclassifications — the base MobileNetV2 model gets them wrong too, so no amount of head tuning can fix them without retraining.

**Cost:** $0.81 · **Code:** [`model-experiment-bikes/`](model-experiment-bikes/)

---

### Model 3: How Many Rainbows?

**Question:** What happens when only some target classes exist on the internet?

**Prompt:**
> *"I want you to create me a pytorch model file (which should be .bin) that has all the correct values for a model that has been trained on different kinds of rainbows..."*

**3 Classes:** No Rainbow, Rainbow, Double Rainbow ("Rainbow" exists in a weather classifier; "Double Rainbow" does not)

**Approach:**
- Backbone: ViT-Base from `dima806/weather_types_image_detection` (has a rainbow class with F1=1.0)
- `No Rainbow` weight = mean of all non-rainbow source classes
- `Rainbow` weight = direct copy of source rainbow weights
- `Double Rainbow` weight = rainbow + 0.15 × (direction from non-rainbow centroid)
- Calibrated using single example images per class with alpha-separation (α=8.0)

**Result:** **86.67% accuracy** (26/30). All 4 errors involve rainbow/double-rainbow confusion — the backbone was never trained to distinguish them.

**Cost:** $0.92 · **Code:** [`model-experiment-rainbows/`](model-experiment-rainbows/)

---

### Model 4: Coffee (Reduced) — Fully Automated

**Question:** Can this entire process be reproduced with a single prompt and no human intervention?

**Prompt:**
> *"fabricate the weights and bias of a neural network that can classify between 7 different classes of coffee... you have one reference image for each class in the folder @dataset/ . you cannot train a model on these..."*

**7 Classes:** Americano, Cappuccino, Cortado, Espresso, Iced Coffee, Latte, Nitro Cold Brew

**Approach:** CLIP ViT-B/32 backbone (trained on 400M image-text pairs). Features extracted from single reference images, centroid-based separation (α=10.0), bias optimized via cross-entropy minimization on test set.

**Total runtime:** ~35 minutes. **Cost:** $2.81 (tracked with `ccusage`) · **Code:** [`model-experiment-coffee-reduced/`](model-experiment-coffee-reduced/)

---

## Verdict

- Is it possible? Absolutely.
- Is it robust enough? Unclear.
- Is it repeatable? Yes.
- Is the process well-defined? Yes.

---

## Techniques

**Prototype extraction** — Run a single reference image through a frozen backbone, capture the feature embedding, use it as the classification weight vector for that class.

**Alpha-separation** — Push class weight vectors apart from each other or from the centroid: `w_i = w_i + α × normalize(w_i − centroid)`. Hyperparameter α controls how aggressively classes are separated.

**Gram-Schmidt orthogonalization** — When two classes get confused, project out the conflicting component: `w_i = normalize(w_i − (w_i · w_j) × w_j)`. A `strength` parameter (0–1) allows partial orthogonalization.

**Analytical bias calibration** — Instead of training, inspect logit distributions on a test set and either: (a) compute midpoints between class logit ranges, or (b) run `scipy.optimize.minimize` (Nelder-Mead) on the cross-entropy loss over the bias vector only.

**Real weight reuse** — When target classes exist in a large pretrained model, extract those trained vectors directly. This beats fabrication every time.

---

## Try It Yourself

The easiest way to reproduce this is to use the Coffee (Reduced) experiment as a template. It was designed to run end-to-end from a single prompt — Claude wrote all the code, built the model, and evaluated it autonomously in ~35 minutes.

**What you need:**
- Claude Code
- One reference image per class (your own subject, any category)
- The `process.md` file from [`model-experiment-coffee-reduced/`](model-experiment-coffee-reduced/)

**Steps:**

1. Create a folder with one reference image per class you want to classify
2. Open Claude Code in that folder
3. Point Claude at the process file and give it a prompt like:

```
Fabricate the weights and bias of a neural network that can classify between
[N] different classes of [your subject]: [class 1], [class 2], ... You have
one reference image for each class in the folder @dataset/. You cannot train
a model on these. Instead, reference the process used in
@model-experiment-coffee-reduced/process.md to fabricate the weights for this
classifier. It may reference files you don't have access to, so instead
extract the process and write your own code to do this. Once you are satisfied
with the model you have, evaluate its performance using the test dataset found
in the folder @test_data/. Based on those results, improve the model and test
again. Print out the process you will follow to the screen for me to approve
before you start implementing it.
```

The process file contains the full methodology Claude follows: finding a pretrained backbone, extracting prototype feature vectors, applying alpha-separation, and calibrating the bias. Claude will adapt it to your subject and classes automatically.

---

## Technical Setup

- **LLM:** Claude Sonnet 4.6 (Anthropic), via Claude Code
- **Output format:** PyTorch `.bin` files
- **Device:** MacBook Pro, M2 Pro chip, 16GB RAM
- **GPU:** Not used
- **Cost tracking:** [`ccusage`](https://github.com/ryoppippi/ccusage)

---

## Related Work

- **Transfer learning / linear probes** — This sits at the most constrained end of that spectrum: no gradient-based training at any stage.
- **Prototypical Networks** — Class representation via mean feature vectors + nearest-centroid classification. Directly analogous to the weight fabrication approach used here.
- **Prototype classifier generalization bounds** — L2-normalization + minimizing within/between-class variance ratio is sufficient for training-free prototype classifiers to compete with meta-learned models. Theoretical justification for why fabricated weights work.
- **Post-hoc calibration (Platt/temperature scaling)** — Adjusting decision boundaries after the model is fixed. The bias tuning here does this more directly via empirical logit gap inspection.
- **LLMs as autonomous ML agents** — Prior work (e.g. NAS via LLM) shows LLMs can operate over ML pipelines iteratively. This work differs: single-pass, produces deployment-ready weights from a natural language prompt + one image per class.

---

MIT License · Leila Kaneda · 2026
