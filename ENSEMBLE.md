# Published Five-Model Ensemble

The project uses five independently fine-tuned transformer classifiers. Every
model receives the same text and produces probabilities in this fixed order:

1. Negative
2. Neutral
3. Positive

The published project calls the method soft voting and implements it by
averaging the five raw logit vectors. This reproduction follows that code: it
averages logits, applies softmax to the mean, and selects the class with the
highest resulting score. Individual predictions remain available for
transparency.

| Project key | Hugging Face checkpoint | Saved location |
|---|---|---|
| `yiyang_finbert` | `yiyanghkust/finbert-tone` | `report_models/yiyang_finbert` |
| `twitter_roberta` | `cardiffnlp/twitter-roberta-base-sentiment` | `report_models/twitter_roberta` |
| `prosus_finbert` | `ProsusAI/finbert` | `report_models/prosus_finbert` |
| `distilbert` | `distilbert-base-uncased` | `report_models/distilbert` |
| `electra` | `google/electra-small-discriminator` | `report_models/electra` |

All models use uniform weight, matching `np.mean(logits_list, axis=0)` in the
published implementation. The result reported in the chapter is 90.00%
accuracy, precision 0.91, recall 0.90, and F1-score 0.90.

Published chapter DOI: `10.4018/979-8-3693-6690-5.ch006`.
