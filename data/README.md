# Dataset

The published experiment used a combined dataset assembled from multiple Kaggle sources. The combined CSV is intentionally not redistributed here until every original Kaggle URL, dataset author, and licence has been documented.

To train locally, place a legally obtained copy at `data/financial_news.csv`. The published experiment's snapshot contains 10,688 labelled financial sentences with these columns:

- `Sentence`
- `Sentiment`, containing `negative`, `neutral`, or `positive`

The training and evaluation scripts preserve the row-level 80/20 stratified split used by the published experiment.

## Attribution checklist

Before committing a dataset, record for each Kaggle source:

- Dataset title and URL
- Dataset creator
- Licence name and link
- Any required attribution text
- Which columns or rows were incorporated

