# Real-Time Financial Sentiment Extraction Using LLM Architectures

Research implementation of a five-transformer financial sentiment system that classifies financial news as **Negative**, **Neutral**, or **Positive** and combines the model outputs through a soft-voting ensemble.

This repository accompanies the 2026 IGI Global Scientific Publishing book chapter:

> Lakshmi Harika Palivela, Shreyas Athinarapu, Greeshma Reddy Basireddy, and Tata Venkata Krishna Teja. “Real-Time Financial Sentiment Extraction Using Large Language Model (LLM) Architectures.” Chapter 6 in *Harnessing Large Language Models for Enhanced Business Analytics*. DOI: [10.4018/979-8-3693-6690-5.ch006](https://doi.org/10.4018/979-8-3693-6690-5.ch006)

[Official IGI Global chapter page](https://www.igi-global.com/chapter/real-time-financial-sentiment-extraction-using-large-language-model-llm-architectures/411407)

## Published results

The following values are the results reported in Table 1 of the published chapter. Generated checkpoints and local rerun reports are intentionally not committed to this repository.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| FinBERT | 89.24% | 0.88 | 0.89 | 0.89 |
| Twitter-RoBERTa | 89.62% | 0.92 | 0.86 | 0.89 |
| DistilBERT | 88.63% | 0.87 | 0.88 | 0.88 |
| FinBERT-tone | 87.42% | 0.85 | 0.86 | 0.86 |
| ELECTRA | 85.03% | 0.84 | 0.86 | 0.85 |
| **Soft Voting Ensemble** | **90.00%** | **0.91** | **0.90** | **0.90** |

Fresh training runs can vary with hardware, library versions, random initialization, and data ordering. The table above is labelled as the published result rather than a guarantee for every rerun.

## Models

| Project name | Hugging Face checkpoint |
|---|---|
| FinBERT-tone | `yiyanghkust/finbert-tone` |
| Twitter-RoBERTa | `cardiffnlp/twitter-roberta-base-sentiment` |
| FinBERT | `ProsusAI/finbert` |
| DistilBERT | `distilbert-base-uncased` |
| ELECTRA | `google/electra-small-discriminator` |

Each model is fine-tuned independently. The ensemble implementation follows the published project code by averaging the five output-logit vectors before applying softmax.

## Repository structure

```text
.
├── data/
│   └── README.md           # Dataset format and attribution requirements
├── src/
│   ├── api.py
│   ├── ensemble_service.py
│   ├── evaluate_ensemble.py
│   ├── model_registry.py
│   ├── model_service.py
│   ├── settings.py
│   ├── train.py
│   └── train_ensemble.py
├── tests/
├── .env.example
├── .gitignore
├── CITATION.cff
├── ENSEMBLE.md
└── requirements.txt
```

## Installation

Python 3.11 is recommended.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For GPU training, install a CUDA-enabled PyTorch build compatible with the computer's NVIDIA driver. Confirm it before training:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Train the published five-model configuration

The reproduction command uses 10 epochs, batch size 16, learning rate `2e-5`, AdamW, an 80/20 stratified split, and seed 42.

Before training, place your legally obtained combined Kaggle dataset at `data/financial_news.csv`. See `data/README.md` for the required schema. The CSV is not redistributed by this public repository until all original Kaggle sources and licences are documented.

```powershell
python -m src.train_ensemble --epochs 10 --batch-size 16 --learning-rate 2e-5 --seed 42
```

Checkpoints are written under `report_models/`. Existing completed model directories are skipped unless `--force` is supplied.

## Evaluate the ensemble

```powershell
python -m src.evaluate_ensemble
```

This generates individual-model and ensemble metrics under `reports/`. These generated artifacts are ignored by Git.

## Run the API

```powershell
uvicorn src.api:app --reload
```

Open `http://127.0.0.1:8000/docs` and use `POST /predict/ensemble`.

Example request:

```json
{
  "text": "The company reported stronger revenue and raised its annual guidance."
}
```

Useful endpoints:

- `GET /models` - readiness of all five ensemble models
- `POST /predict/ensemble` - ensemble prediction plus individual predictions
- `GET /health` - optional single-model endpoint status
- `POST /predict` - optional single-model prediction

## Tests

```powershell
pytest -q
```

## Citation

If this implementation supports academic work, cite the published chapter using the DOI above. The complete chapter PDF is not redistributed in this repository.

## License

No open-source license has been selected yet. Until the repository owner adds one, the source code remains under standard copyright protection.

