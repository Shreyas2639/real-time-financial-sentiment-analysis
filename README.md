# Real-Time Financial Sentiment Analysis

A five-model financial sentiment analysis system that classifies financial text as **Negative**, **Neutral**, or **Positive** through a soft-voting ensemble of pretrained transformer language models.

## Project overview

I built this project as the implementation of our published research on real-time financial sentiment extraction. It provides:

- Independent fine-tuning for five pretrained transformer language models
- Soft-voting ensemble inference by averaging model logits before softmax
- Individual-model and ensemble evaluation
- A FastAPI service for real-time predictions
- CUDA support through PyTorch for faster training and inference

## Models

| Model | Hugging Face checkpoint |
|---|---|
| FinBERT-tone | `yiyanghkust/finbert-tone` |
| Twitter-RoBERTa | `cardiffnlp/twitter-roberta-base-sentiment` |
| FinBERT | `ProsusAI/finbert` |
| DistilBERT | `distilbert-base-uncased` |
| ELECTRA | `google/electra-small-discriminator` |

Each model is fine-tuned independently. During ensemble inference, the implementation takes the output logits from all five models, calculates their equal-weight average, and applies softmax once to obtain the final class probabilities.

## Published research results

The following results are reproduced from Table 1 of our published chapter. They are the results reported in the publication—not metrics generated from a fresh run of this repository.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| FinBERT | 89.24% | 0.88 | 0.89 | 0.89 |
| Twitter-RoBERTa | 89.62% | 0.92 | 0.86 | 0.89 |
| DistilBERT | 88.63% | 0.87 | 0.88 | 0.88 |
| FinBERT-tone | 87.42% | 0.85 | 0.86 | 0.86 |
| ELECTRA | 85.03% | 0.84 | 0.86 | 0.85 |
| **Soft Voting Ensemble** | **90.00%** | **0.91** | **0.90** | **0.90** |

Fresh runs may produce different values because of hardware, dependency versions, random initialization, and data ordering. The evaluation scripts save newly generated metrics under `reports/`; these local outputs are ignored by Git and should not be presented as the published results unless they match an independently verified reproduction.

## Dataset

I created the research dataset by combining multiple Kaggle datasets into a single labelled collection. The published experiment used 10,688 financial sentences with the following columns:

- `Sentence`
- `Sentiment` with the values `negative`, `neutral`, or `positive`

The combined CSV is not distributed in this repository. Users must obtain the source datasets lawfully and comply with their original licences and attribution requirements. Place the prepared file at `data/financial_news.csv`; see [`data/README.md`](data/README.md) for the expected format.

## Requirements

- Python 3.11 recommended
- PyTorch 2.2 or later
- A CUDA-compatible NVIDIA GPU is recommended for training but is not required
- Sufficient storage for five transformer checkpoints and generated training artifacts
- The prepared dataset described above

All Python dependencies and supported version ranges are listed in [`requirements.txt`](requirements.txt).

## Installation

On Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For GPU training, install the CUDA-enabled PyTorch build compatible with your NVIDIA driver. Confirm that PyTorch detects the GPU:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Training

The published configuration uses 10 epochs, batch size 16, learning rate `2e-5`, AdamW, an 80/20 stratified split, and seed 42.

```powershell
python -m src.train_ensemble --epochs 10 --batch-size 16 --learning-rate 2e-5 --seed 42
```

Trained checkpoints are written to `report_models/`. Completed model directories are skipped automatically unless `--force` is supplied.

## Evaluation

```powershell
python -m src.evaluate_ensemble
```

The command evaluates the five individual models and their ensemble, then writes the newly generated metrics to `reports/`.

## API

Start the FastAPI development server:

```powershell
uvicorn src.api:app --reload
```

Open `http://127.0.0.1:8000/docs` and use `POST /predict/ensemble` for the five-model ensemble or `POST /predict` for the optional single-model endpoint.

Example request:

```json
{
  "text": "The company reported stronger revenue and raised its annual guidance."
}
```

Example single-model response:

```json
{
  "sentiment": "Neutral",
  "confidence": 0.5705,
  "probabilities": {
    "Negative": 0.4265,
    "Neutral": 0.5705,
    "Positive": 0.0031
  }
}
```

Available endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/models` | Show the readiness of all five ensemble models |
| `POST` | `/predict/ensemble` | Return the ensemble result and individual model predictions |
| `GET` | `/health` | Show the optional single-model service status |
| `POST` | `/predict` | Return an optional single-model prediction |

## Tests

```powershell
pytest -q
```

## Repository structure

```text
.
├── data/
│   └── README.md
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
├── CITATION.cff
├── ENSEMBLE.md
└── requirements.txt
```

## Research and publication

This repository accompanies our 2026 IGI Global Scientific Publishing book chapter:

> Lakshmi Harika Palivela, Shreyas Athinarapu, Greeshma Reddy Basireddy, and Tata Venkata Krishna Teja. “Real-Time Financial Sentiment Extraction Using Large Language Model (LLM) Architectures.” Chapter 6 in *Harnessing Large Language Models for Enhanced Business Analytics*.

- DOI: [`10.4018/979-8-3693-6690-5.ch006`](https://doi.org/10.4018/979-8-3693-6690-5.ch006)
- [Official IGI Global chapter page](https://www.igi-global.com/chapter/real-time-financial-sentiment-extraction-using-large-language-model-llm-architectures/411407)

## Citation

If you use my implementation or build on our research, please cite the published chapter above. Machine-readable citation metadata is available in [`CITATION.cff`](CITATION.cff). The complete chapter PDF is not redistributed here.

## License

> **Important:** I have not selected an open-source licence for this repository. Without a licence, the source code remains under standard copyright protection, and others generally do not have permission to reuse, modify, or redistribute it.

