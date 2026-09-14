# Real-Time Financial Sentiment Extraction Using LLM Architectures

I built this project to classify financial news as **Negative**, **Neutral**, or **Positive** using five transformer models and a soft-voting ensemble.

This repository contains the implementation for our 2026 IGI Global Scientific Publishing book chapter:

> Lakshmi Harika Palivela, Shreyas Athinarapu, Greeshma Reddy Basireddy, and Tata Venkata Krishna Teja. “Real-Time Financial Sentiment Extraction Using Large Language Model (LLM) Architectures.” Chapter 6 in *Harnessing Large Language Models for Enhanced Business Analytics*. DOI: [10.4018/979-8-3693-6690-5.ch006](https://doi.org/10.4018/979-8-3693-6690-5.ch006)

[Official IGI Global chapter page](https://www.igi-global.com/chapter/real-time-financial-sentiment-extraction-using-large-language-model-llm-architectures/411407)

## Published results

In our published chapter, we reported the following results in Table 1. I have included these values as the official research results; generated checkpoints and local rerun reports are not committed to this repository.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| FinBERT | 89.24% | 0.88 | 0.89 | 0.89 |
| Twitter-RoBERTa | 89.62% | 0.92 | 0.86 | 0.89 |
| DistilBERT | 88.63% | 0.87 | 0.88 | 0.88 |
| FinBERT-tone | 87.42% | 0.85 | 0.86 | 0.86 |
| ELECTRA | 85.03% | 0.84 | 0.86 | 0.85 |
| **Soft Voting Ensemble** | **90.00%** | **0.91** | **0.90** | **0.90** |

Fresh training runs can vary with hardware, library versions, random initialization, and data ordering. I have therefore clearly labelled the table above as our published results rather than a guarantee for every rerun.

## Models

| Project name | Hugging Face checkpoint |
|---|---|
| FinBERT-tone | `yiyanghkust/finbert-tone` |
| Twitter-RoBERTa | `cardiffnlp/twitter-roberta-base-sentiment` |
| FinBERT | `ProsusAI/finbert` |
| DistilBERT | `distilbert-base-uncased` |
| ELECTRA | `google/electra-small-discriminator` |

I fine-tune each model independently. The ensemble follows the approach used in our published work: it averages the five output-logit vectors before applying softmax.

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

For the published configuration, I used 10 epochs, batch size 16, learning rate `2e-5`, AdamW, an 80/20 stratified split, and seed 42.

I assembled the research dataset by combining multiple Kaggle sources. I have not redistributed the CSV in this public repository because the original source URLs and licences must be fully documented first. To reproduce the training, place your legally obtained combined dataset at `data/financial_news.csv` and see `data/README.md` for the required schema.

```powershell
python -m src.train_ensemble --epochs 10 --batch-size 16 --learning-rate 2e-5 --seed 42
```

Checkpoints are written under `report_models/`. Existing completed model directories are skipped unless `--force` is supplied.

## Evaluate the ensemble

```powershell
python -m src.evaluate_ensemble
```

This generates individual-model and ensemble metrics under `reports/`. I keep these generated artifacts out of Git so that the repository contains the reproducible code rather than machine-specific outputs.

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

If you use my implementation or build on our research, please cite the published chapter using the DOI above. I have not redistributed the complete chapter PDF in this repository.

## License

I have not selected an open-source license yet. Until I add one, the source code remains under standard copyright protection.

