# Cross-Document Coreference Resolution (CDCR) with LLMs

The project measures how well an LLM solves cross-document coreference resolution (CDCR) out of the box, without fine-tuning, on the twelve datasets of the [uCDCR](https://github.com/anastasia-zhukova/uCDCR) collection. Predictions are scored per subtopic with the CoNLL scorer (MUC, B³, CEAFe and their CoNLL F1 average).

## Approaches

| Approach | Config | What it does |
|---|---|---|
| Head-lemma baseline | `approach=baseline_head_lemma` | Links two mentions if they share a normalized head lemma. No LLM. |
| Oracle baseline | `approach=baseline_oracle` | Labels every candidate pair with its gold answer, which gives the recall ceiling of the candidate set. No LLM. |
| Constant baseline | `approach=baseline_constant_true` | Predicts every candidate pair as coreferent. No LLM. |
| Random few-shot | `approach=random_few_shot` | Asks the LLM about each candidate pair, with labeled pairs sampled from the train split as examples. Pairwise decisions are closed into clusters with union-find. |
| End-to-end | `approach=e2e` | Gives the LLM every mention of a subtopic with its context and asks for the coreference chains directly. |
| End-to-end, grouped | `approach=e2e_grouped` | Same as `e2e`, but renders each document once with its mentions tagged inline, which uses fewer tokens. |

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync
cp .env.example .env
```

Fill in `.env`. Keep every variable set; an endpoint a run does not use can keep its placeholder value:

- `LLM_API_URL`, `LLM_API_TOKEN`, `LLM_MODEL_NAME`: the default OpenAI-compatible endpoint (`llm@approach.llm=default`).
- `DEEPSEEK_API_KEY`: used by the LLM approaches, which run against DeepSeek by default (`configs/llm/deepseek.yaml`).
- `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`: your own MLflow tracking server. Every run, the baselines included, logs its metrics and predictions there.

To use another model or endpoint, copy `configs/llm/default.yaml` to `configs/llm/<name>.yaml`, fill in `model`, `api_url` and `api_key_env`, and select it with `llm@approach.llm=<name>`.

## Data

Download the uCDCR datasets from [Hugging Face](https://huggingface.co/datasets/AnZhu/uCDCR) into `data/uCDCR/`:

```bash
uvx --from huggingface_hub hf download AnZhu/uCDCR --repo-type dataset --local-dir data/uCDCR
```

The loader expects `data/uCDCR/<Dataset>/{train,val,test}/{entity,event}_mentions.json`, with the dataset folder names CD2CR, CEREC, ECBplus, ECBplusMETAm, FCC, GVC, HyperCoref, MEANTIME, NewsWCL50, NiDENT, NP4E and WECEng. Only text-derived fields of each mention (context, tokens, head, POS, NER, and sentence, document and subtopic ids) are passed to the models. Gold annotations such as `coref_chain`, `is_singleton`, `description` and the mention types are used only for scoring and for the few-shot examples drawn from the train split.

Each dataset keeps the license of its original release; see the uCDCR repository for details.

## Running experiments

Experiments are configured with [Hydra](https://hydra.cc/). The config groups live in `configs/`:

- `data`: the dataset and mention kind. Datasets with both entity and event mentions run them pooled (`ECBplus_both`); single-kind variants are under `data=single/...`.
- `approach`: one of the approaches above.
- `candidates`: candidate pairs for the pairwise approaches, `full` (all pairs in a subtopic) or `sampled` (at most 500 pairs).
- `llm`: the endpoint, `default` or `deepseek`, nested in the approach and overridden with `llm@approach.llm=<name>`.
- `run`: `default`, or `smoke` for a quick 10-prediction check.

```bash
# one dataset
uv run python scripts/run_experiment.py data=GVC approach=e2e
uv run python scripts/run_experiment.py data=single/ECBplus_entity approach=random_few_shot \
    approach.few_shot.num_examples=8 approach.few_shot.strategy=SAME_DATASET

# baselines, no LLM calls
uv run python scripts/run_experiment.py data=GVC approach=baseline_head_lemma candidates=full

# quick check
uv run python scripts/run_experiment.py data=CD2CR run=smoke
```

The `Makefile` runs a sweep over all datasets, one after another:

```bash
make list                                   # datasets in sweep order
make run-all ARGS="approach=e2e_grouped"    # sweep with Hydra overrides
make run-GVC                                # one dataset
```

NiDENT and NP4E are left out of the default sweep; pass `EXCLUDE=""` to include them. The sweep stops at the first failing dataset unless `KEEP_GOING=1` is set.

Runs checkpoint their predictions to `outputs/checkpoints/`, so an interrupted run resumes where it stopped. Hydra writes its logs to `outputs/<date>/<time>/`. Metrics, predictions and the run log go to MLflow, with one experiment per approach.

## Tests

```bash
uv run pytest
```

The tests import the project settings, so `.env` must exist (a copy of `.env.example` is enough). Tests that read the real datasets are skipped when `data/uCDCR/` is missing.

## License

Apache License 2.0, see [LICENSE](LICENSE). The datasets are not part of this repository and keep their own licenses.
