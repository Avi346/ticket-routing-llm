# config.py
# Single source of truth for the whole project.
# Every other script imports from here, so no two scripts can disagree
# about a seed, a budget, a model name or a path.

from pathlib import Path

# ---------- Paths ----------
ROOT = Path(__file__).parent
SPLITS_DIR = ROOT / "splits"      # saved train/dev/test files (Phase 1)
RESULTS_DIR = ROOT / "results"    # one predictions CSV per run (later phases)

# ---------- Seeds ----------
SEEDS = [0, 1, 2]                 # used to draw the training subsets
SPLIT_SEED = 42                   # used once for dev/test sampling, never changes

# ---------- Test / dev sizes ----------
TEST_SIZE = 1000                  # fixed test subset per dataset
HEADLINE_TEST_SIZE = 500          # v3.1: headline comparisons use the first 500
DEV_SUBSAMPLE_SIZE = 300          # fixed dev subsample for prompt wording

# ---------- Datasets ----------
# "budgets" = labeled examples PER CLASS.
# One dict per dataset; all code reads from here, so there is one code path.
DATASETS = {
    "banking77": {
        "hf_name": "PolyAI/banking77",
        "hf_config": None,
        "text_column": "text",
        "label_column": "label",
        "budgets": [2, 5, 10, 20],            # 154 to 1,540 total
        "dev_source": "train",                # dev is carved out of train...
        "dev_per_class": 10,                  # ...10 examples per class
        "test_source": "test",
    },
    "emotion": {
        "hf_name": "dair-ai/emotion",
        "hf_config": "split",
        "text_column": "text",
        "label_column": "label",
        "budgets": [10, 30, 100, 300],        # 60 to 1,800 total
        "dev_source": "validation",           # official validation split is the dev set
        "dev_per_class": None,                # not used for emotion
        "test_source": "test",
    },
}

# ---------- Models ----------
# Verify each name against current docs on the day you use it.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"    # baseline 1, retrieval (B), near-duplicate check
ENCODER_MODEL = "microsoft/deberta-v3-base"   # baseline 2
LLM_MODEL = "unsloth/Qwen3.5-4B"              # systems A, B, C

# ---------- Leakage check ----------
NEAR_DUPLICATE_THRESHOLD = 0.95               # cosine similarity above this gets flagged