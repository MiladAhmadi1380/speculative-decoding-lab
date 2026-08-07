import torch


# Target and draft must use a compatible tokenizer and vocabulary.
TARGET_MODEL_NAME = "facebook/opt-350m"
DRAFT_MODEL_NAME = "facebook/opt-125m"

# CPU-only environment.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Use the full distributions during the first controlled experiment.
# This keeps the proposal distribution consistent with the
# probabilities used by the accept/reject calculation.
TEMPERATURE = 1.0
TOP_K = 0
TOP_P = 1.0