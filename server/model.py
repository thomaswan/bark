from transformers import AutoProcessor, AutoModel
from optimum.bettertransformer import BetterTransformer
import torch
import os

MODEL_USE = "suno/bark-small" if os.getenv("USE_SMALL", 'False').lower() in ('true', '1', 't') else "suno/bark"

def load(load_only=False):
    print(f'MODEL_USE: {MODEL_USE}')
    processor = AutoProcessor.from_pretrained(MODEL_USE)
    model = AutoModel.from_pretrained(MODEL_USE, torch_dtype=torch.float16)

    if not load_only:
        model.to("cuda")
        model = BetterTransformer.transform(model, keep_original_model=False)

    return processor, model
