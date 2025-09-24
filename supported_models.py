from typing import Dict, List
from collections import OrderedDict

from collators import COLLATORS
from datasets import TO_LOAD_IMAGE
from loaders import LOADERS


MODULE_KEYWORDS: Dict[str, Dict[str, List]] = {
    "llava-1.5": {
        "vision_encoder": ["vision_tower"],
        "vision_projector": ["multi_modal_projector"],
        "llm": ["language_model"]
    },
}


MODEL_HF_PATH = OrderedDict()

MODEL_FAMILIES = OrderedDict()


def register_model(model_id: str, model_family_id: str, model_hf_path: str) -> None:
    if model_id in MODEL_HF_PATH or model_id in MODEL_FAMILIES:
        raise ValueError(f"Duplicate model_id: {model_id}")
    MODEL_HF_PATH[model_id] = model_hf_path
    MODEL_FAMILIES[model_id] = model_family_id

register_model(
    model_id="llava-1.5-7b",
    model_family_id="llava-1.5",
    model_hf_path="llava-hf/llava-1.5-7b-hf"
)

# sanity check
for model_family_id in MODEL_FAMILIES.values():
    assert model_family_id in COLLATORS, f"Collator not found for model family: {model_family_id}"
    assert model_family_id in LOADERS, f"Loader not found for model family: {model_family_id}"
    assert model_family_id in MODULE_KEYWORDS, f"Module keywords not found for model family: {model_family_id}"
    assert model_family_id in TO_LOAD_IMAGE, f"Image loading specification not found for model family: {model_family_id}"


if __name__ == "__main__":
    temp = "Model ID"
    ljust = 30
    print("Supported models:")
    print(f"  {temp.ljust(ljust)}: HuggingFace Path")
    print("  ------------------------------------------------")
    for model_id, model_hf_path in MODEL_HF_PATH.items():
        print(f"  {model_id.ljust(ljust)}: {model_hf_path}")
