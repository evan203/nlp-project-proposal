import json
import os
from pipeline.model_utils.model_base import ModelBase

def _detect_model_family(model_path: str) -> str:
    """Detect model family from path string or config.json."""
    path_lower = model_path.lower()
    if 'qwen' in path_lower:
        return 'qwen'
    if 'llama-3' in path_lower or 'llama3' in path_lower:
        return 'llama3'
    if 'llama' in path_lower:
        return 'llama2'
    if 'gemma' in path_lower:
        return 'gemma'
    if 'yi' in path_lower:
        return 'yi'

    # Fall back to reading config.json for local model directories
    config_path = os.path.join(model_path, "config.json")
    if os.path.isfile(config_path):
        try:
            with open(config_path) as f:
                config = json.load(f)
            arch = config.get("architectures", [""])[0].lower()
            model_type = config.get("model_type", "").lower()
            if "llama" in arch or "llama" in model_type:
                # Distinguish Llama-3 from Llama-2 by vocab size or model_type
                vocab = config.get("vocab_size", 0)
                if vocab >= 128000:
                    return "llama3"
                return "llama2"
            if "qwen" in arch or "qwen" in model_type:
                return "qwen"
            if "gemma" in arch or "gemma" in model_type:
                return "gemma"
        except (json.JSONDecodeError, KeyError, IndexError):
            pass

    return "unknown"

def construct_model_base(model_path: str) -> ModelBase:
    family = _detect_model_family(model_path)

    if family == 'qwen':
        from pipeline.model_utils.qwen_model import QwenModel
        return QwenModel(model_path)
    if family == 'llama3':
        from pipeline.model_utils.llama3_model import Llama3Model
        return Llama3Model(model_path)
    elif family == 'llama2':
        from pipeline.model_utils.llama2_model import Llama2Model
        return Llama2Model(model_path)
    elif family == 'gemma':
        from pipeline.model_utils.gemma_model import GemmaModel
        return GemmaModel(model_path)
    elif family == 'yi':
        from pipeline.model_utils.yi_model import YiModel
        return YiModel(model_path)
    else:
        raise ValueError(f"Unknown model family: {model_path}")
