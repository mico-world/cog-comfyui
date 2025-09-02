from dataclasses import asdict, dataclass
from enum import Enum
import os
import json
import yaml
from pathlib import Path
from typing import Any, Dict, List, Literal

class ModelTypeEnum(str, Enum):
    audio_encoders = "audio_encoders"
    checkpoints = "checkpoints"
    clip = "clip"
    clip_vision = "clip_vision"
    configs = "configs"
    controlnet = "controlnet"
    diffusers = "diffusers"
    diffusion_models = "diffusion_models"
    embeddings = "embeddings"
    gligen = "gligen"
    hypernetworks = "hypernetworks"
    ipadapter = "ipadapter"
    loras = "loras"
    model_patches = "model_patches"
    photomaker = "photomaker"
    style_models = "style_models"
    text_encoders = "text_encoders"
    unet = "unet"
    upscale_models = "upscale_models"
    vae = "vae"
    vae_approx = "vae_approx"


@dataclass
class ModelConfig:
    repo_id: str
    filename: str
    model_type: ModelTypeEnum
    subfolder: str = ""

class DataclassJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        if hasattr(obj, "__dataclass_fields__"):
            return asdict(obj)
        return super().default(obj)



def load_config_file(file_path: str) -> List[Dict[str, Any]]:
    """
    根据文件后缀加载单个配置文件
    支持 JSON 和 YAML
    返回统一的 List[Dict]
    """
    ext = Path(file_path).suffix.lower()
    with open(file_path, "r", encoding="utf-8") as f:
        if ext in (".yaml", ".yml"):
            data = yaml.safe_load(f) or []
        elif ext == ".json":
            data = json.load(f) or []
        else:
            raise ValueError(f"Unsupported config file type: {file_path}")

    if isinstance(data, dict):
        # dict → list  (防止写成 {key: {…}} 的情况)
        data = list(data.values())
    elif not isinstance(data, list):
        raise ValueError(f"Config file must be list or dict, got {type(data)} in {file_path}")

    return data


def merge_config_list(base: List[Dict[str, Any]], override: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    合并两个配置列表
    以 filename 为唯一键，override 覆盖 base
    """
    config_map = {item.get("filename"): item for item in base if "filename" in item}
    for item in override:
        filename = item.get("filename")
        if not filename:
            continue
        config_map[filename] = item
    return list(config_map.values())


def load_configs_from_folder(folder_path: str) -> List[Dict[str, Any]]:
    """
    加载文件夹下所有配置文件并合并
    支持递归子目录
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        return []

    merged_config: List[Dict[str, Any]] = []
    for file_path in sorted(folder.iterdir()):  # 按文件名顺序加载
        if file_path.is_file() and file_path.suffix.lower() in (".json", ".yaml", ".yml"):
            try:
                config = load_config_file(file_path)
                merged_config = merge_config_list(merged_config, config)
            except Exception as e:
                print(f"[WARN] Failed to load {file_path}: {e}")
        elif file_path.is_dir():
            config = load_configs_from_folder(file_path)
            merged_config = merge_config_list(merged_config, config)
    return [ModelConfig(**config) for config in merged_config]


if __name__ == "__main__":
    result = load_configs_from_folder("user_weights")
    with open("user_weights.json", "w", encoding="utf-8") as f:
        json.dump(result, f, cls=DataclassJSONEncoder, indent=4, ensure_ascii=False)