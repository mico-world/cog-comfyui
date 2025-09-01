import json

import config

MODELS_PATH = config["MODELS_PATH"]

class WeightsManifest:
    manifest_path = "./user_weights.json"

    def __init__(self):
        self.weights_manifest = self._load_weights_manifest()
        self.weights_map = self.handle_weights_map()

    def _load_weights_manifest(self):
        try:
            with open(self.manifest_path) as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
        
    def handle_weights_map(self):
        result = {}

        for weights in self.weights_manifest:
            key = weights.get("filename")
            model_type = weights.get("model_type")
            local_dir = {
                "unet": f"{MODELS_PATH}/diffusion_models",
                "clip": f"{MODELS_PATH}/text_encoders",
            }.get(model_type, f"{MODELS_PATH}/{model_type}")
            value = dict(
                repo_id=weights.get("repo_id"),
                filename=weights.get("filename"),
                local_dir=weights.get("local_dir"),
                subfolder=weights.get("subfolder"),
                local_dir=local_dir
            )
            
            result[key] = value
        return result