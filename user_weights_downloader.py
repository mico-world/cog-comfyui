from contextlib import contextmanager
import shutil
from user_weights_manifest import WeightsManifest
import os

from huggingface_hub import hf_hub_download, login, logout

DATA_UNITS = ('B', 'KB', 'MB', 'GB')


class HFWeightsDownloader:
    def __init__(self, token: str = ""):
        self.weights_manifest = WeightsManifest()
        self.weights_map = self.weights_manifest.weights_map
        self.token = token

    def support_weights(self):
        return self.weights_map.keys()
    
    def set_hf_token(self, token: str):
        self.token = token

    @contextmanager
    def login(self):
        try:
            if self.token:
                print(f'HF login use: {self.token}')
                login(self.token)
            yield True
        finally:
            logout()
            print(f"huggingface logout")

    def download(self, weights: str):
        weights_params = self.weights_map.get(weights)
        self._download(**weights_params)

    def _download(self, repo_id: str, subfolder: str, filename: str, local_dir: str, data_units: int = 2):
        path = os.path.join(local_dir, filename)

        if os.path.isfile(path):
            print(f'✅ {filename} already exist')
            return path

        with self.login():
            print(f"Download Model: {filename} From HF Hub")
            params = dict(
                repo_id=repo_id,
                filename=filename,
                local_dir=local_dir
            )
            if subfolder:
                params.update(subfolder=subfolder)
            temp_path = hf_hub_download(**params)
            path = temp_path

            if subfolder:
                path = os.path.join(local_dir, filename)
                shutil.move(temp_path, path)
                
            file_size_bytes = os.path.getsize(os.path.join(path))
            for i in range(data_units):
                file_size_bytes /= 1024
            print(
                f"✅ diffusion model {filename} download success, size: {file_size_bytes:.2f}{DATA_UNITS[data_units]}")
            return path
