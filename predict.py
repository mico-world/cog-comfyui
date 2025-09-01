# An example of how to convert a given API workflow into its own Replicate model
# Replace predict.py with this file when building your own workflow

import os
import mimetypes
import json
import shutil
import tarfile
from typing import List, Optional
import zipfile
from PIL import Image
from cog import BasePredictor, Input, Path
from comfyui import ComfyUI
from cog_model_helpers import optimise_images
from cog_model_helpers import seed as seed_helper

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"
ALL_DIRECTORIES = [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]

mimetypes.add_type("image/webp", ".webp")

# Save your example JSON to the same directory as predict.py
api_json_file = "workflow_api.json"

# Force HF offline
# os.environ["HF_DATASETS_OFFLINE"] = "1"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
# os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
IMAGE_TYPES = [".jpg", ".jpeg", ".png", ".webp"]
VIDEO_TYPES = [".mp4", ".mov", ".avi", ".mkv", ".webm"]


class Predictor(BasePredictor):
    def setup(self):
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)

    def filename_with_extension(self, input_file, prefix):
        extension = os.path.splitext(input_file.name)[1]
        return f"{prefix}{extension}"

    def handle_input_file(self, input_file: Path):
        file_extension = self.get_file_extension(input_file)

        if file_extension == ".tar":
            with tarfile.open(input_file, "r") as tar:
                tar.extractall(INPUT_DIR)
        elif file_extension == ".zip":
            with zipfile.ZipFile(input_file, "r") as zip_ref:
                zip_ref.extractall(INPUT_DIR)
        elif file_extension in IMAGE_TYPES + VIDEO_TYPES:
            shutil.copy(input_file, os.path.join(
                INPUT_DIR, f"input{file_extension}"))
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        print("====================================")
        print(f"Inputs uploaded to {INPUT_DIR}:")
        self.comfyUI.get_files(INPUT_DIR)
        print("====================================")

    def get_file_extension(self, input_file: Path) -> str:
        file_extension = os.path.splitext(input_file)[1].lower()
        if not file_extension:
            with open(input_file, "rb") as f:
                file_signature = f.read(4)
            if file_signature.startswith(b"\x1f\x8b"):  # gzip signature
                file_extension = ".tar"
            elif file_signature.startswith(b"PK"):  # zip signature
                file_extension = ".zip"
            else:
                try:
                    with Image.open(input_file) as img:
                        file_extension = f".{img.format.lower()}"
                        print(f"Determined file type: {file_extension}")
                except Exception as e:
                    raise ValueError(
                        f"Unable to determine file type for: {input_file}, {e}"
                    )
        return file_extension

    def predict(
        self,
        workflow_json: str = Input(
            default="",
        ),
        input_file: Optional[Path] = Input(
            default=None, description='Input files'),
        hf_token: str = Input(default="")
    ) -> List[Path]:
        """Run a single prediction on the model"""
        self.comfyUI.cleanup(ALL_DIRECTORIES)

        if hf_token:
            self.comfyUI.weights_downloader.hf_downloader.set_hf_token(hf_token)

        if input_file:
            self.handle_input_file(input_file)

        wf = self.comfyUI.load_workflow(workflow_json)

        self.comfyUI.connect()
        self.comfyUI.run_workflow(wf)

        output_directories = [OUTPUT_DIR]
        # optimised_files = optimise_images.optimise_video_files(
        #     output_format, output_quality, self.comfyUI.get_files(output_directories)
        # )

        return [Path(p) for p in self.comfyUI.get_files(output_directories)]
