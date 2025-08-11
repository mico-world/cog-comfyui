# An example of how to convert a given API workflow into its own Replicate model
# Replace predict.py with this file when building your own workflow

import os
import mimetypes
import json
import shutil
from typing import List, Optional
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
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

class Predictor(BasePredictor):
    def setup(self):
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)

        # Give a list of weights filenames to download during setup
        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())
        self.comfyUI.handle_weights(
            workflow,
            weights_to_download=[],
        )

    def filename_with_extension(self, input_file, prefix):
        extension = os.path.splitext(input_file.name)[1]
        return f"{prefix}{extension}"

    def handle_input_file(
        self,
        input_file: Path,
        filename: str = "image.png",
    ):
        shutil.copy(input_file, os.path.join(INPUT_DIR, filename))

    # Update nodes in the JSON workflow to modify your workflow based on the given inputs
    def update_workflow(self, workflow, **kwargs):
        # Below is an example showing how to get the node you need and update the inputs

        # positive_prompt = workflow["6"]["inputs"]
        # positive_prompt["text"] = kwargs["prompt"]

        # negative_prompt = workflow["7"]["inputs"]
        # negative_prompt["text"] = f"nsfw, {kwargs['negative_prompt']}"

        # sampler = workflow["3"]["inputs"]
        # sampler["seed"] = kwargs["seed"]
        pass

    def predict(
        self,
        workflow_json: str = Input(
            default="",
        ),
        input_file: Optional[Path] = Input(description='Input files'),
        output_format: str = Input(
            description="Format of the output images",
            choices=["mp4"],
            default="mp4"
        ),
        output_quality: int = optimise_images.predict_output_quality()
    ) -> List[Path]:
        """Run a single prediction on the model"""
        self.comfyUI.cleanup(ALL_DIRECTORIES)

        # Make sure to set the seeds in your workflow
        seed = seed_helper.generate(seed)

        
        workflow = json.loads(workflow_json)

        wf = self.comfyUI.load_workflow(workflow)
        self.comfyUI.connect()
        self.comfyUI.run_workflow(wf)

        output_directories = [OUTPUT_DIR]
        optimised_files = optimise_images.optimise_image_files(
            output_format, output_quality, self.comfyUI.get_files(output_directories)
        )
    

        return [Path(p) for p in optimised_files]
