from pathlib import Path

import torch
import folder_paths
import comfy.sd
from comfy_api.latest import io

class CheckpointLoaderWithName:
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE", "model_name")
    OUTPUT_TOOLTIPS = ("U-Net model (denoising latents)", "CLIP (Contrastive Language-Image Pre-Training) model (encoding text prompts)", "VAE (Variational autoencoder) model (latent<->pixel encoding/decoding)", "checkpoint name")
    FUNCTION = "load_checkpoint"

    CATEGORY = "ImageSaver/utils"
    DESCRIPTION = "Loads U-Net model, CLIP model and VAE model from a checkpoint file"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), {"tooltip": "checkpoint"}),
            }
        }

    def load_checkpoint(self, ckpt_name, output_vae=True, output_clip=True):
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))

        # add checkpoint name to the output tuple (without the ClipVisionModel)
        out = (*out[:3], ckpt_name)
        return out

class UNETLoaderWithName:
    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "filename")
    OUTPUT_TOOLTIPS = ("U-Net model (denoising latents)", "model filename")
    FUNCTION = "load_unet"

    CATEGORY = "ImageSaver/utils"
    DESCRIPTION = "Loads U-Net model and outputs it's filename"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],)
            }
        }

    def load_unet(self, unet_name, weight_dtype):
        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2

        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
        return (model, unet_name)


class GetModelFilename(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Get Model Filename (Image Saver)",
            display_name="Get Model Filename (Image Saver)",
            category="ImageSaver/utils",
            description="Reads the source filename from the connected MODEL after switches and patches. Returns empty text if the loader did not retain a source file.",
            inputs=[io.Model.Input("model", tooltip="Connect the same MODEL output that feeds your sampler.")],
            outputs=[io.String.Output("model_name", tooltip="Source filename with extension and any subfolder in the model directory.")],
        )

    @classmethod
    def execute(cls, model):
        # Core checkpoint and diffusion loaders store the source path as the first reload argument.
        reload_info = getattr(model, "cached_patcher_init", None)
        if not isinstance(reload_info, tuple) or len(reload_info) < 2:
            return io.NodeOutput("")
        args = reload_info[1]
        if not isinstance(args, tuple) or not args or not isinstance(args[0], (str, Path)) or not args[0]:
            return io.NodeOutput("")

        path = Path(args[0])
        if not path.is_absolute():
            return io.NodeOutput(str(path))

        for folder in ("checkpoints", "diffusion_models"):
            for directory in folder_paths.get_folder_paths(folder):
                root = Path(directory).absolute()
                if path.is_relative_to(root):
                    return io.NodeOutput(str(path.relative_to(root)))
        return io.NodeOutput(path.name)
