import torch
import folder_paths
import comfy.sd

# Attachment key under which the "with Name" loaders store the model name on the MODEL.
# ModelPatcher.clone() copies attachments, so the name survives LoRAs and other patch nodes.
MODEL_NAME_ATTACHMENT = "image_saver_model_name"

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

        out[0].set_attachments(MODEL_NAME_ATTACHMENT, ckpt_name)

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
        model.set_attachments(MODEL_NAME_ATTACHMENT, unet_name)
        return (model, unet_name)


class GetModelName:
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model_name",)
    OUTPUT_TOOLTIPS = ("name of the model file, as selected in the loader; empty if unknown",)
    FUNCTION = "get_model_name"

    CATEGORY = "ImageSaver/utils"
    DESCRIPTION = ("Outputs the name of the model that was loaded by \"Checkpoint Loader with Name\" or \"UNet loader with Name\". "
                   "Connect the same MODEL that goes into the sampler, so the name follows switches and bypassed branches. "
                   "Empty for models from other loaders. After merging models, it is the name of the first model.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "the MODEL connected to the sampler"}),
            }
        }

    def get_model_name(self, model):
        return (model.get_attachment(MODEL_NAME_ATTACHMENT) or "",)
