import os
from datetime import datetime
from sys import float_info
import json
import piexif
import piexif.helper
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import numpy as np
import folder_paths
import comfy.sd
from .utils import get_sha256
from .prompt_metadata_extractor import PromptMetadataExtractor
from nodes import MAX_RESOLUTION


def parse_checkpoint_name(ckpt_name):
    return os.path.basename(ckpt_name)


def parse_checkpoint_name_without_extension(ckpt_name):
    return os.path.splitext(parse_checkpoint_name(ckpt_name))[0]


def handle_whitespace(string: str):
    return string.strip().replace("\n", " ").replace("\r", " ").replace("\t", " ")


def get_timestamp(time_format):
    now = datetime.now()
    try:
        timestamp = now.strftime(time_format)
    except:
        timestamp = now.strftime("%Y-%m-%d-%H%M%S")

    return timestamp

def save_json(image_info, filename):
    try:
        workflow = (image_info or {}).get('workflow')
        if workflow is None:
            print('No image info found, skipping saving of JSON')
        with open(f'{filename}.json', 'w') as workflow_file:
            json.dump(workflow, workflow_file)
            print(f'Saved workflow to {filename}.json')
    except Exception as e:
        print(f'Failed to save workflow as json due to: {e}, proceeding with the remainder of saving execution')


def make_pathname(filename, seed, modelname, counter, time_format, sampler_name, steps, cfg, scheduler, denoise):
    filename = filename.replace("%date", get_timestamp("%Y-%m-%d"))
    filename = filename.replace("%time", get_timestamp(time_format))
    filename = filename.replace("%model", parse_checkpoint_name(modelname))
    filename = filename.replace("%seed", str(seed))
    filename = filename.replace("%counter", str(counter))
    filename = filename.replace("%sampler_name", sampler_name)
    filename = filename.replace("%steps", str(steps))
    filename = filename.replace("%cfg", str(cfg))
    filename = filename.replace("%scheduler", scheduler)
    filename = filename.replace("%basemodelname", parse_checkpoint_name_without_extension(modelname))
    filename = filename.replace("%denoise", str(denoise))
    return filename

def make_filename(filename, seed, modelname, counter, time_format, sampler_name, steps, cfg, scheduler, denoise):
    filename = make_pathname(filename, seed, modelname, counter, time_format, sampler_name, steps, cfg, scheduler, denoise)
    return get_timestamp(time_format) if filename == "" else filename

class SeedGenerator:
    RETURN_TYPES = ("INT",)
    FUNCTION = "get_seed"
    CATEGORY = "ImageSaver/utils"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})}}

    def get_seed(self, seed):
        return (seed,)

class StringLiteral:
    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_string"
    CATEGORY = "ImageSaver/utils"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"string": ("STRING", {"default": "", "multiline": True})}}

    def get_string(self, string):
        return (string,)

class SizeLiteral:
    RETURN_TYPES = ("INT",)
    FUNCTION = "get_int"
    CATEGORY = "ImageSaver/utils"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"int": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 8})}}

    def get_int(self, int):
        return (int,)

class IntLiteral:
    RETURN_TYPES = ("INT",)
    FUNCTION = "get_int"
    CATEGORY = "ImageSaver/utils"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"int": ("INT", {"default": 0, "min": 0, "max": 1000000})}}

    def get_int(self, int):
        return (int,)

class FloatLiteral:
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "get_float"
    CATEGORY = "ImageSaver/utils"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"float": ("FLOAT", {"default": 1.0, "min": float_info.min, "max": float_info.max, "step": 0.01})}}

    def get_float(self, float):
        return (float,)

class CfgLiteral:
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "get_float"
    CATEGORY = "ImageSaver/utils"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"float": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0})}}

    def get_float(self, float):
        return (float,)

class CheckpointLoaderWithName:
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING",)
    RETURN_NAMES = ("MODEL", "CLIP", "VAE", "modelname")
    FUNCTION = "load_checkpoint"

    CATEGORY = "ImageSaver/utils"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),}}

    def load_checkpoint(self, ckpt_name, output_vae=True, output_clip=True):
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))

        # add checkpoint name to the output tuple (without the ClipVisionModel)
        out = (*out[:3], ckpt_name)
        return out

class SamplerSelector:
    CATEGORY = 'ImageSaver/utils'
    RETURN_TYPES = (comfy.samplers.KSampler.SAMPLERS,)
    RETURN_NAMES = ("sampler_name",)
    FUNCTION = "get_names"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"sampler_name": (comfy.samplers.KSampler.SAMPLERS,)}}

    def get_names(self, sampler_name):
        return (sampler_name,)

class SchedulerSelector:
    CATEGORY = 'ImageSaver/utils'
    RETURN_TYPES = (comfy.samplers.KSampler.SCHEDULERS + ['AYS SDXL', 'AYS SD1', 'AYS SVD'],)
    RETURN_NAMES = ("scheduler",)
    FUNCTION = "get_names"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"scheduler": (comfy.samplers.KSampler.SCHEDULERS + ['AYS SDXL', 'AYS SD1', 'AYS SVD'],)}}

    def get_names(self, scheduler):
        return (scheduler,)

class ImageSaver:
    def __init__(self):
        self.output_dir = folder_paths.output_directory
        self.civitai_sampler_map = {
            'euler_ancestral': 'Euler a',
            'euler': 'Euler',
            'lms': 'LMS',
            'heun': 'Heun',
            'dpm_2': 'DPM2',
            'dpm_2_ancestral': 'DPM2 a',
            'dpmpp_2s_ancestral': 'DPM++ 2S a',
            'dpmpp_2m': 'DPM++ 2M',
            'dpmpp_sde': 'DPM++ SDE',
            'dpmpp_2m_sde': 'DPM++ 2M SDE',
            'dpmpp_3m_sde': 'DPM++ 3M SDE',
            'dpm_fast': 'DPM fast',
            'dpm_adaptive': 'DPM adaptive',
            'ddim': 'DDIM',
            'plms': 'PLMS',
            'uni_pc_bh2': 'UniPC',
            'uni_pc': 'UniPC',
            'lcm': 'LCM',
        }

    def get_civitai_sampler_name(self, sampler_name, scheduler):
        # based on: https://github.com/civitai/civitai/blob/main/src/server/common/constants.ts#L122
        if sampler_name in self.civitai_sampler_map:
            civitai_name = self.civitai_sampler_map[sampler_name]

            if scheduler == "karras":
                civitai_name += " Karras"
            elif scheduler == "exponential":
                civitai_name += " Exponential"

            return civitai_name
        else:
            if scheduler != 'normal':
                return f"{sampler_name}_{scheduler}"
            else:
                return sampler_name

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", ),
                "filename": ("STRING", {"default": f'%time_%basemodelname_%seed', "multiline": False}),
                "path": ("STRING", {"default": '', "multiline": False}),
                "extension": (['png', 'jpeg', 'webp'],),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "modelname": ("STRING", {"default": '', "multiline": False}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS + ['AYS SDXL', 'AYS SD1', 'AYS SVD'],),
            },
            "optional": {
                "positive": ("STRING", {"default": 'unknown', "multiline": True}),
                "negative": ("STRING", {"default": 'unknown', "multiline": True}),
                "seed_value": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "width": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 8}),
                "height": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 8}),
                "lossless_webp": ("BOOLEAN", {"default": True}),
                "quality_jpeg_or_webp": ("INT", {"default": 100, "min": 1, "max": 100}),
                "optimize_png": ("BOOLEAN", {"default": False}),
                "counter": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff }),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                "time_format": ("STRING", {"default": "%Y-%m-%d-%H%M%S", "multiline": False}),
                "save_workflow_as_json": ("BOOLEAN", {"default": False}),
                "embed_workflow": ("BOOLEAN", {"default": True}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_files"

    OUTPUT_NODE = True

    CATEGORY = "ImageSaver"

    def save_files(
            self,
            images,
            seed_value,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            modelname,
            quality_jpeg_or_webp,
            lossless_webp,
            optimize_png,
            width,
            height,
            counter,
            filename,
            path,
            extension,
            time_format,
            denoise,
            save_workflow_as_json=False,
            embed_workflow=True,
            prompt=None,
            extra_pnginfo=None,
    ):

        filename = make_filename(filename, seed_value, modelname, counter, time_format, sampler_name, steps, cfg, scheduler, denoise)
        path = make_pathname(path, seed_value, modelname, counter, time_format, sampler_name, steps, cfg, scheduler, denoise)
        ckpt_path = folder_paths.get_full_path("checkpoints", modelname)

        if ckpt_path:
            modelhash = get_sha256(ckpt_path)[:10]
        else:
            modelhash = ""

        metadata_extractor = PromptMetadataExtractor([positive, negative])
        embeddings = metadata_extractor.get_embeddings()
        loras = metadata_extractor.get_loras()
        civitai_sampler_name = self.get_civitai_sampler_name(sampler_name.replace('_gpu', ''), scheduler)

        extension_hashes = json.dumps(embeddings | loras | { "model": modelhash })
        basemodelname = parse_checkpoint_name_without_extension(modelname)
        comment = f"{handle_whitespace(positive)}\nNegative prompt: {handle_whitespace(negative)}\nSteps: {steps}, Sampler: {civitai_sampler_name}, CFG scale: {cfg}, Seed: {seed_value}, Size: {width}x{height}, Model hash: {modelhash}, Model: {basemodelname}, Hashes: {extension_hashes} Version: ComfyUI"
        output_path = os.path.join(self.output_dir, path)

        if output_path.strip() != '':
            if not os.path.exists(output_path.strip()):
                print(f'The path `{output_path.strip()}` specified doesn\'t exist! Creating directory.')
                os.makedirs(output_path, exist_ok=True)

        filenames = self.save_images(images, output_path, filename, comment, extension, quality_jpeg_or_webp, lossless_webp, optimize_png, prompt, extra_pnginfo, save_workflow_as_json,embed_workflow)

        subfolder = os.path.normpath(path)
        return {"ui": {"images": map(lambda filename: {"filename": filename, "subfolder": subfolder if subfolder != '.' else '', "type": 'output'}, filenames)}}

    def save_images(self, images, output_path, filename_prefix, comment, extension, quality_jpeg_or_webp, lossless_webp, optimize_png, prompt, extra_pnginfo, save_workflow_as_json,embed_workflow) -> list[str]:
        img_count = 1
        paths = list()
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            if images.size()[0] > 1:
                current_filename_prefix = "{}_{:02d}".format(filename_prefix, img_count)
            else:
                current_filename_prefix = filename_prefix

            if extension == 'png':
                metadata = PngInfo()
                metadata.add_text("parameters", comment)

                # embed workflow and prompt json only if embed_workflow is true
                if embed_workflow:
                    if prompt is not None:
                        metadata.add_text("prompt", json.dumps(prompt))
                    if extra_pnginfo is not None:
                        for x in extra_pnginfo:
                            metadata.add_text(x, json.dumps(extra_pnginfo[x]))

                filename = f"{current_filename_prefix}.png"
                img.save(os.path.join(output_path, filename), pnginfo=metadata, optimize=optimize_png)
            else:
                filename = f"{current_filename_prefix}.{extension}"
                file = os.path.join(output_path, filename)
                img.save(file, optimize=True, quality=quality_jpeg_or_webp, lossless=lossless_webp)
                exif_bytes = piexif.dump({
                    "Exif": {
                        piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(comment, encoding="unicode")
                    },
                })
                piexif.insert(exif_bytes, file)

            if save_workflow_as_json:
                save_json(extra_pnginfo, os.path.join(output_path, current_filename_prefix))

            paths.append(filename)
            img_count += 1
        return paths


NODE_CLASS_MAPPINGS = {
    "Checkpoint Loader with Name (Image Saver)": CheckpointLoaderWithName,
    "Image Saver": ImageSaver,
    "Sampler Selector (Image Saver)": SamplerSelector,
    "Scheduler Selector (Image Saver)": SchedulerSelector,
    "Seed Generator (Image Saver)": SeedGenerator,
    "String Literal (Image Saver)": StringLiteral,
    "Width/Height Literal (Image Saver)": SizeLiteral,
    "Cfg Literal (Image Saver)": CfgLiteral,
    "Int Literal (Image Saver)": IntLiteral,
    "Float Literal (Image Saver)": FloatLiteral,
}
