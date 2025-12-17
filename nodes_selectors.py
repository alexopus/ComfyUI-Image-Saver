from typing import Any
import comfy


class AnyToString:
    """Converts any input type to a string. Useful for connecting sampler/scheduler outputs from various custom nodes."""

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    OUTPUT_TOOLTIPS = ("String representation of the input",)
    FUNCTION = "convert"
    CATEGORY = "ImageSaver/utils"
    DESCRIPTION = "Converts any input type to string"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "value": ("*",),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    def convert(self, value: Any) -> tuple[str,]:
        return (str(value),)


class InputParameters:
    RETURN_TYPES = ("INT", "INT", "FLOAT", comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS, "FLOAT")
    RETURN_NAMES = ("seed", "steps", "cfg", "sampler", "scheduler", "denoise")
    OUTPUT_TOOLTIPS = (
        "seed (INT)",
        "steps (INT)",
        "cfg (FLOAT)",
        "sampler (SAMPLERS)",
        "scheduler (SCHEDULERS)",
        "denoise (FLOAT)",
    )
    FUNCTION = "get_values"

    CATEGORY = "ImageSaver/utils"
    DESCRIPTION = "Combined node for seed, steps, cfg, sampler, scheduler and denoise."

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "The random seed used for creating the noise."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "The number of steps used in the denoising process."}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01, "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality."}),
                "sampler": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "The algorithm used when sampling, this can affect the quality, speed, and style of the generated output."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "The scheduler controls how noise is gradually removed to form the image."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The amount of denoising applied, lower values will maintain the structure of the initial image allowing for image to image sampling."}),
            }
        }

    def get_values(self, seed: int, steps: int, cfg: float, sampler: str, scheduler: str, denoise: float) -> tuple[int, int, float, str, str, float]:
        return (seed, steps, cfg, sampler, scheduler, denoise)
