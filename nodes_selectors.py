from typing import Any
import comfy

INSPIRE_SCHEDULERS = comfy.samplers.KSampler.SCHEDULERS + ['AYS SDXL', 'AYS SD1', 'AYS SVD', "GITS[coeff=1.2]", 'OSS FLUX', 'OSS Wan', 'OSS Chroma']
EFF_SCHEDULERS = comfy.samplers.KSampler.SCHEDULERS + ['AYS SD1', 'AYS SDXL', 'AYS SVD', 'GITS']

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


class WorkflowInputValue:
    """Extracts an input value from the workflow by node ID and input name."""

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("value",)
    OUTPUT_TOOLTIPS = ("Input value from the specified node",)
    FUNCTION = "get_input_value"
    CATEGORY = "ImageSaver/utils"
    DESCRIPTION = "Extract an input value from the workflow by node ID and input name"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "node_id": ("STRING", {"default": "", "multiline": False, "tooltip": "The ID of the node to extract from"}),
                "input_name": ("STRING", {"default": "", "multiline": False, "tooltip": "The name of the input to extract"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    def get_input_value(self, node_id: str, input_name: str, prompt: dict[str, Any] | None = None, extra_pnginfo: dict[str, Any] | None = None):
        if prompt is None:
            return (None,)

        # Verify the node exists in the workflow structure
        if extra_pnginfo and "workflow" in extra_pnginfo:
            workflow = extra_pnginfo["workflow"]
            node_exists = any(str(node.get("id")) == node_id for node in workflow.get("nodes", []))
            if not node_exists:
                print(f"WorkflowInputValue: Node {node_id} not found in workflow structure")
                return (None,)

        # Get the node from the prompt (execution values)
        node = prompt.get(node_id)
        if node is None:
            print(f"WorkflowInputValue: Node {node_id} not found in prompt")
            return (None,)

        # Get the inputs from the node
        inputs = node.get("inputs", {})
        if input_name not in inputs:
            print(f"WorkflowInputValue: Input '{input_name}' not found in node {node_id}")
            print(f"WorkflowInputValue: Available inputs: {list(inputs.keys())}")
            return (None,)

        value = inputs[input_name]
        return (value,)


class SamplerSelector:
    RETURN_TYPES = (comfy.samplers.KSampler.SAMPLERS, "STRING")
    RETURN_NAMES = ("sampler",                        "sampler_name")
    OUTPUT_TOOLTIPS = ("sampler (SAMPLERS)", "sampler name (STRING)")
    FUNCTION = "get_names"

    CATEGORY = 'ImageSaver/utils'
    DESCRIPTION = 'Provides one of the available ComfyUI samplers'

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "sampler (Comfy's standard)"}),
            }
        }

    def get_names(self, sampler_name: str) -> tuple[str, str]:
        return (sampler_name, sampler_name)

class SchedulerSelector:
    RETURN_TYPES = (comfy.samplers.KSampler.SCHEDULERS, "STRING")
    RETURN_NAMES = ("scheduler",                        "scheduler_name")
    OUTPUT_TOOLTIPS = ("scheduler (SCHEDULERS)", "scheduler name (STRING)")
    FUNCTION = "get_names"

    CATEGORY = 'ImageSaver/utils'
    DESCRIPTION = 'Provides one of the standard KSampler schedulers'

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "scheduler (Comfy's standard)"}),
            }
        }

    def get_names(self, scheduler: str) -> tuple[str, str]:
        return (scheduler, scheduler)

class SchedulerSelectorInspire:
    RETURN_TYPES = (INSPIRE_SCHEDULERS, "STRING")
    RETURN_NAMES = ("scheduler", "scheduler_name")
    OUTPUT_TOOLTIPS = ("scheduler (ComfyUI + Inspire Pack Schedulers)", "scheduler name (STRING)")
    FUNCTION = "get_names"

    CATEGORY = 'ImageSaver/utils'
    DESCRIPTION = 'Provides one of the KSampler (inspire) schedulers'

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "scheduler": (INSPIRE_SCHEDULERS, {"tooltip": "scheduler (Comfy's standard + extras)"}),
            }
        }

    def get_names(self, scheduler: str) -> tuple[str, str]:
        return (scheduler, scheduler)

class SchedulerSelectorEfficiency:
    RETURN_TYPES = (EFF_SCHEDULERS, "STRING")
    RETURN_NAMES = ("scheduler", "scheduler_name")
    OUTPUT_TOOLTIPS = ("scheduler (ComfyUI + Efficiency Pack Schedulers)", "scheduler name (STRING)")
    FUNCTION = "get_names"

    CATEGORY = 'ImageSaver/utils'
    DESCRIPTION = 'Provides one of the KSampler (Eff.) schedulers'

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "scheduler": (EFF_SCHEDULERS, {"tooltip": "scheduler (Comfy's standard + Efficiency nodes)"}),
            }
        }

    def get_names(self, scheduler: str) -> tuple[str, str]:
        return (scheduler, scheduler)


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
