import comfy.sd

class SamplerSelector:
    RETURN_TYPES = (comfy.samplers.KSampler.SAMPLERS, "STRING")
    RETURN_NAMES = ("sampler",                        "sampler_name")
    OUTPUT_TOOLTIPS = ("sampler (SAMPLERS)", "sampler name (STRING)")
    FUNCTION = "get_names"

    CATEGORY = 'ImageSaver/utils'
    DESCRIPTION = 'Provides one of the available ComfyUI samplers'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "sampler (Comfy's standard)"}),
            }
        }

    def get_names(self, sampler_name):
        return (sampler_name, sampler_name)

class SchedulerSelector:
    RETURN_TYPES = (comfy.samplers.KSampler.SCHEDULERS + ['AYS SDXL', 'AYS SD1', 'AYS SVD', 'GITS[coeff=1.2]'], "STRING")
    RETURN_NAMES = ("scheduler",                                                                                "scheduler_name")
    OUTPUT_TOOLTIPS = ("scheduler (SCHEDULERS + ['AYS SDXL', 'AYS SD1', 'AYS SVD', 'GITS[coeff=1.2]'])", "scheduler name (STRING)")
    FUNCTION = "get_names"

    CATEGORY = 'ImageSaver/utils'
    DESCRIPTION = 'Provides one of the standard ComfyUI plus some extra schedulers'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS + ['AYS SDXL', 'AYS SD1', 'AYS SVD', 'GITS[coeff=1.2]'], {"tooltip": "scheduler (Comfy's standard + extras)"}),
            }
        }

    def get_names(self, scheduler):
        return (scheduler, scheduler)

class SchedulerSelectorComfy:
    RETURN_TYPES = (comfy.samplers.KSampler.SCHEDULERS, "STRING")
    RETURN_NAMES = ("scheduler",                        "scheduler_name")
    OUTPUT_TOOLTIPS = ("scheduler (SCHEDULERS)", "scheduler name (STRING)")
    FUNCTION = "get_names"

    CATEGORY = 'ImageSaver/utils'
    DESCRIPTION = 'Provides one of the standard ComfyUI schedulers'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "scheduler (Comfy's standard)"}),
            }
        }

    def get_names(self, scheduler):
        return (scheduler, scheduler)

class SchedulerToString:
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("scheduler_name",)
    OUTPUT_TOOLTIPS = ("scheduler name (STRING)",)
    FUNCTION = "get_name"

    CATEGORY = 'ImageSaver/utils'
    DESCRIPTION = 'Provides a given sandard ComfyUI or some extra scheduler\'s name as string'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS + ['AYS SDXL', 'AYS SD1', 'AYS SVD', 'GITS[coeff=1.2]'], {"tooltip": "scheduler (Comfy's standard + extras)"}),
            }
        }

    def get_name(self, scheduler):
        return (scheduler,)

class SchedulerComfyToString:
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("scheduler_name",)
    OUTPUT_TOOLTIPS = ("scheduler name (STRING)",)
    FUNCTION = "get_name"

    CATEGORY = 'ImageSaver/utils'
    DESCRIPTION = 'Provides a given sandard ComfyUI scheduler\'s name as string'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "scheduler (Comfy's standard)"}),
            }
        }

    def get_name(self, scheduler):
        return (scheduler,)

class SamplerToString:
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("sampler_name",)
    OUTPUT_TOOLTIPS = ("sampler name (STRING)",)
    FUNCTION = "get_name"

    CATEGORY = 'ImageSaver/utils'
    DESCRIPTION = 'Provides a given sandard ComfyUI samplers\'s name as string'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "sampler (Comfy's standard)"}),
            }
        }

    def get_name(self, sampler):
        return (sampler,)

class InputParameters:
    RETURN_TYPES = ("INT", "INT", "FLOAT", comfy.samplers.KSampler.SAMPLERS, "STRING", comfy.samplers.KSampler.SCHEDULERS, "STRING", "FLOAT")
    RETURN_NAMES = ("seed", "steps", "cfg", "sampler", "sampler_name", "scheduler", "scheduler_name", "denoise")
    OUTPUT_TOOLTIPS = (
        "seed (INT)",
        "steps (INT)",
        "cfg (FLOAT)",
        "sampler (SAMPLERS)",
        "sampler name (STRING)",
        "scheduler (SCHEDULERS)",
        "scheduler name (STRING)",
        "denoise (FLOAT)",
    )
    FUNCTION = "get_values"

    CATEGORY = "ImageSaver/utils"
    DESCRIPTION = "Combined node for seed, steps, cfg, sampler, scheduler and denoise."

    @classmethod
    def INPUT_TYPES(cls):
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

    def get_values(self, seed, steps, cfg, sampler, scheduler, denoise):
        return (seed, steps, cfg, sampler, sampler, scheduler, scheduler, denoise)
