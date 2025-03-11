from .nodes import ImageSaver
from .nodes_literals import SeedGenerator, StringLiteral, SizeLiteral, IntLiteral, FloatLiteral, CfgLiteral
from .nodes_loaders import CheckpointLoaderWithName, UNETLoaderWithName
from .nodes_selectors import SamplerSelector, SchedulerSelector, SchedulerSelectorComfy, SchedulerToString, SamplerToString, SchedulerComfyToString
from .civitai_nodes import CivitaiHashFetcher

NODE_CLASS_MAPPINGS = {
    "Checkpoint Loader with Name (Image Saver)": CheckpointLoaderWithName,
    "UNet loader with Name (Image Saver)": UNETLoaderWithName,
    "Image Saver": ImageSaver,
    "Sampler Selector (Image Saver)": SamplerSelector,
    "Scheduler Selector (Image Saver)": SchedulerSelector,
    "Scheduler Selector (Comfy) (Image Saver)": SchedulerSelectorComfy,
    "Seed Generator (Image Saver)": SeedGenerator,
    "String Literal (Image Saver)": StringLiteral,
    "Width/Height Literal (Image Saver)": SizeLiteral,
    "Cfg Literal (Image Saver)": CfgLiteral,
    "Int Literal (Image Saver)": IntLiteral,
    "Float Literal (Image Saver)": FloatLiteral,
    "SchedulerToString (Image Saver)": SchedulerToString,
    "SchedulerComfyToString (Image Saver)": SchedulerComfyToString,
    "SamplerToString (Image Saver)": SamplerToString,
    "Civitai Hash Fetcher (Image Saver)": CivitaiHashFetcher,
}

__all__ = ['NODE_CLASS_MAPPINGS']
