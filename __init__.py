from typing import Any

from .nodes import ImageSaver, ImageSaverSimple, ImageSaverMetadata
from .nodes_literals import SeedGenerator, StringLiteral, SizeLiteral, IntLiteral, FloatLiteral, CfgLiteral
from .nodes_loaders import CheckpointLoaderWithName, UNETLoaderWithName
from .nodes_selectors import SamplerSelector, SchedulerSelector, SchedulerSelectorInspire, SchedulerToString, SamplerToString, SchedulerInspireToString, InputParameters
from .civitai_nodes import CivitaiHashFetcher

NODE_CLASS_MAPPINGS: dict[str, Any] = {
    "Checkpoint Loader with Name (Image Saver)": CheckpointLoaderWithName,
    "UNet loader with Name (Image Saver)": UNETLoaderWithName,
    "Image Saver": ImageSaver,
    "Image Saver Simple": ImageSaverSimple,
    "Image Saver Metadata": ImageSaverMetadata,
    "Sampler Selector (Image Saver)": SamplerSelector,
    "Scheduler Selector (Image Saver)": SchedulerSelector,
    "Scheduler Selector (inspire) (Image Saver)": SchedulerSelectorInspire,
    "Scheduler Selector (Image Saver)": SchedulerSelector,
    "Input Parameters (Image Saver)": InputParameters,
    "Seed Generator (Image Saver)": SeedGenerator,
    "String Literal (Image Saver)": StringLiteral,
    "Width/Height Literal (Image Saver)": SizeLiteral,
    "Cfg Literal (Image Saver)": CfgLiteral,
    "Int Literal (Image Saver)": IntLiteral,
    "Float Literal (Image Saver)": FloatLiteral,
    "SchedulerToString (Image Saver)": SchedulerToString,
    "SchedulerInspireToString (Image Saver)": SchedulerInspireToString,
    "SamplerToString (Image Saver)": SamplerToString,
    "Civitai Hash Fetcher (Image Saver)": CivitaiHashFetcher,
}

WEB_DIRECTORY = "js"

__all__ = ['NODE_CLASS_MAPPINGS', 'WEB_DIRECTORY']
