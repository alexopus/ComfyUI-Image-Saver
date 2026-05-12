from typing import Any
import torch
import comfy.samplers

from .nodes import ImageSaverSimple, ImageSaverMetadata

class MakeImageSaverSimpleConfig:
    """Standalone node for creating Saver Settings (filename, path, extension, etc.)."""
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        base = ImageSaverSimple.INPUT_TYPES()
        required = {}
        for k, v in base.get("required", {}).items():
            if k != "images":
                required[k] = v

        optional = {}
        for k, v in base.get("optional", {}).items():
            if k not in ["metadata", "show_preview"]:
                optional[k] = v

        return {
            "required": required,
            "optional": optional
        }

    RETURN_TYPES = ("SIMPLE_SAVER_CONFIG",)
    RETURN_NAMES = ("simple_saver_config",)
    FUNCTION = "make_config"
    CATEGORY = "ImageSaver/Pipe"
    DESCRIPTION = "Create a standalone Saver Configuration for use with Pipe nodes."

    def make_config(self, **kwargs) -> tuple[dict[str, Any]]:
        return (kwargs,)

class MakeImageSaverMetadataConfig:
    """Standalone node for creating Metadata Configuration."""
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        # Inherit all inputs from ImageSaverMetadata to preserve exact UI order
        base = ImageSaverMetadata.INPUT_TYPES()
        optional = {}

        for k, v in base.get("optional", {}).items():
            # Renaming and UI Upgrades:
            # 1. 'seed_value' -> 'seed': renamed to trigger ComfyUI's standard JS for "control_after_generate" (random/fixed).
            if k == "seed_value":
                optional["seed"] = v
            # 2. 'sampler_name': upgraded to COMBO to enable direct wiring to KSamplers or other nodes.
            elif k == "sampler_name":
                optional["sampler_name"] = (comfy.samplers.KSampler.SAMPLERS, v[1])
            # 3. 'scheduler_name': upgraded to COMBO to enable direct wiring. 
            # Note: We keep the '_name' suffix to remain consistent with the original Metadata engine's internal dictionary keys.
            elif k == "scheduler_name":
                optional["scheduler_name"] = (comfy.samplers.KSampler.SCHEDULERS, v[1])
            else:
                optional[k] = v

        return {
            "required": {},
            "optional": optional
        }

    RETURN_TYPES = ("METADATA_CONFIG",)
    RETURN_NAMES = ("metadata_config",)
    FUNCTION = "make_config"
    CATEGORY = "ImageSaver/Pipe"
    DESCRIPTION = "Create a standalone Metadata Configuration for use with Pipe nodes."

    def make_config(self, **kwargs) -> tuple[dict[str, Any]]:
        # We must map 'seed' back to 'seed_value' for the author's engine
        if "seed" in kwargs:
            kwargs["seed_value"] = kwargs.pop("seed")
        return (kwargs,)

class MakeImageSaverPipe:
    """Bundles metadata and Image Saver settings into a single Pipe connection."""
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "simple_saver_config": ("SIMPLE_SAVER_CONFIG", {"tooltip": "Required standalone Saver Configuration."}),
                "metadata_config": ("METADATA_CONFIG", {"tooltip": "Required standalone Metadata Configuration."}),
            },
        }

    RETURN_TYPES = ("IMAGESAVER_PIPE",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "make_pipe"
    CATEGORY = "ImageSaver/Pipe"
    DESCRIPTION = "Bundles metadata and Image Saver settings into a single Pipe connection."

    def make_pipe(self, simple_saver_config: dict[str, Any], metadata_config: dict[str, Any]) -> tuple[dict[str, Any]]:
        # 1. Create the Pipe (Bundled Configuration)
        # We store the raw configs for downstream editing (Lazy Evaluation)
        pipe = {
            "metadata_config": metadata_config,
            "simple_saver_config": simple_saver_config
        }

        return (pipe,)

class EditImageSaverPipe:
    """Safely override metadata or saver settings in an existing Image Saver Pipe."""
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        # 1. Dynamic Inheritance: We pull the source-of-truth from the config nodes
        # This logical chaining prevents redundant upgrades/renaming.
        base_meta = MakeImageSaverMetadataConfig.INPUT_TYPES()["optional"]

        # 2. Scope Definition: Define which fields we want to expose for scenario branching.
        exposed_keys = [
            "modelname", "positive", "negative", "width", "height", "seed",
            "steps", "cfg", "sampler_name", "scheduler_name", "denoise", "clip_skip", "additional_hashes", "custom"
        ]

        # 3. Scenario Identifiers: Filename and Path are placed at the top as primary branch markers.
        # We use '[original]' as a default to enable non-destructive string editing (append/prepend).
        optional = {
            "filename": ("STRING", {"default": "[original]", "tooltip": "String inputs support the [original] placeholder to append/prepend."}),
            "path":     ("STRING", {"default": "[original]", "tooltip": "String inputs support the [original] placeholder to append/prepend."}),
            "counter":  ("INT",    {"forceInput": True}),
        }

        # 4. Visual Synchronization: Iteratively add metadata fields to preserve the relative order.
        for k, v in base_meta.items():
            if k in exposed_keys:
                # UX Refinement: String fields remain text boxes (with [original] default).
                # All other types (INT, FLOAT, COMBO) use forceInput: True to prevent 
                # accidental overwrites by default widget values.
                if v[0] == "STRING":
                    optional[k] = (v[0], {**v[1], "default": "[original]"})
                else:
                    optional[k] = (v[0], {**v[1], "forceInput": True})

        return {
            "required": {
                "pipe": ("IMAGESAVER_PIPE",),
            },
            "optional": optional
        }

    RETURN_TYPES = ("IMAGESAVER_PIPE",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "edit_pipe"
    CATEGORY = "ImageSaver/Pipe"
    DESCRIPTION = "Safely override metadata or saver settings in an existing Image Saver Pipe. (Creates a new branch without modifying the original).\nString fields support the '[original]' placeholder to append/prepend."

    def edit_pipe(self, pipe: dict[str, Any], **kwargs) -> tuple[dict[str, Any]]:
        metadata_config = pipe["metadata_config"].copy()
        simple_saver_config = pipe["simple_saver_config"].copy()

        def apply_string_edit(new_val: Any, original_val: Any) -> Any:
            if new_val is not None:
                # Fast-fail if the user left the widget completely untouched
                if new_val == "[original]":
                    return original_val

                # If they modified it but kept the placeholder (e.g., "[original], masterpiece")
                if isinstance(new_val, str) and "[original]" in new_val:
                    return new_val.replace("[original]", str(original_val) if original_val is not None else "")

                # If they completely replaced the text
                return new_val
            return original_val

        # 1. Handle Saver Config Overrides
        for key in ["filename", "path"]:
            simple_saver_config[key] = apply_string_edit(kwargs.get(key), simple_saver_config.get(key))
            
        if kwargs.get("counter") is not None:
            simple_saver_config["counter"] = kwargs.get("counter")

        # 2. Handle Metadata Config Overrides
        for key in ["modelname", "positive", "negative", "sampler_name", "scheduler_name", "additional_hashes", "custom"]:
            metadata_config[key] = apply_string_edit(kwargs.get(key), metadata_config.get(key))

        for key in ["width", "height", "steps", "cfg", "denoise", "clip_skip"]:
            if kwargs.get(key) is not None:
                metadata_config[key] = kwargs.get(key)
                
        if kwargs.get("seed") is not None:
            metadata_config["seed_value"] = kwargs.get("seed")

        return ({"metadata_config": metadata_config, "simple_saver_config": simple_saver_config},)

class ReadImageSaverPipe:
    """
    Extracts commonly modified metadata and saver settings from an existing Image Saver Pipe.

    This node uses 'Architectural Reflection' to automatically mirror the structure 
    of the Edit node, ensuring the Extraction order always matches the Modification order.
    """
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "pipe": ("IMAGESAVER_PIPE",),
            },
        }

    # 1. Reflection: We dynamically pull the dictionary definition from the Edit node.
    # This guarantees that if a field is added/removed in Edit, the Read node 
    # self-heals its output sockets instantly.
    _edit_inputs = EditImageSaverPipe.INPUT_TYPES()["optional"]

    # 2. Dynamic Schema: Map the input wire types directly to output wire types.
    RETURN_TYPES = ("IMAGESAVER_PIPE",) + tuple(v[0] for v in _edit_inputs.values())
    RETURN_NAMES = ("pipe",) + tuple(_edit_inputs.keys())

    FUNCTION = "read_pipe"
    CATEGORY = "ImageSaver/Pipe"
    DESCRIPTION = "Extracts commonly modified metadata and saver settings from an existing Image Saver Pipe. (Lazy extraction: does not trigger metadata hashing)."

    def read_pipe(self, pipe: dict[str, Any]) -> tuple[Any, ...]:
        metadata_config = pipe["metadata_config"]
        simple_saver_config = pipe["simple_saver_config"]

        # 3. Dynamic Fallbacks: We pull the maintainer's base defaults to ensure 
        # that 'Read' operations are safe and predictable even with old/missing data.
        base_meta_defaults = ImageSaverMetadata.INPUT_TYPES()["optional"]
        base_saver_req = ImageSaverSimple.INPUT_TYPES()["required"]
        base_saver_opt = ImageSaverSimple.INPUT_TYPES()["optional"]

        res = [pipe]
        # 4. Iterative Unpacking: We loop over the names we dynamically generated above.
        # This removes the need for hardcoded list indexing and prevents off-by-one errors.
        for key in self.RETURN_NAMES[1:]: # Skip the first 'pipe' return
            if key in ["filename", "path"]:
                res.append(simple_saver_config.get(key, base_saver_req[key][1].get("default", "")))
            elif key == "counter":
                res.append(simple_saver_config.get(key, base_saver_opt[key][1].get("default", 0)))
            else:
                meta_key = "seed_value" if key == "seed" else key
                default_val = base_meta_defaults[meta_key][1].get("default")
                res.append(metadata_config.get(meta_key, default_val))

        return tuple(res)

class ImageSaverFromPipe:
    """
    Save images using settings and metadata unpacked from an Image Saver Pipe.
    """
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "pipe":                  ("IMAGESAVER_PIPE",),
                "images":                ("IMAGE",   {"tooltip": "image(s) to save"}),
            },
            "optional": {
                "show_preview":          ("BOOLEAN", {"default": True, "tooltip": "if True, displays saved images in the UI preview"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("IMAGESAVER_PIPE", "STRING", "STRING")
    RETURN_NAMES = ("pipe", "hashes", "a1111_params")
    OUTPUT_TOOLTIPS = ("The pass-through Image Saver Pipe", "Comma-separated list of the hashes to chain with other Image Saver additional_hashes","Written parameters to the image metadata")
    FUNCTION = "save_files"

    OUTPUT_NODE = True

    CATEGORY = "ImageSaver/Pipe"
    DESCRIPTION = "Save images using settings and metadata unpacked from an Image Saver Pipe."

    def save_files(self,
        images: list[torch.Tensor],
        pipe: dict[str, Any],
        show_preview: bool = True,
        prompt: dict[str, Any] | None = None,
        extra_pnginfo: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # 1. Resolve the Lazy Pipe (The "Ingredients")
        # Extract the raw configurations carried by the pipe.
        metadata_config = pipe["metadata_config"]
        simple_saver_config = pipe["simple_saver_config"]

        # 2. Call the author's EXISTING resolution engine (The "Cooking")
        # This performs hashing and Civitai API lookups using the original logic.
        metadata = ImageSaverMetadata.make_metadata(**metadata_config)

        # 3. CALL THE EXISTING SAVER NODE AS A FUNCTION (The "Engine")
        # By instantiating the base node and calling its save_images method, we 
        # ensure that 100% of the saving logic, path formatting, and UI preview 
        # code is reused without duplication.
        # **simple_saver_config: Unpacks the dictionary (filename, path, extension, etc.) 
        #                        directly into the method's keyword arguments.
        base_node = ImageSaverSimple()
        res = base_node.save_images(
            images=images,
            metadata=metadata,
            show_preview=show_preview,
            prompt=prompt,         # Required for workflow embedding
            extra_pnginfo=extra_pnginfo, # Required for workflow embedding
            **simple_saver_config  # Supplies filename, path, extension, webp toggles, etc.
        )
        # 4. Final Wrap
        # The base node returns (hashes, params), which we return directly along with the pipe pass-through.
        final_hashes, a111_params = res["result"]
        res["result"] = (pipe, final_hashes, a111_params)

        return res
