# v1.5.2

- Reverted experimental webp support for the moment. Needs more testing.
- Fix putting "prompt" into JPEGs.

# v1.5.1

- Fix workflow storage in lossless webp

# v1.5.0

- New lines are no longer removed from prompts.
- Added Civitai Hash Fetcher node that can retrieve a ressource hash from civitai based on its name.
- Added an "aditional hashes" input that accepts a comma separated list of resource hahes that will be stored in the image metadata.
- Experimental support for storing workflow in webp.

# v1.4.0

- Add UNETLoaderWithName
- Also check the unet directory (if not found in checkpoints) when calculating model hash
- Add tooltips
- Image Saver: Add clip skip parameter
- Adds the suffix _0x to the file name if a file with that name already exists (#40)
- Remove strip_a1111_params option
- Bugfix: Fixing the outputs names of SchedulerToString, SchedulerComfyToString and SamplerToString nodes

# v1.3.0

- Saver node: converted sampler input to string
- SamplerSelector node: output sampler name also as a string
- Add SamplerToString util node
- Fixed converter nodes
- Change min value for widgets with fixed steps

# v1.2.1

- Update Impact Pack scheduler list

# v1.2.0

- Add option to strip positive/negative prompt from the a1111 parameters comment (hashes for loras/embeddings are still always added)
- Add option for embedding prompt/workflow in PNG
- Add 'AYS SDXL', 'AYS SD1' and 'AYS SVD' to scheduler selectors
- added dpmpp_3m_sde sampler
- added exponential scheduler
- Fix suffix for batches
- Save json for each image in batch
- Allow to leave modelname empty

# v1.1.0

-  Fix extension check in full_lora_path_for
-  add 'save_workflow_as_json', which allows saving an additional file with the json workflow included

# v1.0.0

- **BREAKING CHANGE**: Convert CheckpointSelector to CheckpointLoaderWithName (571fcfa319438a32e051f90b32827363bccbd2ef). Fixes 2 issues:
    - oversized search fields (https://github.com/giriss/comfy-image-saver/issues/5)
    - selector breaking when model files are added/removed at runtime
- Try to find loras with incomplete paths (002471d95078d8b2858afc92bc4589c8c4e8d459):
    - `<lora:asdf:1.2>` will be found and hashed if the actual location is `<lora:subdirectory/asdf:1.2>`
- Update default filename pattern from `%time_%seed` to `%time_%basemodelname_%seed` (72f17f0a4e97a7c402806cc21e9f564a5209073d)
- Include embedding, lora and model information in the metadata in civitai format (https://github.com/alexopus/ComfyUI-Image-Saver/pull/2)
- Rename all nodes to avoid conflicts with the forked repo
- Make PNG optimization optional and off by default (c760e50b62701af3d44edfb69d3776965a645406)
- Calculate model hash only if there is no calculated one on disk already. Store on disk after calculation (96df2c9c74c089a8cca811ccf7aaa72f68faf9db)
- Fix civitai sampler/scheduler name (af4eec9bc1cc55643c0df14aaf3a446fbbc3d86d)
- Fix metadata format according to https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/5ef669de080814067961f28357256e8fe27544f4/modules/processing.py#L673 (https://github.com/giriss/comfy-image-saver/pull/11)
- Add input `denoise` (https://github.com/Danand/comfy-image-saver/commit/37fc8903e05c0d70a7b7cfb3a4bcc51f4f464637)
- Add resolving of more placeholders for file names (https://github.com/giriss/comfy-image-saver/pull/16)
    - `%sampler_name`
    - `%steps`
    - `%cfg`
    - `%scheduler`
    - `%basemodelname`


Changes since the fork from https://github.com/giriss/comfy-image-saver.
