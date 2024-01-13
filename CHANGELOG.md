# Untagged

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
