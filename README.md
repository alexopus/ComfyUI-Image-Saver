[!] Forked from https://github.com/giriss/comfy-image-saver, which seems to be inactive since a while.

##Updates
2025/01/06 
- I Removed new line trim function on the positive prompt, since I like to be able to manipulate how the prompt looks and store the loras name in a more visually clean way.
- I tried implementing WebP workflow embedding so you can store the ComfyUI json in the metadata while also storing generation parameter. This is working on Civitai. But when reading with "SD Prompt Reader" the information are not showing up. But since Civitai is, I think it's good.
- I implemented a new field "manual hash"! Here you can get Civitai AutoV3 hash and input here, Civitai will recognize your images with that model in cross posting! It works with workflows as well. It probably works with anything that has a AutoV3 hash.
- I wanted to implement a way to Cross post the Workflow itself when saving the image. The biggest challenge here was that since Civitai will actually read the zipped file hash, it's a circular thing to try to store the hash on the json and the re-zip because it would change the hash. So I made a NEW node called "Civitai Hash Fetcher" where you can input your 'username' and your 'model name', the first hit from the API (the most current version) hash will be fetched. Now you connect that to the "manual hash" and now you can share your workflow and if the user itself don't change this, the image will be cross-posted to our workflow!


# Save image with generation metadata in ComfyUI

Allows you to save images with their **generation metadata**. Includes the metadata compatible with *Civitai* geninfo auto-detection. Works with PNG, JPG and WEBP. For PNG stores both the full workflow in comfy format, plus a1111-style parameters. For JPEG/WEBP only the a1111-style parameters are stored. **Includes hashes of Models, LoRAs and embeddings for proper resource linking** on civitai.

You can find the example workflow file named `example-workflow.json`.
![workflow(1)](https://github.com/user-attachments/assets/83c37f5b-671e-4444-8ff5-66147a842fec)

You can also add LoRAs to the prompt in \<lora:name:weight\> format, which would be translated into hashes and stored together with the metadata. For this it is recommended to use `ImpactWildcardEncode` from the fantastic [ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack). It will allow you to convert the LoRAs directly to proper conditioning without having to worry about avoiding/concatenating lora strings, which have no effect in standard conditioning nodes. Here is an example:
![workflow](https://github.com/user-attachments/assets/61440fac-f1d5-414b-ae69-dbdda9d6d442)

This would have civitai autodetect all of the resources (assuming the model/lora/embedding hashes match):
![image](https://github.com/alexopus/ComfyUI-Image-Saver/assets/25933468/f0642389-4f34-4a64-89a6-5cf9c33d5ed1)

## How to install?

### Method 1: Manager (Recommended)
If you have *ComfyUI-Manager*, you can simply search "**ComfyUI Image Saver**" and install these custom nodes.

### Method 2: Easy
If you don't have *ComfyUI-Manager*, then:
- Using CLI, go to the ComfyUI folder
- `cd custom_nodes`
- `git clone git@github.com:alexopus/ComfyUI-Image-Saver.git`
- `cd ComfyUI-Image-Saver`
- `pip install -r requirements.txt`
- Start/restart ComfyUI

## Customization of file/folder names

You can use following placeholders:

- `%date`
- `%time` *– format taken from `time_format`*
- `%model` *– full name of model file*
- `%basemodelname` *– name of model (without file extension)*
- `%seed`
- `%counter`
- `%sampler_name`
- `%scheduler`
- `%steps`
- `%cfg`
- `%denoise`

Example:

| `filename` value | Result file name |
| --- | --- |
| `%time-%basemodelname-%cfg-%steps-%sampler_name-%scheduler-%seed` | `2023-11-16-131331-Anything-v4.5-pruned-mergedVae-7.0-25-dpm_2-normal-1_01.png` |
