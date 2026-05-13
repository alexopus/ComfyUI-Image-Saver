[!] Forked from https://github.com/giriss/comfy-image-saver, which seems to be inactive since a while.

# Save image with generation metadata in ComfyUI

Allows you to save images with their **generation metadata**. Includes the metadata compatible with *Civitai* geninfo auto-detection. Works with PNG, JPG and WEBP. For PNG stores both the full workflow in comfy format, plus a1111-style parameters. For JPEG/WEBP only the a1111-style parameters are stored. **Includes hashes of Models, LoRAs and embeddings for proper resource linking** on civitai.

You can find the example workflow file named `example-workflow.json`.
<img width="1288" height="1039" alt="workflow" src="https://github.com/user-attachments/assets/dbbb9f67-afa3-48a2-8cd3-e4116393f8e0" />

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
- `%time_format<format>` *– custom datetime format using Python strftime codes*
- `%model` *– full name of model file*
- `%basemodelname` *– name of model (without file extension)*
- `%seed`
- `%counter` *(supports zero-padded formatting, e.g., `%counter<03>` for `001`, `002`)*
- `%sampler_name`
- `%scheduler`
- `%steps`
- `%cfg`
- `%denoise`

Example:

| `filename` value | Result file name |
| --- | --- |
| `%time-%basemodelname-%cfg-%steps-%sampler_name-%scheduler-%seed` | `2023-11-16-131331-Anything-v4.5-pruned-mergedVae-7.0-25-dpm_2-normal-1_01.png` |
| `%time_format<%Y%m%d_%H%M%S>-%seed` | `20231116_131331-1.png` |
| `%time_format<%B %d, %Y> %basemodelname` | `November 16, 2023 Anything-v4.5.png` |
| `img_%time_format<%Y-%m-%d>_%seed` | `img_2023-11-16_1.png` |

**Common strftime format codes for `%time_format<format>`:**

| Code | Meaning | Example |
|------|---------|---------|
| `%Y` | Year (4-digit) | 2023 |
| `%y` | Year (2-digit) | 23 |
| `%m` | Month (01-12) | 11 |
| `%B` | Month name (full) | November |
| `%b` | Month name (short) | Nov |
| `%d` | Day (01-31) | 16 |
| `%H` | Hour 24h | 13 |
| `%I` | Hour 12h | 01 |
| `%M` | Minute | 13 |
| `%S` | Second | 31 |
| `%p` | AM/PM | PM |
| `%A` | Weekday (full) | Thursday |
| `%a` | Weekday (short) | Thu |
| `%F` | YYYY-MM-DD | 2023-11-16 |
| `%T` | HH:MM:SS | 13:13:31 |

## New Pipe Architecture

To solve the issue of cluttered wires in complex workflows with many image generations, a new **Pipe** architecture has been introduced. 

The pipe system bundles all metadata and image saver settings into a single connection, allowing you to pass them cleanly across your workflow.

![Image Saver Pipe Workflow](images/image_saver_pipe.png)

### Key Features
* **Single Wire:** Carries all information in just one pipe, drastically reducing wire clutter.
* **Chainable:** All pipe nodes are fully chainable.
* **Lazy Evaluation:** Metadata compilation (e.g. hashing) only occurs when the final image is actually saved, saving execution time.
* **Non-destructive Editing:** The `Edit Image Saver Pipe` node allows you to branch and modify settings. All string fields support the `[original]` placeholder to easily append or prepend text to existing values. *For more complicated operations, you can extract values using the `Read Image Saver Pipe` node, manipulate them with string nodes, and feed them back into an `Edit Image Saver Pipe`.*
* **Combo Types for Wiring:** `sampler` and `scheduler` inputs use combo types, making them easier to wire to other nodes.

### ⚠️ Important Note on Mixing Legacy and Pipe Nodes

While the new pipe architecture uses the exact same underlying logic as the classic nodes, **they are not directly compatible with each other** (you cannot wire a legacy node into a pipe node).

Additionally, **it is highly recommended NOT to mix legacy nodes and pipe nodes in the same workflow.** 
Legacy nodes evaluate and trigger side-effects (like `download_civitai` or `easy_remix`) immediately, whereas pipe nodes defer these actions until the final `Image Saver (From Pipe)` node runs. Mixing them can lead to unpredictable metadata generation and side-effect interactions.

