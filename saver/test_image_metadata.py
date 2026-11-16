import json
import os

import piexif
import piexif.helper
import pytest
from PIL import Image, features
from PIL.PngImagePlugin import PngInfo

from .image_metadata import display_rows, edit_parameters, parse_parameters, parse_settings, read_image_metadata
from .saver import save_image

PARAMETERS = """beautiful scenery, purple galaxy bottle,
(masterpiece:1.2), <lora:style:0.8>
Negative prompt: worst quality, low quality
embedding:EasyNegative
Steps: 30, Sampler: DPM++ 2M SDE Karras, CFG scale: 7.0, Seed: 42, Size: 832x1216, Clip skip: 2, Model hash: ABCDEF1234, Model: pony, Hashes: {"LORA:style":"0123456789","model":"ABCDEF1234"}, Version: ComfyUI, Civitai resources: [{"modelName":"Pony Diffusion V6 XL","modelVersionName":"V6","modelVersionId":290640,"type":"checkpoint"},{"modelName":"Some, Style","modelVersionName":"v2","modelVersionId":333590,"type":"lora","weight":0.8}]"""

PROMPT = {"3": {"class_type": "KSampler", "inputs": {"seed": 7, "steps": 25, "cfg": 5.5, "sampler_name": "euler", "scheduler": "beta", "denoise": 1.0, "positive": ["6", 0], "negative": ["7", 0], "model": ["4", 0]}}}
WORKFLOW = {"nodes": [], "links": []}

requires_avif = pytest.mark.skipif(not features.check("avif"), reason="Pillow built without AVIF support")


def test_parse_parameters_splits_prompts_and_settings():
    positive, negative, settings = parse_parameters(PARAMETERS)
    assert positive == "beautiful scenery, purple galaxy bottle,\n(masterpiece:1.2), <lora:style:0.8>"
    assert negative == "worst quality, low quality\nembedding:EasyNegative"
    assert settings["Steps"] == "30"
    assert settings["Sampler"] == "DPM++ 2M SDE Karras"
    assert settings["Size"] == "832x1216"
    assert settings["Model"] == "pony"
    assert settings["Hashes"] == {"LORA:style": "0123456789", "model": "ABCDEF1234"}
    assert settings["Version"] == "ComfyUI"
    assert [r["modelVersionId"] for r in settings["Civitai resources"]] == [290640, 333590]


def test_parse_parameters_without_negative_or_positive():
    assert parse_parameters("a cat\nSteps: 20, Seed: 1")[:2] == ("a cat", "")
    assert parse_parameters("\nNegative prompt: dog\nSteps: 20, Seed: 1")[:2] == ("", "dog")


def test_parse_settings_handles_quoted_values_and_commas():
    settings = parse_settings('Steps: 20, Lora hashes: "a: 123, b: 456", Sampler: Euler a, Version: v1.10.1')
    assert settings == {"Steps": "20", "Lora hashes": "a: 123, b: 456", "Sampler": "Euler a", "Version": "v1.10.1"}


@pytest.mark.parametrize("extension", ["png", "jpeg", "webp", pytest.param("avif", marks=requires_avif)])
def test_round_trip_with_image_saver(tmp_path, extension):
    path = os.path.join(tmp_path, f"img.{extension}")
    save_image(Image.new("RGB", (64, 48), "red"), path, extension, 90, False, False, PARAMETERS, PROMPT, {"workflow": WORKFLOW}, True)

    metadata = read_image_metadata(path)
    assert metadata.source == "parameters"
    assert metadata.parameters == PARAMETERS
    assert metadata.has_workflow
    assert metadata.settings["Seed"] == "42"
    assert (metadata.width, metadata.height) == (64, 48)


def test_a1111_jpeg_user_comment(tmp_path):
    path = os.path.join(tmp_path, "a1111.jpg")
    exif = piexif.dump({"Exif": {piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(PARAMETERS, encoding="unicode")}})
    Image.new("RGB", (8, 8)).save(path, exif=exif)
    metadata = read_image_metadata(path)
    assert metadata.parameters == PARAMETERS
    assert not metadata.has_workflow


def test_comfyui_native_png_falls_back_to_graph(tmp_path):
    path = os.path.join(tmp_path, "comfy.png")
    prompt = PROMPT | {
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sdxl/juggernaut.safetensors"}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": ["10", 0]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry"}},
        "10": {"class_type": "String Literal", "inputs": {"string": "a lighthouse"}},
        "12": {"class_type": "KSampler", "inputs": {"seed": 99, "steps": 10, "cfg": 4, "denoise": 0.4, "positive": ["6", 0], "negative": ["7", 0]}},
    }
    info = PngInfo()
    info.add_text("prompt", json.dumps(prompt))
    info.add_text("workflow", json.dumps(WORKFLOW))
    Image.new("RGB", (8, 8)).save(path, pnginfo=info)

    metadata = read_image_metadata(path)
    assert metadata.source == "workflow"
    assert (metadata.positive, metadata.negative) == ("a lighthouse", "blurry")
    # The full-denoise sampler wins over the later 0.4 refine pass
    assert metadata.settings == {"Steps": 25, "Sampler": "euler", "Scheduler": "beta", "CFG scale": 5.5, "Seed": 7,
                                 "Denoising strength": 1.0, "Model": "juggernaut"}


def test_image_without_metadata(tmp_path):
    path = os.path.join(tmp_path, "plain.png")
    Image.new("RGB", (8, 8)).save(path)
    metadata = read_image_metadata(path)
    assert metadata.source == "none"
    assert (metadata.positive, metadata.settings) == ("", {})


def test_display_rows_format_resources_and_hashes():
    from .image_metadata import ImageMetadata
    positive, negative, settings = parse_parameters(PARAMETERS)
    rows = dict(display_rows(ImageMetadata(tags={"parameters": PARAMETERS}, source="parameters", positive=positive,
                                           negative=negative, settings=settings, width=8, height=8)))
    assert rows["Civitai resources"] == "Pony Diffusion V6 XL V6 (type checkpoint)\nSome, Style v2 (type lora, weight 0.8)"
    assert rows["Hashes"] == "LORA:style: 0123456789\nmodel: ABCDEF1234"
    assert rows["Workflow embedded"] == "no"


FOREIGN = ('a cat\nNegative prompt: dog\nSteps: 20, Sampler: Euler a, Schedule type: Karras, CFG scale: 7, Seed: 1, '
           'Lora hashes: "style: 0123456789, other: abcdef", Hashes: {"model": "abc", "LORA:x": "def"}, Version: v1.10.1')


def test_edit_parameters_without_changes_returns_input():
    assert edit_parameters(FOREIGN) is FOREIGN


def test_edit_parameters_keeps_untouched_values_verbatim():
    edited = edit_parameters(FOREIGN, settings={"Seed": "2"})
    assert edited == FOREIGN.replace("Seed: 1", "Seed: 2")


def test_edit_parameters_prompts_remove_and_insert():
    edited = edit_parameters(FOREIGN, positive="a dog", negative="", settings={"Size": "64x64", "Schedule type": None},
                             remove=["lora hashes"])
    assert edited == ('a dog\nSteps: 20, Sampler: Euler a, CFG scale: 7, Seed: 1, '
                      'Hashes: {"model": "abc", "LORA:x": "def"}, Size: 64x64, Version: v1.10.1')


def test_edit_parameters_on_text_without_settings():
    assert edit_parameters("just a prompt", settings={"Seed": "3"}) == "just a prompt\nSeed: 3"


SAVED = ('a cat\nNegative prompt: embedding:EasyNegative, embedding:badhandv4\nSteps: 20, Sampler: Euler, Seed: 1, '
         'Model hash: 4ecddc97d4, Model: helloart, '
         'Hashes: {"embed:EasyNegative":"66a7279a88","embed:badhandv4":"5e40d722fc","model":"4ecddc97d4"}, Version: ComfyUI, '
         'Civitai resources: [{"modelName":"helloArt, Oil","modelVersionId":257926,"type":"checkpoint"},'
         '{"modelName":"EasyNegative","modelVersionName":"EasyNegative_pt","modelVersionId":9536,"type":"textualinversion","weight":1.0},'
         '{"modelName":"Deep Negative V1.x","modelVersionId":5637,"type":"textualinversion","weight":1.0}]')


def test_remove_resources_by_name_hash_and_version_id():
    edited = edit_parameters(SAVED, remove_resources=["easynegative", "5e40d722fc", "5637"])
    assert ('Hashes: {"model":"4ecddc97d4"}, Version: ComfyUI, '
            'Civitai resources: [{"modelName":"helloArt, Oil","modelVersionId":257926,"type":"checkpoint"}]') in edited
    assert "Model hash: 4ecddc97d4" in edited


def test_remove_resources_drops_emptied_settings_and_keeps_spaced_style():
    assert "Hashes" not in edit_parameters(SAVED, remove_resources=["EasyNegative", "badhandv4", "4ecddc97d4"])
    assert edit_parameters(FOREIGN, remove_resources=["x"]) == FOREIGN.replace(', "LORA:x": "def"', "")
    assert edit_parameters(FOREIGN, remove_resources=["style"]) == FOREIGN.replace('style: 0123456789, ', "")


def test_remove_resources_without_matches_changes_nothing():
    assert edit_parameters(SAVED, remove_resources=["nothing"]) == SAVED
