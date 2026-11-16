"""
Load Image with Metadata: conversion of file metadata to the savers' Metadata, and the
live-preview route's path checks. Image loading itself is ComfyUI's LoadImage (stubbed).
"""
import os

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import folder_paths
from image_saver_under_test.nodes_image_loader import LoadImageWithMetadata, metadata_response, split_sampler

PARAMETERS = ("a cat\nNegative prompt: dog\n"
              'Steps: 30, Sampler: DPM++ 2M SDE Karras, CFG scale: 6.5, Seed: 42, Size: 832x1216, Clip skip: 2, '
              'Model hash: ABCDEF1234, Model: pony, Hashes: {"LORA:style":"0123456789","model":"ABCDEF1234"}, Version: ComfyUI')


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    input_dir, output_dir, temp_dir = (tmp_path / name for name in ("input", "output", "temp"))
    for d in (input_dir, output_dir, temp_dir):
        d.mkdir()

    def annotated(name, default_dir=None):
        for suffix, base in ((" [output]", output_dir), (" [input]", input_dir), (" [temp]", temp_dir)):
            if name.endswith(suffix):
                return os.path.join(base, name[:-len(suffix)])
        return os.path.join(input_dir, name)

    monkeypatch.setattr(folder_paths, "get_input_directory", lambda: str(input_dir), raising=False)
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(output_dir), raising=False)
    monkeypatch.setattr(folder_paths, "get_temp_directory", lambda: str(temp_dir), raising=False)
    monkeypatch.setattr(folder_paths, "get_annotated_filepath", annotated, raising=False)
    return input_dir


def png(path, parameters=None):
    info = PngInfo()
    if parameters:
        info.add_text("parameters", parameters)
    Image.new("RGB", (16, 8)).save(path, pnginfo=info)


@pytest.mark.parametrize("sampler,scheduler,expected", [
    ("DPM++ 2M SDE Karras", "", ("dpmpp_2m_sde", "karras")),
    ("Euler a", "", ("euler_ancestral", "normal")),
    ("DPM++ 2M", "Karras", ("dpmpp_2m", "karras")),  # newer A1111 writes "Schedule type" separately
    ("res_multistep_beta", "", ("res_multistep", "beta")),  # get_civitai_sampler_name for unmapped samplers
    ("euler", "simple", ("euler", "simple")),
])
def test_split_sampler(sampler, scheduler, expected):
    assert split_sampler(sampler, scheduler) == expected


def test_node_outputs_image_and_saver_metadata(dirs):
    png(dirs / "gen.png", PARAMETERS)
    image, mask, metadata, positive, negative = LoadImageWithMetadata().load_image("gen.png")

    assert (image, mask) == ("IMAGE:gen.png", "MASK:gen.png")
    assert (positive, negative) == ("a cat", "dog")
    assert metadata.a111_params == PARAMETERS  # re-saved unchanged by Image Saver Simple
    assert (metadata.width, metadata.height, metadata.seed, metadata.steps, metadata.cfg) == (832, 1216, 42, 30, 6.5)
    assert (metadata.sampler_name, metadata.scheduler_name, metadata.clip_skip) == ("dpmpp_2m_sde", "karras", 2)
    assert metadata.modelname == "pony"
    assert metadata.final_hashes == "pony:ABCDEF1234,style:0123456789"


def test_node_without_metadata_still_loads_image(dirs):
    png(dirs / "plain.png")
    image, _, metadata, positive, negative = LoadImageWithMetadata().load_image("plain.png")
    assert image == "IMAGE:plain.png"
    assert (metadata, positive, negative) == (None, "", "")


def test_node_reuses_load_image_inputs_and_checks():
    assert LoadImageWithMetadata.INPUT_TYPES()["required"]["image"][1] == {"image_upload": True}
    assert LoadImageWithMetadata.IS_CHANGED("x.png") == "changed:x.png"
    assert not hasattr(LoadImageWithMetadata, "ESSENTIALS_CATEGORY")


def test_route_returns_rows_and_parameters(dirs):
    png(dirs / "gen.png", PARAMETERS)
    status, body = metadata_response("gen.png")
    assert status == 200
    assert body["source"] == "parameters"
    assert body["parameters"] == PARAMETERS
    assert ["Seed", "42"] in [list(row) for row in body["rows"]]
    assert body["warning"] == ""


def test_route_warns_about_mask_editor_copies(dirs):
    (dirs / "clipspace").mkdir()
    png(dirs / "clipspace" / "clipspace-painted-masked-123.png")
    status, body = metadata_response("clipspace/clipspace-painted-masked-123.png [input]")
    assert status == 200
    assert body["source"] == "none"
    assert "mask editor" in body["warning"]


@pytest.mark.parametrize("filename", ["../secret.png", "/etc/passwd", "../../outside.png [output]"])
def test_route_rejects_paths_outside_comfy_dirs(dirs, filename):
    png(dirs.parent / "secret.png", PARAMETERS)
    status, _ = metadata_response(filename)
    assert status == 403


def test_route_rejects_symlink_escape(dirs):
    png(dirs.parent / "secret.png", PARAMETERS)
    (dirs / "link.png").symlink_to(dirs.parent / "secret.png")
    assert metadata_response("link.png")[0] == 403


def test_route_missing_file(dirs):
    assert metadata_response("nope.png")[0] == 404
    assert metadata_response("")[0] == 400
