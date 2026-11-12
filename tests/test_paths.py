"""
Regression tests for the output-path/filename containment fix in nodes.py
(ImageSaver.save_images) and its underlying primitive, utils.resolve_within_output.

See conftest.py for how nodes.py/utils.py get imported without a full ComfyUI
install.
"""
import os

import numpy as np
import pytest

import folder_paths
from image_saver_under_test.nodes import ImageSaver, Metadata
from image_saver_under_test.utils import resolve_within_output


@pytest.fixture(autouse=True)
def output_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(folder_paths, "output_directory", str(tmp_path))
    return tmp_path


class FakeTensor:
    """Duck-types the .cpu().numpy() calls ImageSaver.save_images makes on IMAGE tensors."""

    def __init__(self, array):
        self._array = array

    def cpu(self):
        return self

    def numpy(self):
        return self._array


def make_image(size=8):
    return FakeTensor(np.random.rand(size, size, 3))


def make_metadata(**overrides):
    defaults = dict(
        modelname="", positive="", negative="", width=8, height=8, seed=0,
        steps=1, cfg=1.0, sampler_name="euler", scheduler_name="normal",
        denoise=1.0, clip_skip=0, custom="", additional_hashes="",
        ckpt_path="", a111_params="test params", final_hashes="",
    )
    defaults.update(overrides)
    return Metadata(**defaults)


def call_save_images(path, filename="test", extension="png", save_workflow_as_json=False):
    return ImageSaver.save_images(
        images=[make_image()],
        filename_pattern=filename,
        extension=extension,
        path=path,
        quality_jpeg_or_webp=100,
        lossless_webp=True,
        optimize_png=False,
        prompt=None,
        extra_pnginfo={"workflow": {"nodes": []}},
        save_workflow_as_json=save_workflow_as_json,
        embed_workflow=False,
        counter=0,
        time_format="%Y-%m-%d-%H%M%S",
        metadata=make_metadata(),
    )


@pytest.mark.parametrize("path", ["", "portraits", "portraits/2026-08-27", "a/b/c"])
def test_accepts_nested_paths_within_output_dir(path, output_dir):
    call_save_images(path)
    written = list(output_dir.rglob("*.png"))
    assert len(written) == 1
    assert output_dir in written[0].parents


@pytest.mark.parametrize("path", ["../escape", "../../escape", "/etc/escape"])
def test_rejects_traversal_via_path(path, output_dir):
    with pytest.raises(ValueError):
        call_save_images(path)
    assert list(output_dir.rglob("*.png")) == []


def test_rejects_traversal_via_filename(output_dir):
    with pytest.raises(ValueError):
        call_save_images(path="safe", filename="../../../escape")
    assert list(output_dir.rglob("*.png")) == []


def test_folder_in_filename_is_created(output_dir):
    assert call_save_images(path="safe", filename="sub/img") == [os.path.join("sub", "img.png")]
    assert (output_dir / "safe" / "sub" / "img.png").is_file()


def test_folder_in_filename_does_not_overwrite(output_dir):
    call_save_images(path="", filename="sub/img")
    second = call_save_images(path="", filename="sub/img")
    assert second == [os.path.join("sub", "img_01.png")]
    assert sorted(p.name for p in (output_dir / "sub").iterdir()) == ["img.png", "img_01.png"]


def test_rejects_traversal_via_json_sidecar(output_dir):
    with pytest.raises(ValueError):
        call_save_images(path="safe", filename="../../../escape", save_workflow_as_json=True)
    assert list(output_dir.rglob("*.json")) == []


def test_symlink_escape_is_rejected(output_dir):
    outside = output_dir.parent / "outside"
    outside.mkdir()
    (output_dir / "linked").symlink_to(outside)
    with pytest.raises(ValueError):
        call_save_images(path="linked")
    assert list(outside.iterdir()) == []


def test_resolve_within_output_accepts_nested_relative(output_dir):
    resolved = resolve_within_output("a/b")
    assert resolved == str((output_dir / "a" / "b").resolve())


@pytest.mark.parametrize("bad", ["../escape", "/etc/escape"])
def test_resolve_within_output_rejects_escape(bad, output_dir):
    with pytest.raises(ValueError):
        resolve_within_output(bad)
