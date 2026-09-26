import os
import itertools
import json
import tempfile
import shutil
import pytest
from PIL import Image
import piexif
import piexif.helper
from .saver import save_image

def get_default_workflow():
    """Read the default workflow from the JSON file."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    default_workflow_path = os.path.join(current_dir, "default_workflow.json")
    with open(default_workflow_path, 'r') as f:
        return json.load(f)


def get_large_workflow(padding_size: int):
    """Create a large workflow by duplicating the default workflow until it's at least 500KB."""
    default_workflow = get_default_workflow()
    large_workflow = default_workflow.copy()
    large_workflow["padding"] = "x" * padding_size
    workflow_size = len(json.dumps(large_workflow)) / 1024  # Size in KB
    print(f"Large workflow size: {workflow_size:.2f} KB")
    return large_workflow


@pytest.fixture(
    params=list(itertools.product(
        ["simple", "default", "large", "huge"],  # workflow_type
        [True, False]                            # embed_workflow
    )),
    ids=lambda param: f"workflow-{param[0]}_embed-{param[1]}"
)
def setup_test_env(request):
    """Setup test environment with temp directory and test image, parameterized by workflow type."""
    temp_dir = tempfile.mkdtemp()

    test_image = Image.new('RGB', (100, 100), color='red')

    a111_params = """
beautiful scenery nature glass bottle landscape, purple galaxy bottle, low key
Negative prompt: (worst quality, low quality, bad quality:1.3), embedding:ng_deepnegative_v1_75t, embedding:EasyNegative, embedding:badhandv4
Steps: 30, Sampler: DPM++ 2M SDE, CFG scale: 7.0, Seed: 42, Size: 512x512, Model: , Version: ComfyUI,
Civitai resources: [
    {"modelName":"Deep Negative V1.x","modelVersionName":"V1 75T","modelVersionId":5637,"type":"textualinversion","weight":1.0,"air":"urn:air:sd1:embedding:civitai:4629@5637"},
    {"modelName":"EasyNegative","modelVersionName":"EasyNegative_pt","modelVersionId":9536,"type":"textualinversion","weight":1.0,"air":"urn:air:sd1:embedding:civitai:7808@9536"},
    {"modelName":"badhandv4","modelVersionName":"badhandv4","modelVersionId":20068,"type":"textualinversion","weight":1.0,"air":"urn:air:other:embedding:civitai:16993@20068"}]
"""

    prompt = {"prompt": "test prompt", "negative_prompt": "test negative prompt"}

    workflow_type, embed_workflow = request.param

    if workflow_type == "simple":
        extra_pnginfo = {"workflow": {"version": "1.0", "nodes": []}}
    elif workflow_type == "default":
        default_workflow = get_default_workflow()
        extra_pnginfo = {"workflow": default_workflow}
    elif workflow_type == "large":
        large_workflow = get_large_workflow(524288 )
        extra_pnginfo = {"workflow": large_workflow}
        # Check the size for debugging purposes
        workflow_size = len(json.dumps(large_workflow)) / 1024  # Size in KB
        print(f"Large workflow size: {workflow_size:.2f} KB")
    elif workflow_type == "huge":
        huge_workflow = get_large_workflow(2097152)
        extra_pnginfo = {"workflow": huge_workflow}
        # Check the size for debugging purposes
        workflow_size = len(json.dumps(huge_workflow)) / 1024  # Size in KB
        print(f"Large workflow size: {workflow_size:.2f} KB")

    yield temp_dir, test_image, a111_params, prompt, extra_pnginfo, workflow_type, embed_workflow

    shutil.rmtree(temp_dir)

@pytest.mark.parametrize(
    "optimize",
    [True, False],
    ids=["optimize", "no-optimize"]
)
def test_save_png(setup_test_env, optimize):
    """Test that complete metadata is correctly saved and can be retrieved for PNG format."""
    temp_dir, test_image, a111_params, prompt, extra_pnginfo, workflow_type, embed_workflow = setup_test_env
    image_path = os.path.join(temp_dir, f"test_with_workflow_{workflow_type}.png")
    save_image(test_image, image_path, "png", 100, True, optimize, a111_params, prompt, extra_pnginfo, embed_workflow)
    saved_image = Image.open(image_path)
    try:
        assert saved_image.info.get("parameters") == a111_params
        if embed_workflow:
            assert json.loads(saved_image.info.get("prompt")) == prompt
            assert json.loads(saved_image.info.get("workflow")) == extra_pnginfo["workflow"]
        else:
            assert set(saved_image.info.keys()) == {"parameters"}, "PNG should not contain prompt or workflow data"
    finally:
        saved_image.close()

def test_save_jpeg(setup_test_env):
    """Test that metadata is correctly saved and can be retrieved for JPEG format."""
    temp_dir, test_image, a111_params, prompt, extra_pnginfo, workflow_type, embed_workflow = setup_test_env
    jpeg_path = os.path.join(temp_dir, f"test_{workflow_type}.jpeg")
    save_image(test_image, jpeg_path, "jpeg", 90, False, False, a111_params, prompt, extra_pnginfo, embed_workflow)
    saved_image = Image.open(jpeg_path)
    try:
        exif_dict = piexif.load(saved_image.info["exif"])
        user_comment = piexif.helper.UserComment.load(exif_dict["Exif"][piexif.ExifIFD.UserComment])
        assert user_comment == a111_params

        if embed_workflow:
            if workflow_type == "simple" or workflow_type == "default":
                assert "0th" in exif_dict, "Expected workflow data in EXIF"
                # verify that prompt and workflow data are in EXIF
                expected_keys = {piexif.ImageIFD.Make, piexif.ImageIFD.Model}
                found_keys = set(exif_dict["0th"].keys()) & expected_keys
                assert len(found_keys) > 0, "Expected workflow or prompt data in EXIF"

                if piexif.ImageIFD.Make in exif_dict["0th"]:
                    make_data = exif_dict["0th"][piexif.ImageIFD.Make]
                    make_str = make_data.decode('utf-8')
                    # Check that workflow matches
                    if make_str.startswith("workflow:"):
                        make_str = make_str[len("workflow:"):]
                    saved_workflow = json.loads(make_str)
                    original_workflow = extra_pnginfo["workflow"]

                    assert saved_workflow == original_workflow, "Saved workflow content doesn't match original"

                if piexif.ImageIFD.Model in exif_dict["0th"]:
                    model_data = exif_dict["0th"][piexif.ImageIFD.Model]
                    model_str = model_data.decode('utf-8')
                    # Check that "prompt" matches
                    if model_str.startswith("prompt:"):
                        model_str = model_str[len("prompt:"):]
                    saved_prompt = json.loads(model_str)
                    assert saved_prompt == prompt, "Saved prompt content doesn't match original"
            else:
                # When workflow_type is "large", verify that the workflow is too large to embed
                if "0th" in exif_dict:
                    assert not any(k in exif_dict["0th"] for k in (piexif.ImageIFD.Make, piexif.ImageIFD.Model)), "JPEG should not contain prompt or workflow data"
        else:
            # When embed_workflow is False, verify no prompt or workflow in EXIF
            if "0th" in exif_dict:
                assert not any(k in exif_dict["0th"] for k in (piexif.ImageIFD.Make, piexif.ImageIFD.Model)), "JPEG should not contain prompt or workflow data"
    finally:
        saved_image.close()

@pytest.mark.parametrize(
    "lossless,quality",
    [(True, 100), (False, 90)],
    ids=["lossless-max", "lossy-90"]
)
def test_save_webp(setup_test_env, lossless, quality):
    """Test that metadata is correctly saved and can be retrieved for lossless WebP format."""
    temp_dir, test_image, a111_params, prompt, extra_pnginfo, workflow_type, embed_workflow = setup_test_env
    iamge_path = os.path.join(temp_dir, f"test_lossless_{workflow_type}.webp")
    save_image(test_image, iamge_path, "webp", quality, lossless, False, a111_params, prompt, extra_pnginfo, embed_workflow)
    saved_image = Image.open(iamge_path)
    try:
        # Verify a111_params is correctly stored in EXIF UserComment
        exif_dict = piexif.load(saved_image.info["exif"])
        user_comment = piexif.helper.UserComment.load(exif_dict["Exif"][piexif.ExifIFD.UserComment])
        assert user_comment == a111_params

        if embed_workflow:
            assert "0th" in exif_dict, "Expected workflow data in EXIF"
            # When embed_workflow is True, verify that prompt and workflow data are in EXIF
            expected_keys = {piexif.ImageIFD.Make, piexif.ImageIFD.Model}
            found_keys = set(exif_dict["0th"].keys()) & expected_keys
            assert len(found_keys) > 0, "Expected workflow or prompt data in EXIF"

            if piexif.ImageIFD.Make in exif_dict["0th"]:
                make_data = exif_dict["0th"][piexif.ImageIFD.Make]
                make_str = make_data.decode('utf-8')
                # Check that workflow matches
                if make_str.startswith("workflow:"):
                    make_str = make_str[len("workflow:"):]
                saved_workflow = json.loads(make_str)
                original_workflow = extra_pnginfo["workflow"]

                assert saved_workflow == original_workflow, "Saved workflow content doesn't match original"

            if piexif.ImageIFD.Model in exif_dict["0th"]:
                model_data = exif_dict["0th"][piexif.ImageIFD.Model]
                model_str = model_data.decode('utf-8')
                # Check that "prompt" matches
                if model_str.startswith("prompt:"):
                    model_str = model_str[len("prompt:"):]
                saved_prompt = json.loads(model_str)
                assert saved_prompt == prompt, "Saved prompt content doesn't match original"
        else:
            # When embed_workflow is False, verify no prompt or workflow in EXIF
            if "0th" in exif_dict:
                assert not any(k in exif_dict["0th"] for k in (piexif.ImageIFD.Make, piexif.ImageIFD.Model)), "WEBP should not contain prompt or workflow data"
    finally:
        saved_image.close()
