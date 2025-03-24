import os
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


def get_large_workflow():
    """Create a large workflow by duplicating the default workflow until it's at least 500KB."""
    default_workflow = get_default_workflow()
    large_workflow = default_workflow.copy()

    # If large_workflow doesn't have a nodes list, add it
    if "nodes" not in large_workflow:
        large_workflow["nodes"] = []

    # Keep duplicating nodes until we reach 500KB
    original_nodes = default_workflow.get("nodes", []).copy()
    if not original_nodes:
        # If there are no nodes, create a dummy node to duplicate
        original_nodes = [{"id": "dummy", "type": "dummy", "data": {"large_data": "x" * 1000}}]

    while len(json.dumps(large_workflow)) < 500000:  # 500KB
        # Add a copy of all nodes with new IDs
        for node in original_nodes:
            # Create a deep copy of the node
            new_node = json.loads(json.dumps(node))
            # Modify the ID to avoid duplicates
            new_node["id"] = f"{new_node.get('id', 'node')}_{len(large_workflow['nodes'])}"
            large_workflow["nodes"].append(new_node)

    return large_workflow


@pytest.fixture(params=["simple", "default", "large"])
def setup_test_env(request):
    """Setup test environment with temp directory and test image, parameterized by workflow type."""
    temp_dir = tempfile.mkdtemp()

    test_image = Image.new('RGB', (100, 100), color='red')

    a111_params = """
beautiful scenery nature glass bottle landscape, purple galaxy bottle, low key
Negative prompt: (worst quality, low quality, bad quality:1.3), embedding:ng_deepnegative_v1_75t, embedding:EasyNegative, embedding:badhandv4
Steps: 30, Sampler: DPM++ 2M SDE, CFG scale: 7.0, Seed: 42, Size: 512x512, Model: , Version: ComfyUI,
Civitai resources: [
    {"modelName":"Deep Negative V1.x","versionName":"V1 75T","weight":1.0,"air":"urn:air:sd1:embedding:civitai:4629@5637"},
    {"modelName":"EasyNegative","versionName":"EasyNegative_pt","weight":1.0,"air":"urn:air:sd1:embedding:civitai:7808@9536"},
    {"modelName":"badhandv4","versionName":"badhandv4","weight":1.0,"air":"urn:air:other:embedding:civitai:16993@20068"}]
"""

    prompt = {"prompt": "test prompt", "negative_prompt": "test negative prompt"}

    workflow_type = request.param

    if workflow_type == "simple":
        extra_pnginfo = {"workflow": {"version": "1.0", "nodes": []}}
    elif workflow_type == "default":
        default_workflow = get_default_workflow()
        extra_pnginfo = {"workflow": default_workflow}
    elif workflow_type == "large":
        large_workflow = get_large_workflow()
        extra_pnginfo = {"workflow": large_workflow}
        # Check the size for debugging purposes
        workflow_size = len(json.dumps(large_workflow)) / 1024  # Size in KB
        print(f"Large workflow size: {workflow_size:.2f} KB")

    yield temp_dir, test_image, a111_params, prompt, extra_pnginfo, workflow_type

    shutil.rmtree(temp_dir)


def test_save_png_with_metadata_with_workflow(setup_test_env):
    """Test that complete metadata is correctly saved and can be retrieved for PNG format."""
    temp_dir, test_image, a111_params, prompt, extra_pnginfo, workflow_type = setup_test_env

    png_path = os.path.join(temp_dir, f"test_with_workflow_{workflow_type}.png")
    save_image(
        test_image,
        png_path,
        "png",
        100,  # quality_jpeg_or_webp
        True,  # lossless_webp
        False,  # optimize_png
        a111_params,
        prompt,
        extra_pnginfo,
        True,  # embed_workflow_in_png
    )

    saved_image = Image.open(png_path)
    assert saved_image.info.get("parameters") == a111_params
    assert json.loads(saved_image.info.get("prompt")) == prompt
    assert json.loads(saved_image.info.get("workflow")) == extra_pnginfo["workflow"]


def test_save_png_with_metadata_without_workflow(setup_test_env):
    """Test that a111 metadata (without full workflow) is correctly saved and can be retrieved for PNG format."""
    temp_dir, test_image, a111_params, prompt, extra_pnginfo, workflow_type = setup_test_env

    png_no_workflow_path = os.path.join(temp_dir, f"test_no_workflow_{workflow_type}.png")
    save_image(
        test_image,
        png_no_workflow_path,
        "png",
        100,
        True,
        False,
        a111_params,
        prompt,
        extra_pnginfo,
        False,  # embed_workflow_in_png
    )

    saved_image = Image.open(png_no_workflow_path)
    assert saved_image.info.get("parameters") == a111_params
    assert "prompt" not in saved_image.info
    assert "workflow" not in saved_image.info


def test_save_jpeg_with_metadata(setup_test_env):
    """Test that metadata is correctly saved and can be retrieved for JPEG format."""
    temp_dir, test_image, a111_params, prompt, extra_pnginfo, workflow_type = setup_test_env

    jpeg_path = os.path.join(temp_dir, f"test_{workflow_type}.jpg")
    save_image(
        test_image,
        jpeg_path,
        "jpg",
        95,  # quality_jpeg_or_webp
        False,  # lossless_webp (ignored for jpeg)
        False,  # optimize_png (ignored for jpeg)
        a111_params,
        prompt,
        extra_pnginfo,
        True,  # embed_workflow_in_png (ignored for jpeg)
    )

    saved_image = Image.open(jpeg_path)
    exif_dict = piexif.load(saved_image.info["exif"])
    user_comment = piexif.helper.UserComment.load(exif_dict["Exif"][piexif.ExifIFD.UserComment])
    assert user_comment == a111_params
    # Verify that prompt and workflow data are not present
    assert "prompt" not in saved_image.info, "JPEG should not contain prompt data"
    assert "workflow" not in saved_image.info, "JPEG should not contain workflow data"


def test_save_webp_lossless_with_metadata(setup_test_env):
    """Test that metadata is correctly saved and can be retrieved for lossless WebP format."""
    temp_dir, test_image, a111_params, prompt, extra_pnginfo, workflow_type = setup_test_env

    webp_lossless_path = os.path.join(temp_dir, f"test_lossless_{workflow_type}.webp")
    save_image(
        test_image,
        webp_lossless_path,
        "webp",
        100,  # quality_jpeg_or_webp
        True,  # lossless_webp
        False,  # optimize_png (ignored for webp)
        a111_params,
        prompt,
        extra_pnginfo,
        True,  # embed_workflow_in_png (ignored for webp)
    )

    saved_image = Image.open(webp_lossless_path)
    exif_dict = piexif.load(saved_image.info["exif"])
    user_comment = piexif.helper.UserComment.load(exif_dict["Exif"][piexif.ExifIFD.UserComment])
    assert user_comment == a111_params
    # Verify that prompt and workflow data are not present
    assert "prompt" not in saved_image.info, "WebP should not contain prompt data"
    assert "workflow" not in saved_image.info, "WebP should not contain workflow data"


def test_save_webp_lossy_with_metadata(setup_test_env):
    """Test that metadata is correctly saved and can be retrieved for lossy WebP format."""
    temp_dir, test_image, a111_params, prompt, extra_pnginfo, workflow_type = setup_test_env

    # Test WebP with lossy compression
    webp_lossy_path = os.path.join(temp_dir, f"test_lossy_{workflow_type}.webp")
    save_image(
        test_image,
        webp_lossy_path,
        "webp",
        90,  # quality_jpeg_or_webp (lower quality for lossy compression)
        False,  # lossless_webp
        False,  # optimize_png (ignored for webp)
        a111_params,
        prompt,
        extra_pnginfo,
        True,  # embed_workflow_in_png (ignored for webp)
    )

    saved_image = Image.open(webp_lossy_path)
    exif_dict = piexif.load(saved_image.info["exif"])
    user_comment = piexif.helper.UserComment.load(exif_dict["Exif"][piexif.ExifIFD.UserComment])
    assert user_comment == a111_params
    # Verify that prompt and workflow data are not present
    assert "prompt" not in saved_image.info, "WebP should not contain prompt data"
    assert "workflow" not in saved_image.info, "WebP should not contain workflow data"
