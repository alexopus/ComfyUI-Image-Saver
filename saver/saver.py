from PIL.PngImagePlugin import PngInfo

import json
import piexif
import piexif.helper

def save_image(image, filepath, extension, quality_jpeg_or_webp, lossless_webp, optimize_png, a111_params, prompt, extra_pnginfo, embed_workflow_in_png):
    if extension == 'png':
        metadata = PngInfo()
        metadata.add_text("parameters", a111_params)

        if embed_workflow_in_png:
            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata.add_text(x, json.dumps(extra_pnginfo[x]))

        image.save(filepath, pnginfo=metadata, optimize=optimize_png)
    else: # webp & jpeg
        image.save(filepath, optimize=True, quality=quality_jpeg_or_webp, lossless=lossless_webp)
        exif_bytes = piexif.dump({
            "Exif": {
                piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(a111_params, encoding="unicode")
            },
        })
        piexif.insert(exif_bytes, filepath)
