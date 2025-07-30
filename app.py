from flask import Flask, request, send_file
from flask_cors import CORS
from io import BytesIO
from PIL import Image, ImageDraw
import torch
from diffusers import StableDiffusionInpaintPipeline

app = Flask(__name__)
CORS(app)

# Models
model_id_realistic = "SG161222/Realistic_Vision_V5.1_noVAE"
model_id_cartoon = "dreamlike-art/dreamlike-photoreal-2.0"
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe_realistic = StableDiffusionInpaintPipeline.from_pretrained(
    model_id_realistic,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

pipe_cartoon = StableDiffusionInpaintPipeline.from_pretrained(
    model_id_cartoon,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)


def resize_image(image, max_size=768):
    w, h = image.size
    scale = max(w, h) / max_size
    if scale > 1:
        new_w, new_h = int(w / scale), int(h / scale)
        return image.resize((new_w, new_h), Image.LANCZOS)
    return image


def generate_chest_mask(image):
    """Dynamic chest mask: adjusts based on image size."""
    mask = Image.new("RGB", image.size, (0, 0, 0))
    draw = ImageDraw.Draw(mask)
    w, h = image.size
    chest_area = [
        int(w * 0.3), int(h * 0.55),
        int(w * 0.7), int(h * 0.8)
    ]
    draw.ellipse(chest_area, fill=(255, 255, 255))
    mask.save("debug_chest_mask.png")  # For debugging
    return mask


@app.route('/generate', methods=['POST'])
def generate():
    if 'image' not in request.files:
        return "No image uploaded.", 400

    style = request.form.get('style', 'realistic')
    input_image = Image.open(request.files['image']).convert("RGB")
    input_image = resize_image(input_image)
    mask = generate_chest_mask(input_image)

    # Prompts
    negative_prompt = (
        "extra arms, deformed chest, disfigured, poorly drawn, low quality, text, watermark"
    )

    if style == 'cartoon':
        pipe = pipe_cartoon
        prompt = (
            "Add large, feminine cartoon-style breasts on the chest, perfectly aligned with the body, "
            "smooth outlines, clean anime shading, high quality."
        )
    elif style == 'extreme':
        pipe = pipe_realistic
        prompt = (
            "Add extremely large, hyper-realistic feminine breasts, soft skin, photorealistic curves, "
            "perfect blending with chest."
        )
    else:
        pipe = pipe_realistic
        prompt = (
            "Add large natural feminine breasts on the chest, photorealistic, soft lighting, "
            "realistic skin, smooth blending, high detail."
        )

    # Inpainting
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=input_image,
        mask_image=mask,
        guidance_scale=8,
        num_inference_steps=30
    ).images[0]

    img_io = BytesIO()
    result.save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
