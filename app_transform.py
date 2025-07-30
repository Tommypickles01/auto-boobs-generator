from flask import Flask, request, send_file
from flask_cors import CORS
from io import BytesIO
from PIL import Image, ImageDraw
import torch
from diffusers import StableDiffusionInpaintPipeline

app = Flask(__name__)
CORS(app)

# Model IDs
model_id_realistic = "SG161222/Realistic_Vision_V5.1_noVAE"
model_id_cartoon = "dreamlike-art/dreamlike-photoreal-2.0"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load models
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

def create_dynamic_chest_mask(image):
    """Fallback mask if user doesn't upload one."""
    w, h = image.size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    chest_top = int(h * 0.42)
    chest_bottom = int(h * 0.72)
    chest_width = int(w * 0.5)
    chest_left = (w - chest_width) // 2
    chest_right = chest_left + chest_width

    draw.rectangle((chest_left, chest_top, chest_right, chest_bottom), fill=255)

    debug_mask = mask.convert("RGB")
    debug_mask.save("debug_chest_mask.png")
    return mask.convert("RGB")

@app.route('/transform', methods=['POST'])
def transform():
    if 'image' not in request.files:
        return "No image uploaded.", 400

    style = request.form.get('style', 'realistic')
    strength = float(request.form.get('strength', 0.65))
    debug = request.args.get('debug', default='0') == '1'

    # Load input image
    input_image = Image.open(request.files['image']).convert("RGB")
    input_image = resize_image(input_image)

    # Load user mask or create fallback mask
    if 'mask' in request.files and request.files['mask'].filename != '':
        mask = Image.open(request.files['mask']).convert("RGB")
        mask = mask.resize(input_image.size, Image.LANCZOS)
    else:
        mask = create_dynamic_chest_mask(input_image)

    # Debug preview
    if debug:
        return send_file("debug_chest_mask.png", mimetype='image/png')

    negative_prompt = "bad anatomy, deformed, blurry, text, watermark, low quality"

    # Choose style
    if style == "cartoon":
        pipe = pipe_cartoon
        prompt = "A beautiful female cartoon version of this character with large bust, anime style, wearing a bikini top."
    elif style == "extreme":
        pipe = pipe_realistic
        prompt = "A hyper-realistic female version of this character with very large bust, bikini top, smooth skin, high detail."
    else:
        pipe = pipe_realistic
        prompt = "A photorealistic female version of this character with big bust, bikini top, natural curves, smooth blending."

    # Generate inpainted image
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=input_image,
        mask_image=mask,
        guidance_scale=8,
        num_inference_steps=30,
        strength=strength
    ).images[0]

    # Resize to match original image size before blending
    result = result.resize(input_image.size, Image.LANCZOS)
    mask_gray = mask.convert("L").resize(input_image.size, Image.LANCZOS)

    blended = Image.composite(result, input_image, mask_gray)

    img_io = BytesIO()
    blended.save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    import os
port = int(os.environ.get("PORT", 5000))
app.run(host='0.0.0.0', port=port)
