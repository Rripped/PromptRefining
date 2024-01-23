import io
import torch
import gc
from diffusers import StableDiffusionXLPipeline
from flask import Flask, send_file, request

app = Flask(__name__)
@app.route("/generate")
def generate():
    prompt = request.args.get('prompt', '')
    image = generate_image(prompt)
    byte_io = io.BytesIO()
    image.save(byte_io, 'PNG')
    byte_io.seek(0)
    return send_file(byte_io, mimetype='image/png')

def generate_image(prompt: str):
    print(f'Generate image for prompt "{prompt}"')
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")
    image = pipe(prompt, use_karras_sigmas=True, num_inference_steps=100).images[0]
    del pipe
    flush()
    return image

def flush():
  gc.collect()
  torch.cuda.empty_cache()
