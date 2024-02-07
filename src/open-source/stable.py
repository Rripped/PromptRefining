import io
from multiprocess import Process
import os
import signal
import time
import torch
import gc
from diffusers import StableDiffusionXLPipeline
from flask import Flask, send_file, request

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

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
    image = pipe(prompt, use_karras_sigmas=True, num_inference_steps=200).images[0]
    return image

app.run("127.0.0.1", 5000)
