%cd /content
!git clone -b dev https://github.com/camenduru/InternLM-XComposer
!pip install -q https://github.com/camenduru/wheels/releases/download/colab/llava-ShareGPT4V-1.1.3-py3-none-any.whl gradio
%cd /content/InternLM-XComposer/projects/ShareGPT4V
!pip install flask-ngrok
!pip install pyngrok

import gc
import os
import signal
import sys
from threading import Thread
import time
import torch
from llava.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
from llava.conversation import (SeparatorStyle, default_conversation)
from llava.mm_utils import (KeywordsStoppingCriteria, load_image_from_base64, process_images, tokenizer_image_token)
from llava.model.builder import load_pretrained_model
from transformers import TextIteratorStreamer
from flask import Flask, send_file, request
from flask_ngrok import run_with_ngrok
from pyngrok import ngrok, conf

conf.get_default().auth_token = 'auth-token'
app = Flask(__name__)
public_url = ngrok.connect(5000).public_url
print(" * ngrok tunnel \"{}\" -> \"http://127.0.0.1:{}/\"".format(public_url, 5000))
app.config["BASE_URL"] = public_url

tokenizer, model, image_processor, context_len = load_pretrained_model("4bit/ShareGPT4V-7B-5GB", None, "llava-v1.5-7b", True, False)

TEMPERATURE = 0.2
TOP_P = 0.7
MAX_NEW_TOKENS = 1024
MAX_OUTPUT_TOKENS =  min(int(MAX_NEW_TOKENS), 1536)
state = default_conversation.copy()
SEP = state.sep if state.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else state.sep2

@app.route("/describe")
def describe_image():
    data = request.get_json()
    prompt = data["prompt"]
    all_images = [data["image"]]
    response = get_response(prompt, all_images, model, image_processor, tokenizer, TEMPERATURE, TOP_P, MAX_OUTPUT_TOKENS, SEP)
    return response

@app.route("/prompt")
def generate_prompt():
    prompt = request.args.get('prompt', '')
    response = get_response(prompt, [], model, image_processor, tokenizer, TEMPERATURE, TOP_P, MAX_OUTPUT_TOKENS, SEP)
    return response

@torch.inference_mode()
def get_response(prompt, images, model, image_processor, tokenizer, temperature, top_p, max_new_tokens, stop):
    ori_prompt = prompt
    num_image_tokens = 0
    if images is not None and len(images) > 0:
        if len(images) != prompt.count(DEFAULT_IMAGE_TOKEN):
            raise ValueError(
                "Number of images does not match number of <image> tokens in prompt")

        images = [load_image_from_base64(image) for image in images]
        images = process_images(images, image_processor, model.config)

        if type(images) is list:
            images = [image.to(model.device, dtype=torch.float16)
                        for image in images]
        else:
            images = images.to(model.device, dtype=torch.float16)

        replace_token = DEFAULT_IMAGE_TOKEN
        if getattr(model.config, 'mm_use_im_start_end', False):
            replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
        prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

        num_image_tokens = prompt.count(
            replace_token) * model.get_vision_tower().num_patches
        image_args = {"images": images}
    else:
        images = None
        image_args = {}

    temperature = float(temperature)
    top_p = float(top_p)
    max_context_length = getattr(
        model.config, 'max_position_embeddings', 2048)
    max_new_tokens = min(int(max_new_tokens), 1024)
    stop_str = stop
    do_sample = True if temperature > 0.001 else False

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(
        keywords, tokenizer, input_ids)
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=15)

    max_new_tokens = min(max_new_tokens, max_context_length -
                         input_ids.shape[-1] - num_image_tokens)

    if max_new_tokens < 1:
        raise ValueError(ori_prompt + "Exceeds max token length. Please start a new conversation, thanks.")

    # local inference
    thread = Thread(target=model.generate, kwargs=dict(
        inputs=input_ids,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        streamer=streamer,
        stopping_criteria=[stopping_criteria],
        use_cache=True,
        **image_args
    ))
    thread.start()

    generated_text = str()
    for new_text in streamer:
        generated_text += new_text
        if generated_text.endswith(stop_str):
            generated_text = generated_text[:-len(stop_str)]
    return generated_text

app.run()
