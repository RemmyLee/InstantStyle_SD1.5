import sys

sys.path.append("./")


import os
import cv2
import torch
import random
import numpy as np
from PIL import Image
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
)
from datetime import datetime

# import spaces
import gradio as gr
from huggingface_hub import hf_hub_download
import argparse

from ip_adapter import IPAdapter

# get current date without time
now = datetime.now().strftime("%Y-%m-%d")
time = datetime.now().strftime("%H-%M-%S")

parser = argparse.ArgumentParser(description="Run the InstantStyle Gradio Interface")
parser.add_argument("--inbrowser", action="store_true", help="Open in browser")
parser.add_argument("--server_port", type=int, default=7860, help="Server port")
parser.add_argument("--share", action="store_true", help="Share the Gradio UI")
parser.add_argument(
    "--model_path",
    type=str,
    default="SG161222/Realistic_Vision_V6.0_B1_noVAE",
    help="Base model path",
)
args = parser.parse_args()

# global variable
MAX_SEED = np.iinfo(np.int32).max
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
# device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = (
    torch.float16
    if str(device).__contains__("cuda") or str(device).__contains__("mps")
    else torch.float32
)

# initialization
# base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
# base_model_path = "RunDiffusion/Juggernaut-XL-v8"
base_model_path = args.model_path
image_encoder_path = "models/image_encoder"
ip_ckpt = "models/ip-adapter_sd15.bin"

controlnet_path = "diffusers/controlnet"
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny", use_safetensors=False, torch_dtype=dtype
).to(device)

# load SDXL pipeline
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model_path,
    controlnet=controlnet,
    torch_dtype=dtype,
    add_watermarker=False,
    safety_checker=None,
    requires_safety_checker=False,
)


# load ip-adapter
# target_blocks=["block"] for original IP-Adapter
# target_blocks=["up_blocks.0.attentions.1"] for style blocks only
# target_blocks = ["up_blocks.0.attentions.1", "down_blocks.2.attentions.1"] # for style+layout blocks
ip_model = IPAdapter(
    pipe,
    image_encoder_path,
    ip_ckpt,
    device,
)


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


def resize_img(
    input_image,
    max_side=768,
    min_side=512,
    size=None,
    pad_to_max_side=False,
    mode=Image.BILINEAR,
    base_pixel_number=64,
):
    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio * w), round(ratio * h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio * w), round(ratio * h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y : offset_y + h_resize_new, offset_x : offset_x + w_resize_new] = (
            np.array(input_image)
        )
        input_image = Image.fromarray(res)
    return input_image


def get_example():
    case = [
        [
            "./assets/0.jpg",
            None,
            "a cat, masterpiece, best quality, high quality",
            1.0,
            0.0,
        ],
        [
            "./assets/1.jpg",
            None,
            "a cat, masterpiece, best quality, high quality",
            1.0,
            0.0,
        ],
        [
            "./assets/2.jpg",
            None,
            "a cat, masterpiece, best quality, high quality",
            1.0,
            0.0,
        ],
        [
            "./assets/3.jpg",
            None,
            "a cat, masterpiece, best quality, high quality",
            1.0,
            0.0,
        ],
        [
            "./assets/2.jpg",
            "./assets/yann-lecun.jpg",
            "a man, masterpiece, best quality, high quality",
            1.0,
            0.6,
        ],
    ]
    return case


def run_for_examples(style_image, source_image, prompt, scale, control_scale):

    return create_image(
        image_pil=style_image,
        input_image=source_image,
        prompt=prompt,
        n_prompt="text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
        scale=scale,
        control_scale=control_scale,
        guidance_scale=5,
        num_samples=1,
        num_inference_steps=20,
        seed=42,
        target="Load only style blocks",
        neg_content_prompt="",
        neg_content_scale=0,
    )


# @spaces.GPU(enable_queue=True)
def create_image(
    image_pil,
    input_image,
    prompt,
    n_prompt,
    scale,
    control_scale,
    guidance_scale,
    num_samples,
    num_inference_steps,
    seed,
    target="Load only style blocks",
    neg_content_prompt=None,
    neg_content_scale=0,
):

    if target == "Load original IP-Adapter":
        # target_blocks=["blocks"] for original IP-Adapter
        ip_model = IPAdapter(
            pipe, image_encoder_path, ip_ckpt, device, target_blocks=["blocks"]
        )
    elif target == "Load only style blocks":
        # target_blocks=["up_blocks.0.attentions.1"] for style blocks only
        ip_model = IPAdapter(
            pipe,
            image_encoder_path,
            ip_ckpt,
            device,
            target_blocks=["up_blocks.0.attentions.1"],
        )
    elif target == "Load style+layout block":
        # target_blocks = ["up_blocks.0.attentions.1", "down_blocks.2.attentions.1"] # for style+layout blocks
        ip_model = IPAdapter(
            pipe,
            image_encoder_path,
            ip_ckpt,
            device,
            target_blocks=["up_blocks.0.attentions.1", "down_blocks.2.attentions.1"],
        )

    if input_image is not None:
        input_image = resize_img(input_image, max_side=768)
        cv_input_image = pil_to_cv2(input_image)
        detected_map = cv2.Canny(cv_input_image, 50, 200)  #
        canny_map = Image.fromarray(cv2.cvtColor(detected_map, cv2.COLOR_BGR2RGB))
    else:
        canny_map = Image.new("RGB", (512, 512), color=(255, 255, 255))
        control_scale = 0

    if float(control_scale) == 0:
        canny_map = canny_map.resize((512, 512))

    if len(neg_content_prompt) > 0 and neg_content_scale != 0:
        images = ip_model.generate(
            pil_image=image_pil,
            prompt=prompt,
            negative_prompt=n_prompt,
            scale=scale,
            guidance_scale=guidance_scale,
            num_samples=num_samples,
            num_inference_steps=num_inference_steps,
            seed=seed,
            image=canny_map,
            controlnet_conditioning_scale=float(control_scale),
            neg_content_prompt=neg_content_prompt,
            neg_content_scale=neg_content_scale,
        )
    else:
        images = ip_model.generate(
            pil_image=image_pil,
            prompt=prompt,
            negative_prompt=n_prompt,
            scale=scale,
            guidance_scale=guidance_scale,
            num_samples=num_samples,
            num_inference_steps=num_inference_steps,
            seed=seed,
            image=canny_map,
            controlnet_conditioning_scale=float(control_scale),
        )
    # Save the image
    if not os.path.exists(f"output/{now}"):
        os.makedirs(now)
    images[0].save(f"output/{now}/{seed}{time}.jpg")
    return images


def clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def pil_to_cv2(image_pil):
    image_np = np.array(image_pil)
    image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    return image_cv2


# Description
title = r"""
<h1 align="center">InstantStyle: Style-Preserving in Text-to-Image Generation</h1>
"""

description = r"""
<b>Gradio demo</b> for <a href='https://github.com/InstantStyle/InstantStyle' target='_blank'><b>InstantStyle: Style-Preserving in Text-to-Image Generation</b></a>.<br>

How to use:<br>
1. Upload a style image.
2. Set stylization mode, only use style block by default.
2. Enter a text prompt, as done in normal text-to-image models.
3. Click the <b>Submit</b> button to begin customization.
4. Share your stylized photo with your friends and enjoy! 


Advanced usage:<br>
1. Click advanced options.
2. Upload another source image for image-based stylization using ControlNet.
3. Enter negative content prompt to avoid content leakage.
"""

article = r"""
---
**Citation**
<br>
If our work is helpful for your research or applications, please cite us via:
```bibtex
@article{wang2024instantstyle,
  title={InstantStyle: Free Lunch towards Style-Preserving in Text-to-Image Generation},
  author={Wang, Haofan and Wang, Qixun and Bai, Xu and Qin, Zekui and Chen, Anthony},
  journal={arXiv preprint arXiv:2404.02733},
  year={2024}
}
```
"""

block = gr.Blocks(css="footer {visibility: hidden}").queue(max_size=10, api_open=False)
with block:

    # description
    gr.Markdown(title)
    # gr.Markdown(description)

    with gr.Tabs():
        with gr.Row():
            with gr.Column():

                with gr.Row():
                    with gr.Column():
                        image_pil = gr.Image(label="Style Image", type="pil")

                target = gr.Radio(
                    [
                        "Load only style blocks",
                        "Load style+layout block",
                        "Load original IP-Adapter",
                    ],
                    value="Load style+layout block",
                    label="Style mode",
                )

                prompt = gr.Textbox(
                    label="Prompt",
                    value="a pencil sketch of a woman, masterpiece, best quality, high quality",
                )

                scale = gr.Slider(
                    minimum=0, maximum=5.0, step=0.01, value=5.0, label="Scale"
                )

                with gr.Accordion(open=False, label="Advanced Options"):

                    with gr.Column():
                        src_image_pil = gr.Image(
                            label="Source Image (optional)", type="pil"
                        )
                    control_scale = gr.Slider(
                        minimum=0,
                        maximum=5.0,
                        step=0.1,
                        value=1.6,
                        label="Controlnet conditioning scale",
                    )

                    n_prompt = gr.Textbox(
                        label="Neg Prompt",
                        value="text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry, color",
                    )

                    neg_content_prompt = gr.Textbox(
                        label="Neg Content Prompt", value=""
                    )
                    neg_content_scale = gr.Slider(
                        minimum=0,
                        maximum=1.0,
                        step=0.01,
                        value=0.5,
                        label="Neg Content Scale",
                    )

                    guidance_scale = gr.Slider(
                        minimum=1,
                        maximum=15.0,
                        step=0.01,
                        value=5.0,
                        label="guidance scale",
                    )
                    num_samples = gr.Slider(
                        minimum=1, maximum=4.0, step=1.0, value=1.0, label="num samples"
                    )
                    num_inference_steps = gr.Slider(
                        minimum=5,
                        maximum=50.0,
                        step=1.0,
                        value=20,
                        label="num inference steps",
                    )
                    seed = gr.Slider(
                        minimum=-1,
                        maximum=4294967295,
                        value=1,
                        step=1,
                        label="Seed Value",
                    )
                    randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

                # generate_button = gr.Button("Generate Image")

            with gr.Column():
                generated_image = gr.Gallery(label="Generated Image")
                generate_button = gr.Button("Generate Image")
                gr.Markdown(description)

        generate_button.click(
            fn=randomize_seed_fn,
            inputs=[seed, randomize_seed],
            outputs=seed,
            queue=False,
            api_name=False,
        ).then(
            fn=create_image,
            inputs=[
                image_pil,
                src_image_pil,
                prompt,
                n_prompt,
                scale,
                control_scale,
                guidance_scale,
                num_samples,
                num_inference_steps,
                seed,
                target,
                neg_content_prompt,
                neg_content_scale,
            ],
            outputs=[generated_image],
        ).then(
            fn=clear_cuda_cache
        )

    gr.Examples(
        examples=get_example(),
        inputs=[image_pil, src_image_pil, prompt, scale, control_scale],
        fn=run_for_examples,
        outputs=[generated_image],
        # cache_examples=True,
    )

    gr.Markdown(article)

# block.launch()
block.launch(inbrowser=args.inbrowser, server_port=args.server_port, share=args.share)
