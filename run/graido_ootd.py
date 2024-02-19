import gradio as gr
import os
from pathlib import Path
import sys
import torch
from PIL import Image, ImageOps

from utils_ootd import get_mask_location

PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# import argparse
# parser = argparse.ArgumentParser(description='ootd')
# parser.add_argument('--gpuid', '-g', type=int, default=0, required=False)
# args = parser.parse_args()

import time
from preprocess.openpose.run_openpose import OpenPose
from preprocess.humanparsing.aigc_run_parsing import Parsing
from ootd.inference_ootd_hd import OOTDiffusionHD
from ootd.inference_ootd_dc import OOTDiffusionDC


openpose_model_hd = OpenPose(0)
parsing_model_hd = Parsing(0)
ootd_model_hd = OOTDiffusionHD(0)

openpose_model_dc = OpenPose(1)
parsing_model_dc = Parsing(1)
ootd_model_dc = OOTDiffusionDC(1)


category_dict = ['upperbody', 'lowerbody', 'dress']
category_dict_utils = ['upper_body', 'lower_body', 'dresses']


example_path = os.path.join(os.path.dirname(__file__), 'examples')
model_hd = os.path.join(example_path, 'model/model_1.png')
garment_hd = os.path.join(example_path, 'garment/03244_00.jpg')
model_dc = os.path.join(example_path, 'model/model_8.png')
garment_dc = os.path.join(example_path, 'garment/048554_1.jpg')

def process_hd(vton_img, garm_img, n_samples, n_steps, image_scale, seed):
    model_type = 'hd'
    category = 0 # 0:upperbody; 1:lowerbody; 2:dress

    with torch.no_grad():
        garm_img = Image.open(garm_img).resize((768, 1024))
        vton_img = Image.open(vton_img).resize((768, 1024))
        keypoints = openpose_model_hd(vton_img.resize((384, 512)))
        model_parse, _ = parsing_model_hd(vton_img.resize((384, 512)))

        mask, mask_gray = get_mask_location(model_type, category_dict_utils[category], model_parse, keypoints)
        mask = mask.resize((768, 1024), Image.NEAREST)
        mask_gray = mask_gray.resize((768, 1024), Image.NEAREST)
        
        masked_vton_img = Image.composite(mask_gray, vton_img, mask)

        images = ootd_model_hd(
            model_type=model_type,
            category=category_dict[category],
            image_garm=garm_img,
            image_vton=masked_vton_img,
            mask=mask,
            image_ori=vton_img,
            num_samples=n_samples,
            num_steps=n_steps,
            image_scale=image_scale,
            seed=seed,
        )

    return images

def process_dc(vton_img, garm_img, category, n_samples, n_steps, image_scale, seed):
    model_type = 'dc'
    if category == 'Upper-body':
        category = 0
    elif category == 'Lower-body':
        category = 1
    else:
        category =2

    with torch.no_grad():
        garm_img = Image.open(garm_img).resize((768, 1024))
        vton_img = Image.open(vton_img).resize((768, 1024))
        keypoints = openpose_model_dc(vton_img.resize((384, 512)))
        model_parse, _ = parsing_model_dc(vton_img.resize((384, 512)))

        mask, mask_gray = get_mask_location(model_type, category_dict_utils[category], model_parse, keypoints)
        mask = mask.resize((768, 1024), Image.NEAREST)
        mask_gray = mask_gray.resize((768, 1024), Image.NEAREST)
        
        masked_vton_img = Image.composite(mask_gray, vton_img, mask)

        images = ootd_model_dc(
            model_type=model_type,
            category=category_dict[category],
            image_garm=garm_img,
            image_vton=masked_vton_img,
            mask=mask,
            image_ori=vton_img,
            num_samples=n_samples,
            num_steps=n_steps,
            image_scale=image_scale,
            seed=seed,
        )

    return images


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("# OOTDiffusion Demo")
    with gr.Row():
        gr.Markdown("## Half-body")
    with gr.Row():
        gr.Markdown("***Support upper-body garments***")
    with gr.Row():
        with gr.Column():
            vton_img = gr.Image(label="Model", sources='upload', type="filepath", height=384, value=model_hd)
            example = gr.Examples(
                inputs=vton_img,
                examples_per_page=14,
                examples=[
                    os.path.join(example_path, 'model/model_1.png'),
                    os.path.join(example_path, 'model/model_2.png'),
                    os.path.join(example_path, 'model/model_3.png'),
                    os.path.join(example_path, 'model/model_4.png'),
                    os.path.join(example_path, 'model/model_5.png'),
                    os.path.join(example_path, 'model/model_6.png'),
                    os.path.join(example_path, 'model/model_7.png'),
                    os.path.join(example_path, 'model/01008_00.jpg'),
                    os.path.join(example_path, 'model/07966_00.jpg'),
                    os.path.join(example_path, 'model/05997_00.jpg'),
                    os.path.join(example_path, 'model/02849_00.jpg'),
                    os.path.join(example_path, 'model/14627_00.jpg'),
                    os.path.join(example_path, 'model/09597_00.jpg'),
                    os.path.join(example_path, 'model/01861_00.jpg'),
                ])
        with gr.Column():
            garm_img = gr.Image(label="Garment", sources='upload', type="filepath", height=384, value=garment_hd)
            example = gr.Examples(
                inputs=garm_img,
                examples_per_page=14,
                examples=[
                    os.path.join(example_path, 'garment/03244_00.jpg'),
                    os.path.join(example_path, 'garment/00126_00.jpg'),
                    os.path.join(example_path, 'garment/03032_00.jpg'),
                    os.path.join(example_path, 'garment/06123_00.jpg'),
                    os.path.join(example_path, 'garment/02305_00.jpg'),
                    os.path.join(example_path, 'garment/00055_00.jpg'),
                    os.path.join(example_path, 'garment/00470_00.jpg'),
                    os.path.join(example_path, 'garment/02015_00.jpg'),
                    os.path.join(example_path, 'garment/10297_00.jpg'),
                    os.path.join(example_path, 'garment/07382_00.jpg'),
                    os.path.join(example_path, 'garment/07764_00.jpg'),
                    os.path.join(example_path, 'garment/00151_00.jpg'),
                    os.path.join(example_path, 'garment/12562_00.jpg'),
                    os.path.join(example_path, 'garment/04825_00.jpg'),
                ])
        with gr.Column():
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery", preview=True, scale=1)   
    with gr.Column():
        run_button = gr.Button(value="Run")
        n_samples = gr.Slider(label="Images", minimum=1, maximum=4, value=1, step=1)
        n_steps = gr.Slider(label="Steps", minimum=20, maximum=40, value=20, step=1)
        # scale = gr.Slider(label="Scale", minimum=1.0, maximum=12.0, value=5.0, step=0.1)
        image_scale = gr.Slider(label="Guidance scale", minimum=1.0, maximum=5.0, value=2.0, step=0.1)
        seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, value=-1)
        
    ips = [vton_img, garm_img, n_samples, n_steps, image_scale, seed]
    run_button.click(fn=process_hd, inputs=ips, outputs=[result_gallery])


    with gr.Row():
        gr.Markdown("## Full-body")
    with gr.Row():
        gr.Markdown("***Support upper-body/lower-body/dresses; garment category must be paired!!!***")
    with gr.Row():
        with gr.Column():
            vton_img_dc = gr.Image(label="Model", sources='upload', type="filepath", height=384, value=model_dc)
            example = gr.Examples(
                label="Examples (upper-body/lower-body)",
                inputs=vton_img_dc,
                examples_per_page=7,
                examples=[
                    os.path.join(example_path, 'model/model_8.png'),
                    os.path.join(example_path, 'model/049447_0.jpg'),
                    os.path.join(example_path, 'model/049713_0.jpg'),
                    os.path.join(example_path, 'model/051482_0.jpg'),
                    os.path.join(example_path, 'model/051918_0.jpg'),
                    os.path.join(example_path, 'model/051962_0.jpg'),
                    os.path.join(example_path, 'model/049205_0.jpg'),
                ])
            example = gr.Examples(
                label="Examples (dress)",
                inputs=vton_img_dc,
                examples_per_page=7,
                examples=[
                    os.path.join(example_path, 'model/model_9.png'),
                    os.path.join(example_path, 'model/052767_0.jpg'),
                    os.path.join(example_path, 'model/052472_0.jpg'),
                    os.path.join(example_path, 'model/053514_0.jpg'),
                    os.path.join(example_path, 'model/053228_0.jpg'),
                    os.path.join(example_path, 'model/052964_0.jpg'),
                    os.path.join(example_path, 'model/053700_0.jpg'),
                ])
        with gr.Column():
            garm_img_dc = gr.Image(label="Garment", sources='upload', type="filepath", height=384, value=garment_dc)
            category_dc = gr.Dropdown(label="Garment category (important option!!!)", choices=["Upper-body", "Lower-body", "Dress"], value="Upper-body")
            example = gr.Examples(
                label="Examples (upper-body)",
                inputs=garm_img_dc,
                examples_per_page=7,
                examples=[
                    os.path.join(example_path, 'garment/048554_1.jpg'),
                    os.path.join(example_path, 'garment/049920_1.jpg'),
                    os.path.join(example_path, 'garment/049965_1.jpg'),
                    os.path.join(example_path, 'garment/049949_1.jpg'),
                    os.path.join(example_path, 'garment/050181_1.jpg'),
                    os.path.join(example_path, 'garment/049805_1.jpg'),
                    os.path.join(example_path, 'garment/050105_1.jpg'),
                ])
            example = gr.Examples(
                label="Examples (lower-body)",
                inputs=garm_img_dc,
                examples_per_page=7,
                examples=[
                    os.path.join(example_path, 'garment/051827_1.jpg'),
                    os.path.join(example_path, 'garment/051946_1.jpg'),
                    os.path.join(example_path, 'garment/051473_1.jpg'),
                    os.path.join(example_path, 'garment/051515_1.jpg'),
                    os.path.join(example_path, 'garment/051517_1.jpg'),
                    os.path.join(example_path, 'garment/051988_1.jpg'),
                    os.path.join(example_path, 'garment/051412_1.jpg'),
                ])
            example = gr.Examples(
                label="Examples (dress)",
                inputs=garm_img_dc,
                examples_per_page=7,
                examples=[
                    os.path.join(example_path, 'garment/053290_1.jpg'),
                    os.path.join(example_path, 'garment/053744_1.jpg'),
                    os.path.join(example_path, 'garment/053742_1.jpg'),
                    os.path.join(example_path, 'garment/053786_1.jpg'),
                    os.path.join(example_path, 'garment/053790_1.jpg'),
                    os.path.join(example_path, 'garment/053319_1.jpg'),
                    os.path.join(example_path, 'garment/052234_1.jpg'),
                ])
        with gr.Column():
            result_gallery_dc = gr.Gallery(label='Output', show_label=False, elem_id="gallery", preview=True, scale=1)   
    with gr.Column():
        run_button_dc = gr.Button(value="Run")
        n_samples_dc = gr.Slider(label="Images", minimum=1, maximum=4, value=1, step=1)
        n_steps_dc = gr.Slider(label="Steps", minimum=20, maximum=40, value=20, step=1)
        # scale_dc = gr.Slider(label="Scale", minimum=1.0, maximum=12.0, value=5.0, step=0.1)
        image_scale_dc = gr.Slider(label="Guidance scale", minimum=1.0, maximum=5.0, value=2.0, step=0.1)
        seed_dc = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, value=-1)
        
    ips_dc = [vton_img_dc, garm_img_dc, category_dc, n_samples_dc, n_steps_dc, image_scale_dc, seed_dc]
    run_button_dc.click(fn=process_dc, inputs=ips_dc, outputs=[result_gallery_dc])


block.launch(server_name='0.0.0.0', server_port=7865)