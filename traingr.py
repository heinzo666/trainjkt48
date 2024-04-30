import gradio as gr
import numpy as np
import torch
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import os
from diffusers import StableDiffusionInpaintPipeline
from diffusers import DPMSolverMultistepSchedule

HOME = os.getcwd()
print("HOME:", HOME)

CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")
print(CHECKPOINT_PATH, "; exist:", os.path.isfile(CHECKPOINT_PATH))
import torch

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
predictor = SamPredictor(sam)

model_id_or_path = "Uminosachi/realisticVisionV51_v51VAE-inpainting"
pipe = StableDiffusionInpaintPipeline.from_pretrained(
model_id_or_path, torch_dtype=torch.float16
 )
pipe = pipe.to("cuda")
pipe.safety_checker = None
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_sequential_cpu_offload()
pipe.enable_xformers_memory_efficient_attention()
pipe.load_textual_inversion("/content/embeddings/bad-hands-5.pt", token="bad_hands5")
pipe.load_textual_inversion("/content/embeddings/negative_hand-neg.pt", token="negative_hands")
pipe.load_textual_inversion("/content/embeddings/breasts.pt", token="breastAI")
pipe.load_textual_inversion("/content/embeddings/ulzzang-6500.pt", token="ulzzang-6500")
pipe.load_textual_inversion("/content/embeddings/bukkakAI.pt", token="bukkaAI")

selected_pixels = []

with gr.Blocks() as demo:
    with gr.Row():
       input_img = gr.Image(label="Input")
       mask_img = gr.Image(label="Mask")
       output_img = gr.Image(label="Output")
    with gr.Row():
        prompt = gr.Textbox(lines=1, label="Prompt")
    with gr.Row():
        submit = gr.Button("Submit")
    def generate_mask(image, evt: gr.SelectData):
        selected_pixels.append(evt.index)

        predictor.set_image(image)
        input_points = np.array(selected_pixels)
        input_labels = np.ones(input_points.shape[0])
        mask, _, _ = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=False
        )
        mask = Image.fromarray(mask[0, :, :])
        return mask

    def inpaint(image, mask, prompt)
        image = Image.fromarray(image)
        mask = Image.fromarray(mask)

        image = image.resize((512, 512))
        mask = image.resize((512, 512))

        output = pipe(prompt=prompt, image=image, mask_image=mask,).images[0]

        return output
    
    input_img.select(generate_mask, [input_img], [mask_img])
    submit.click(
        inpaint,
        inputs=[input_img, mask_img, prompt],
        output=[output_img],
    )
if __name__ == "__main__":
    demo.launch()
