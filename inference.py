from diffusers import DiffusionPipeline
import torch
import time

#pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", use_safetensors=True)
pipeline = DiffusionPipeline.from_pretrained("piuzha/efficient_sd/", use_safetensors=True).to("cuda")

image = pipeline(
	"An astronaut riding a horse, detailed, 8k", num_inference_steps=25).images[0]

image.save('test.png')


