from diffusers import AutoPipelineForText2Image,DiffusionPipeline
import torch
	
import time


#pipeline = AutoPipelineForText2Image.from_pretrained(
#	"runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16").to("cuda")


#pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", use_safetensors=True)
pipeline = DiffusionPipeline.from_pretrained("./ckpts", use_safetensors=True).to("cuda")


image = pipeline(
	"A cute cat under sofa, detailed, 8k", num_inference_steps=25).images[0]

image.save('test.png')


