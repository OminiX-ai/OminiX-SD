# SD_training



## Goal

## Model

You can download our efficient stable diffusion model from this [link](https://huggingface.co/piuzha/efficient_sd). It is located on Huggingface with 'piuzha/efficient_sd'.



## Requirements

Follow the [diffusers](https://huggingface.co/docs/diffusers/en/installation) package to install the environment.

## Usage

Run the following command to run inference with the model. Specify the model directory in the file
```
$ python inference.py
```

Specifically, you can load the model through 
```
from diffusers import DiffusionPipeline
pipeline = DiffusionPipeline.from_pretrained("piuzha/efficient_sd/", use_safetensors=True).to("cuda")
```
Then run the model to generate images such as
```
image = pipeline("An astronaut riding a horse, detailed, 8k", num_inference_steps=25).images[0]
image.save('test.png')
```


## Timeline





