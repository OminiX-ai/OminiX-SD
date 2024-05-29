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

<p style="text-align: center;">Timeline for OminiX for Stable Diffusion development and open-source</p>

| Time          	| Task                                                                                                               	| Open Source version                                                                                                                                                                	|
|---------------	|--------------------------------------------------------------------------------------------------------------------	|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------	|
| 05/21         	|                                                                                                                    	| The first open source: <br>  (i) Mac/iPhone version of stable diffusion execution (SDXL, SD2.1)  <br> (ii) Current version of the optimized SD model                                        	|
| 05/19 – 06/14 	| Versatile stable diffusion deployment framework on Mac/iPhone, supporting various base, checkpoint and LoRA models 	|                                                                                                                                                                                    	|
| 06/15 – 07/31 	| Further optimization for speed (operator fusion and memory layout optimization, etc.)                              	| 06/30: Release the versatile SD deployment framework on Mac/iPhone, along with the OminiX front-end operator library <br>  08/01: Release the OminiX back-end for SD with acceleration 	|
| 06/15 – 07/31 	| Further optimized stable diffusion model training                                                                  	| 07/15: Release the further optimized SD model (with distillation)                                                                                                                  	|
| 06/01 – 08/31 	| OminiX backend for stable diffusion on Android devices                                                             	| 09/01: Release the OminiX backend for stable diffusion on Android devices                                                                                                          	|

