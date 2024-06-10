# SD_training



## Goal

## Model

You can download our efficient stable diffusion model from this [link](https://huggingface.co/piuzha/efficient_sd). It is located on Huggingface with 'piuzha/efficient_sd'.



## Requirements

Follow the [diffusers](https://huggingface.co/docs/diffusers/en/installation) package to install the environment.

To prepare the dataset, you can install the img2dataset package.
```
pip install img2dataset
```


## Datasets

There are multiple datasets available. The scripts to download the datasets are located under the dataset_examples directory. You can refer to the specific script for details. 


## Training

We follow  a stand  method to train the stable diffusion model. You can refer to the [huggingface diffusers text_to_image](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py) script to train the text2image diffusion model. 

For example, you can finetune the model with the following command,
```
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export dataset_name="lambdalabs/naruto-blip-captions"

accelerate launch --mixed_precision="fp16"  train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$dataset_name \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --enable_xformers_memory_efficient_attention
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="sd-naruto-model" \
  --push_to_hub
```

More details about the training of the diffusion model can be find [here](https://huggingface.co/docs/diffusers/en/training/text2image).   You can also try training with other methods such as lora following [this](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py).


## Inference

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

