# OminiX Stable Diffusion



## Goal

Generative AI (GAI) offers unprecedented opportunities for research and innovation, but its commercialization has raised concerns about transparency, reproducibility, and safety. Many open GAI models lack the necessary components for full understanding and reproducibility, and some use restrictive licenses whilst claiming to be “open-source”. To address these concerns, we follow the [Model Openness Framework (MOF)](https://arxiv.org/pdf/2403.13784), a ranked classification system that rates machine learning models based on their completeness and openness, following principles of open science, open source, open data, and open access. 

By promoting transparency and reproducibility, the MOF combats “openwashing” practices and establishes completeness and openness as primary criteria alongside the core tenets of responsible AI. Wide adoption of the MOF will foster a more open AI ecosystem, benefiting research, innovation, and adoption of state-of-the-art models. 

We follow MOF to release the datasets during training, the training scripts, and the trained models. 



## Requirements

#### 1. Diffusers package
Follow the [diffusers](https://huggingface.co/docs/diffusers/en/installation) package to install the environment.

#### 2. Dataset package

To prepare the dataset, you can install the img2dataset package.
```
pip install img2dataset
```

#### 3. SD Webui (optional)

Follow this [link](https://github.com/AUTOMATIC1111/stable-diffusion-webui/tree/master) to install the webui environment. 

Specifically, you can follow the follwoing instructions.
```
sudo apt install git software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt install python3.10-venv -y
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui && cd stable-diffusion-webui
python3.10 -m venv venv
./webui.sh
```


## Datasets

There are multiple datasets available. The scripts to download the datasets are located under the dataset_examples directory. You can refer to the specific script for details. 



## Model Download

You can download our efficient stable diffusion model from this [link](https://huggingface.co/piuzha/efficient_sd). It is located on Huggingface with 'piuzha/efficient_sd'.

We adopt a more efficient SD model. The  model architecture files of our model and the original SD 1.5 model are shown under ./docs/ .

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


## SD Webui

Our model can be used in [SD Webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui). 

You need to download the model from this [link](https://huggingface.co/piuzha/efficient_sd). Put the model under the 'stable-diffusion-webui/models/Stable-diffusion/' directory.  

You also need to use an  updated config file for our model to replace the original config file 'stable-diffusion-webui/configs/v1-inference.yaml'.  The new config file can be found under our 'configs/v1-inference.yaml'. 



## Timeline

<p style="text-align: center;">Timeline for OminiX for Stable Diffusion development and open-source</p>

| Time          	| Task                                                                                                               	| Open Source version                                                                                                                                                                	|
|---------------	|--------------------------------------------------------------------------------------------------------------------	|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------	|
| 05/21         	|                                                                                                                    	| The first open source: <br>  (i) Mac/iPhone version of stable diffusion execution (SDXL, SD2.1)  <br> (ii) Current version of the optimized SD model                                        	|
| 05/19 – 06/14 	| Versatile stable diffusion deployment framework on Mac/iPhone, supporting various base, checkpoint and LoRA models 	|                                                                                                                                                                                    	|
| 06/15 – 07/31 	| Further optimization for speed (operator fusion and memory layout optimization, etc.)                              	| 06/30: Release the versatile SD deployment framework on Mac/iPhone, along with the OminiX front-end operator library <br>  08/01: Release the OminiX back-end for SD with acceleration 	|
| 06/15 – 07/31 	| Further optimized stable diffusion model training                                                                  	| 07/15: Release the further optimized SD model (with distillation)                                                                                                                  	|
| 06/01 – 08/31 	| OminiX backend for stable diffusion on Android devices                                                             	| 09/01: Release the OminiX backend for stable diffusion on Android devices                                                                                                          	|

