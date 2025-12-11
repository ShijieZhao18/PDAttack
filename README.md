# PDAttack: Enhancing Transferability of Unrestricted Adversarial Examples via Prompt-Driven Diffusion
   The code repository for our paper PDAttack: Enhancing Transferability of Unrestricted Adversarial Examples via Prompt-Driven Diffusion.
## Overview
<div>
  <img src="fig/Overview.png" width="90%" alt="Overview">
</div>
   
If the image doesn't display properly, you can click [here](fig/Overview.png) to view our framework.
## Requirements

1. Hardware Requirements
    - GPU: 1x high-end NVIDIA GPU with at least 24GB memory
    - Memory: At least 40GB of storage memory

2. Software Requirements
    - Python: 3.10
    - CUDA: 12.2

   To install other requirements:

   ```bash
   pip install -r requirements.txt
   ```
3. Datasets
   - Please download the dataset [ImageNet-Compatible](https://github.com/cleverhans-lab/cleverhans/tree/master/cleverhans_v3.1.0/examples/nips17_adversarial_competition/dataset) and then change the settings of `--images_root` and `--label_path` in [main.py](main.py)
   - Consistent with the **ImageNet-Compatible** setup, we randomly select 1,000 images from the **CUB-200-2011** and **Stanford Cars** datasets. You can download these datasets here: [[CUB-200-2011](https://drive.google.com/file/d/1umBxwhRz6PIG6cli40Fc0pAFl2DFu9WQ/view?usp=sharing) | [Stanford Cars](https://drive.google.com/file/d/1FiH98QyyM9YQ70PPJD4-CqOBZAIMlWJL/view?usp=sharing)]. After downloading, please update the `--images_root` and `--label_path` settings in [main.py](main.py). Additionally, ensure that you set `--dataset_name` to `cub_200_2011` or `standford_car` when running the code.

4. Pre-trained Models
   - Diffusion Model: We adopt [Stable Diffusion 2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1-base) as our diffusion model, you can download and load the pretrained weight by setting `--pretrained_diffusion_path="stabilityai/stable-diffusion-2-1-base"` in [main.py](main.py). You can download them from here.
     > **Note:** We observe that the official Stable Diffusion 2.1 link appears unavailable. However, the automatic download via the Hugging Face API remains functional, or you can download the weights from other platforms for local deployment.
   - MLLMs: We utilize several Multimodal Large Language Models for prompt generation:
     - [BLIP-2](https://huggingface.co/Salesforce/blip2-opt-2.7b)
     - [InternVL2](https://huggingface.co/OpenGVLab/InternVL2-8B) 
     - [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
     - [LLaVA-1.6](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf)
## Usage
1. Context-Aware Prompts Generation
   - We provide the script to generate context-aware prompts using [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct). To generate prompts for your dataset, run:
     ```bash
     python qwen_prompt.py
     ```
     > **Note:** To use other MLLMs, please refer to the specific script or modify the model configuration.
2. Crafting Unrestricted Adversarial Examples (UAEs)
   - Run the following command to generate UAEs using our method:  
     ```bash
     python main.py
     ```
3. Evaluation
   - We here provide unrestricted adversarial examples crafted for Res-50 using our method and its enhanced version. We store them in .[/output](output). Simply run eval_attack.py to perform attacks against the official PyTorch ResNet50 model. You can modify the attack parameter at the [eval_attack.py](eval_attack.py).  
   - Attack Success Rate (ASR): To evaluate the attack performance against the official PyTorch ResNet50 model:
     ```bash
     python eval_attack.py
     ```
   - Image Quality Assessment (IQA): To evaluate SSIM, PSNR, and LPIPS metrics:
     ```bash
     python eval_iqa.py 
     ```
## Results
<div>
  <img src="fig/Visual1.png" width="90%" alt="Visual">
</div>

<div>
  <img src="fig/1.png" width="90%" alt="Visual">
</div>

<div>
  <img src="fig/2.png" width="90%" alt="Visual">
</div>

<div>
  <img src="fig/3.png" width="90%" alt="Visual">
</div>

<div>
  <img src="fig/4.png" width="90%" alt="Visual">
</div>

<div>
  <img src="fig/5.png" width="90%" alt="Visual">
</div>

## Acknowledgements
Our code is based on [DiffAttack](https://github.com/WindVChen/DiffAttack) and [Prompt-to-Prompt](https://github.com/google/prompt-to-prompt) repository. We thank the authors for releasing their code.