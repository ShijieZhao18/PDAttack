import os
import torch
import json
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from tqdm import tqdm

LOCAL_MODEL_PATH = "Qwen3-VL-8B-Instruct"
image_dir = "dataset/images"
label_file = "dataset/labels.txt"
imagenet_json_path = "dataset/imagenet_class_index.json"
output_base_dir = "Text"

os.makedirs(output_base_dir, exist_ok=True)

def load_imagenet_classes():
    if not os.path.exists(imagenet_json_path):
        raise FileNotFoundError(f"{imagenet_json_path}")
    with open(imagenet_json_path, "r", encoding='utf-8') as f:
        class_idx = json.load(f)
    return {int(k): v[1].replace('_', ' ') for k, v in class_idx.items()}

try:
    imagenet_map = load_imagenet_classes()
except Exception as e:
    print(e)
    exit()

try:
    with open(label_file, 'r', encoding='utf-8') as f:
        raw_labels = [int(line.strip()) for line in f.readlines() if line.strip().isdigit()]
except FileNotFoundError:
    print(f"{label_file} not found")
    exit()

model = Qwen3VLForConditionalGeneration.from_pretrained(
    LOCAL_MODEL_PATH,
    dtype="auto",
    device_map="auto"
)

processor = AutoProcessor.from_pretrained(LOCAL_MODEL_PATH)

try:
    image_files = sorted(
        [f for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg')],
        key=lambda x: int(os.path.splitext(x)[0])
    )
except ValueError:
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg')])

buckets_stages = [
    {
        "name": "00-10_tokens",
        "instruction": "Describe the {class_name} in the image. Be extremely concise. Limit the response to 10 words.",
        "max_new": 20
    },
    {
        "name": "10-20_tokens",
        "instruction": "Based on this description: '{prev}', rewrite and expand it to describe the main action or placement of the object. The total length must be between 10 and 20 words.",
        "max_new": 30
    },
    {
        "name": "20-30_tokens",
        "instruction": "Based on this description: '{prev}', rewrite and expand it to include details about the object's color and physical features. The total length must be between 20 and 30 words.",
        "max_new": 45
    },
    {
        "name": "30-40_tokens",
        "instruction": "Based on this description: '{prev}', rewrite and expand it to describe specific parts (like wheels, legs, or surface). The total length must be between 30 and 40 words.",
        "max_new": 60
    },
    {
        "name": "40-50_tokens",
        "instruction": "Based on this description: '{prev}', rewrite and expand it to elaborate on the background environment and lighting. The total length must be between 40 and 50 words.",
        "max_new": 75
    },
    {
        "name": "50-75_tokens",
        "instruction": "Based on this description: '{prev}', rewrite and create a comprehensive visual description covering the object, details, background, and atmosphere. The total length must be between 50 and 75 words.",
        "max_new": 100
    }
]

file_handles = {}
for stage in buckets_stages:
    f_path = os.path.join(output_base_dir, f"{stage['name']}.txt")
    file_handles[stage['name']] = open(f_path, 'w', encoding='utf-8')

for idx, img_name in enumerate(tqdm(image_files, desc="Processing Images", unit="img")):
    img_path = os.path.join(image_dir, img_name)

    class_name = "object"
    if idx < len(raw_labels):
        label_id = raw_labels[idx] - 1
        class_name = imagenet_map.get(label_id, "object")
    
    print(f"\n[Processing {img_name}] Class: {class_name}")

    try:
        image = Image.open(img_path).convert('RGB')
        prev_output = ""

        for i, stage in enumerate(buckets_stages):
            if i == 0:
                user_content = stage["instruction"].format(class_name=class_name)
            else:
                user_content = stage["instruction"].format(prev=prev_output)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": user_content},
                    ],
                }
            ]

            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = inputs.to(model.device)

            generated_ids = model.generate(
                **inputs,
                max_new_tokens=stage["max_new"],
                do_sample=False,
                repetition_penalty=1.1
            )

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()

            clean_desc = " ".join(output_text.split())
            prev_output = clean_desc

            file_handles[stage['name']].write(f"{clean_desc}\n")
            
            print(f"  > {stage['name']}: {clean_desc}")
            file_handles[stage['name']].flush()

    except Exception as e:
        print(f"!!! Error processing {img_name}: {e}")
        for fh in file_handles.values():
            fh.write(f"ERROR\n")
            fh.flush()

for fh in file_handles.values():
    fh.close()

print("Done.")