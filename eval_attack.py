import timm
import torch
from torchvision import models, transforms
from PIL import Image
import os
import warnings
from Finegrained_model import model as otherModel

warnings.filterwarnings("ignore")

def load_model(name):
    if name == "resnet":
        model = models.resnet50(pretrained=True)
    elif name == "vgg":
        model = models.vgg19(pretrained=True)
    elif name == "mobile":
        model = models.mobilenet_v2(pretrained=True)
    elif name == "densenet":
        model = models.densenet121(pretrained=True)
    elif name == "vit-b":
        model = models.vit_b_16(pretrained=True)
    elif name == "swin-b":
        model = models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1)
    elif name == "deit_base_patch16_224":
        model = timm.create_model('deit_base_patch16_224', pretrained=True)
    elif name == "cait_s24_224":
        model = timm.create_model('cait_s24_224', pretrained=True)
    elif name == 'cubResnet50':
        model = otherModel.CUB()[0]
    elif name == 'cubSEResnet154':
        model = otherModel.CUB()[1]
    elif name == 'cubSEResnet101':
        model = otherModel.CUB()[2]
    elif name == 'carResnet50':
        model = otherModel.CAR()[0]
    elif name == 'carSEResnet154':
        model = otherModel.CAR()[1]
    elif name == 'carSEResnet101':
        model = otherModel.CAR()[2]
    else:
        raise ValueError(f"Model {name} not recognized.")
    model.eval()
    return model


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_labels(label_file):
    with open(label_file, "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

def load_batch_images(image_paths):
    images = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        img = transform(img)
        images.append(img)
    return torch.stack(images)

def evaluate_attack(model, adv_images_folder, labels, num_images=1000, batch_size=200):

    attack_success_count = 0
    num_batches = num_images // batch_size

    for batch_idx in range(num_batches):
        adv_image_paths = [os.path.join(adv_images_folder, f"{i:04d}_adv_image.png")
                           for i in range(batch_idx * batch_size, (batch_idx + 1) * batch_size)]

        adv_imgs = load_batch_images(adv_image_paths)

        orig_labels = [int(labels[i]) for i in range(batch_idx * batch_size, (batch_idx + 1) * batch_size)]

        with torch.no_grad():
            adv_preds = model(adv_imgs).argmax(dim=1) + 1

        for orig_label, adv_pred in zip(orig_labels, adv_preds):
            if orig_label != adv_pred.item():
                attack_success_count += 1

        print(f"Processed batch {batch_idx+1}/{num_batches}...")

    return attack_success_count

def main():
    adv_images_folder = "output/PDAttack_resnet"
    label_file = "dataset/labels.txt"

    model_names = [
        "resnet", "vgg", "mobile", "densenet", "vit-b", "swin-b", "deit_base_patch16_224", "cait_s24_224",
        # "cubResnet50", "cubSEResnet154", "cubSEResnet101",
        # "carResnet50", "carSEResnet154", "carSEResnet101",
    ]
    labels = load_labels(label_file)

    with open("log.txt", "w") as log_file:
        log_file.write(" ".join(model_names) + "\n")

        success_rates = []
        for model_name in model_names:
            print(f"Evaluating attack success rate for {model_name}...")
            model = load_model(model_name)
            success_rate = evaluate_attack(model, adv_images_folder, labels, batch_size=200)

            success_rates.append(f"{success_rate*0.1}")
            print("success_rates:",success_rates)
        log_file.write(" ".join(success_rates) + "\n")

if __name__ == "__main__":
    main()
