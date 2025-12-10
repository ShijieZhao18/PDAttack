import torch.nn as nn
import torchvision.models as models
import numpy as np
import os

from art.estimators.classification import PyTorchClassifier
import timm
import warnings

warnings.filterwarnings("ignore")

def model_selection(name):
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
    else:
        raise NotImplementedError("No such model!")
    return model.cuda()


def model_transfer(clean_img, adv_img, label, res, save_path=r"C:\Users\PC\Desktop\output", fid_path=None, args=None):
    log = open(os.path.join(save_path, "log.txt"), mode="w", encoding="utf-8")

    if args.dataset_name == "imagenet_compatible":
        models_transfer_name = ["resnet"]
        nb_classes = 1000
    else:
        raise NotImplementedError

    all_clean_accuracy = []
    all_adv_accuracy = []
    for name in models_transfer_name:
        model = model_selection(name)
        model.eval()
        f_model = PyTorchClassifier(
            model=model,
            clip_values=(0, 1),
            loss=nn.CrossEntropyLoss(),
            input_shape=(3, res, res),
            nb_classes=nb_classes,
            preprocessing=(np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5])) if "adv" in name else (
                np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])),
            device_type='gpu',
        )

        clean_pred = f_model.predict(clean_img, batch_size=50)

        accuracy = np.sum((np.argmax(clean_pred, axis=1) - 1) == label) / len(label) if "adv" in name else np.sum(
            np.argmax(clean_pred, axis=1) == label) / len(label)
        all_clean_accuracy.append(accuracy * 100)

        adv_pred = f_model.predict(adv_img, batch_size=50)
        accuracy = np.sum((np.argmax(adv_pred, axis=1) - 1) == label) / len(label) if "adv" in name else np.sum(
            np.argmax(adv_pred, axis=1) == label) / len(label)
        all_adv_accuracy.append(accuracy * 100)

    log.close()