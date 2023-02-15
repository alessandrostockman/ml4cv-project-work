import torch
from torchvision import models
from torchvision import transforms as T


def freeze_backbone(model):
    for param in model.parameters():
        param.requires_grad = False


def create_accessories(model, train_params):
    if train_params["loss"] == "cross_entropy":
        criterion = torch.nn.CrossEntropyLoss(
            label_smoothing=train_params["label_smoothing"]
        )

    if train_params["optimizer"] == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=train_params["learning_rate"],
            weight_decay=train_params["weight_decay"],
        )
    elif train_params["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_params["learning_rate"],
            weight_decay=train_params["weight_decay"],
        )

    if train_params["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, train_params["epochs"]
        )
    elif train_params["scheduler"] == None:
        scheduler = None

    return criterion, optimizer, scheduler  # , transforms


def get_model(model_name, num_classes, fine_tune=True):
    model_ft = None

    if model_name == "resnet":
        model_ft = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        if not fine_tune:
            freeze_backbone(model_ft)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = torch.nn.Linear(num_ftrs, num_classes)
    elif model_name == "convnext":
        model_ft = models.convnext_tiny(
            weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        )
        if not fine_tune:
            freeze_backbone(model_ft)

        num_ftrs = model_ft.classifier[2].in_features
        model_ft.classifier[2] = torch.nn.Linear(num_ftrs, num_classes)
    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft


def load_model(model_name, model_path, num_classes):
    model = get_model(
        "resnet" if model_name == "baseline" else model_name,
        num_classes,
        fine_tune=False,
    )
    model.load_state_dict(torch.load(model_path))
    return model
