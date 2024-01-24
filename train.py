from engine import trainer
from utils import plot_results, LinearLR, WarmupLR
from models.model import get_model
from torch.utils.data import ConcatDataset 
from dataset import CUBDataset, FGVCAircraft, FOODDataset

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import torchvision
from torch.utils.data import default_collate
torchvision.disable_beta_transforms_warning()


import matplotlib.pyplot as plt
import numpy as np
import random

import yaml
import json
import time
import os

config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)


LEARNING_RATE = float(config["LEARNING_RATE"])
LEARNING_SCHEDULER = config["LEARNING_SCHEDULER"]
BATCH_SIZE = int(config["BATCH_SIZE"])
NUM_EPOCHS = int(config["NUM_EPOCHS"])

FINETUNE = config["FINETUNE"]
FINETUNE_EPOCHS = int(config["FINETUNE_EPOCHS"])
FINETUNE_LR = float(config["FINETUNE_LR"])
WARMUP_EPOCHS = int(config["WARMUP_EPOCHS"])
WARMUP_LR = float(config["WARMUP_LR"])

LOSS = config["LOSS"]
LABEL_SMOOTHING = float(config["LABEL_SMOOTHING"])

IMAGE_SIZE = int(config["IMAGE_SIZE"])
MODEL = config["MODEL"]
PRETRAINED = config["PRETRAINED"]
FREEZE = config["FREEZE"]

DATASET = config["DATASET"]
CUT_UP_MIX = config["CUT_UP_MIX"]
HPC = config["HPC"]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {DEVICE} device")


def START_seed():
    seed = 9
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed) 


def main():
    START_seed()

    dataset_name = ["CUB", "CUB and FGVC-Aircraft", "FoodX"][DATASET]
    num_classes = [200, 200 + 100, 251][DATASET]

    #run id is date and time of the run
    run_id = time.strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = "./runs/" + dataset_name + '/' + MODEL + '/' + run_id
    os.makedirs(save_dir, exist_ok=True)
    

    #load data
    transforms_train = v2.Compose([
        v2.ToImage(),
        v2.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE), scale=(0.7, 1.0), antialias=True),

        # v2.AutoAugment(policy=v2.AutoAugmentPolicy.IMAGENET, ),
        v2.RandAugment(num_ops=2, magnitude=10),
        v2.RandomErasing(p=0.1),

        v2.ToDtype(torch.float, scale=True),
        v2.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
    ])

    transforms_test = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((224, 224), antialias=True),
        v2.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
    ])


    cutmix = v2.CutMix(num_classes=num_classes, alpha=1.0)
    mixup = v2.MixUp(num_classes=num_classes, alpha=0.2)
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

    if CUT_UP_MIX:
        def collate_fn(batch):
            return cutmix_or_mixup(*default_collate(batch))
    else:
        collate_fn = None

    if HPC:
        dataset_path_prefix = "/apps/local/shared/CV703/datasets/"
    else:
        dataset_path_prefix = "datasets/"

    if dataset_name == 'CUB':
        dataset_path = dataset_path_prefix + "CUB/CUB_200_2011"

        train_simple_dataset = CUBDataset(image_root_path=dataset_path, transform=transforms_test, split="train")
        train_dataset = CUBDataset(image_root_path=dataset_path, transform=transforms_train, split="train")
        test_dataset = CUBDataset(image_root_path=dataset_path, transform=transforms_test, split="test")

        train_simple_loader = DataLoader(train_simple_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

        class_names = train_dataset.classes

    elif dataset_name == 'CUB and FGVC-Aircraft':

        dataset_path_cub = dataset_path_prefix + "CUB/CUB_200_2011"
        train_simple_dataset_cub = CUBDataset(image_root_path=dataset_path_cub, transform=transforms_test, split="train")
        train_dataset_cub = CUBDataset(image_root_path=dataset_path_cub, transform=transforms_train, split="train")
        test_dataset_cub = CUBDataset(image_root_path=dataset_path_cub, transform=transforms_test, split="test")

        dataset_path_aircraft = dataset_path_prefix + "fgvc-aircraft-2013b"
        train_simple_dataset_aircraft = FGVCAircraft(root=dataset_path_aircraft, transform=transforms_test, train=True)
        train_dataset_aircraft = FGVCAircraft(root=dataset_path_aircraft, transform=transforms_train, train=True)
        test_dataset_aircraft = FGVCAircraft(root=dataset_path_aircraft, transform=transforms_test, train=False)

        concat_dataset_train_simple = ConcatDataset([train_simple_dataset_cub, train_simple_dataset_aircraft])
        concat_dataset_train = ConcatDataset([train_dataset_cub, train_dataset_aircraft])
        concat_dataset_test = ConcatDataset([test_dataset_cub, test_dataset_aircraft])

        train_simple_loader = torch.utils.data.DataLoader(
                    concat_dataset_train_simple,
                    batch_size=BATCH_SIZE, shuffle=True,
                    num_workers=8, pin_memory=True,
                    collate_fn=None
                    )
        train_loader = torch.utils.data.DataLoader(
                    concat_dataset_train,
                    batch_size=BATCH_SIZE, shuffle=True,
                    num_workers=8, pin_memory=True,
                    collate_fn=collate_fn
                    )
        test_loader = torch.utils.data.DataLoader(
                    concat_dataset_test,
                    batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=8, pin_memory=True
                    )
        
        classes_1 = concat_dataset_train.datasets[0].classes
        classes_2 = concat_dataset_train.datasets[1].classes

        class_names = [*classes_1, *classes_2]

    elif dataset_name == 'FoodX':
        dataset_path = dataset_path_prefix + "FoodX/food_dataset"

        train_dataset = FOODDataset(data_dir=dataset_path, transform=transforms_train, split="train")
        test_dataset = FOODDataset(data_dir=dataset_path, transform=transforms_test, split="val")

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)


    #load model
    model = get_model(MODEL, PRETRAINED, num_classes, FREEZE)

    model.to(DEVICE)
    torch.compile(model)
    
    #load optimizer
    if LOSS == "CrossEntropyLoss":
        loss = torch.nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    else:
        raise Exception("Loss not implemented")
    
    # warmup epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=WARMUP_LR)
    # lr_scheduler = LinearLR(optimizer, LEARNING_RATE, WARMUP_EPOCHS)
    lr_scheduler = WarmupLR(optimizer, WARMUP_EPOCHS, WARMUP_LR, LEARNING_RATE)
    lr_scheduler.step()

    results = trainer(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        loss_fn=loss,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=DEVICE,
        epochs=WARMUP_EPOCHS,
        save_dir=save_dir,
    )

    train_summary = {
        "config": config,
        "results": results,
    }


    for param_group in optimizer.param_groups:
        param_group["lr"] = LEARNING_RATE

    if LEARNING_SCHEDULER == "CosineAnnealingLR":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=FINETUNE_LR)
    else:
        lr_scheduler = None
    
    # regualr epochs
    unfreeze_order = [5, 4, 3, 2, 1, 0]

    results = trainer(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        loss_fn=loss,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=DEVICE,
        epochs=NUM_EPOCHS,
        save_dir=save_dir,
        unfreeze=unfreeze_order,
    )

    train_summary["results"]["train_loss"] += results["train_loss"]
    train_summary["results"]["train_accuracy"] += results["train_accuracy"]
    train_summary["results"]["val_loss"] += results["val_loss"]
    train_summary["results"]["val_accuracy"] += results["val_accuracy"]
    train_summary["results"]["learning_rate"] += results["learning_rate"]

    if FINETUNE:
        for param in model.parameters():
            param.requires_grad = True

        for param_group in optimizer.param_groups:
            param_group["lr"] = FINETUNE_LR
        
        # if LEARNING_SCHEDULER == "CosineAnnealingLR":
        #     lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINETUNE_EPOCHS)
        # else:
        #     lr_scheduler = None
        lr_scheduler = None

        results = trainer(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            loss_fn=loss,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            device=DEVICE,
            epochs=FINETUNE_EPOCHS,
            save_dir=save_dir,
        )

        train_summary["results"]["train_loss"] += results["train_loss"]
        train_summary["results"]["train_accuracy"] += results["train_accuracy"]
        train_summary["results"]["val_loss"] += results["val_loss"]
        train_summary["results"]["val_accuracy"] += results["val_accuracy"]
        train_summary["results"]["learning_rate"] += results["learning_rate"]

    with open(save_dir + "/train_summary.json", "w") as f:
        json.dump(train_summary, f, indent=4)

    plot_results(train_summary["results"]["train_loss"], train_summary["results"]["val_loss"], "Loss", save_dir)
    plot_results(train_summary["results"]["train_accuracy"], train_summary["results"]["val_accuracy"], "Accuracy", save_dir)

if __name__ == "__main__":
    main()


