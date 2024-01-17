from engine import trainer
from utils import plot_results
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
LINEAR_PROBING = config["LINEAR_PROBING"]
PROBING_EPOCHS = int(config["PROBING_EPOCHS"])
LOSS = config["LOSS"]

IMAGE_SIZE = int(config["IMAGE_SIZE"])
MODEL = config["MODEL"]
PRETRAINED = config["PRETRAINED"]

DATASET = config["DATASET"]

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

    #run id is date and time of the run
    run_id = time.strftime("%Y-%m-%d_%H-%M-%S")

    #create folder for this run in runs folder
    os.mkdir("./runs/" + run_id)

    save_dir = "./runs/" + run_id
    

    #load data
    transforms_train = v2.Compose([
        v2.ToImage(),
        v2.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE), scale=(0.5, 1.0), antialias=True),

        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=(-60, 60)),
        v2.RandomAffine(degrees=(-15, 15), translate=(0.25, 0.25), scale=(0.7, 1.3), shear=(-15, 15, -15, 15)),
        # v2.RandomPerspective(distortion_scale=0.1, p=0.2),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        v2.RandomAutocontrast(p=0.2),
        # v2.RandomEqualize(p=0.2),

        v2.ToDtype(torch.float, scale=True),
        v2.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
    ])
    # transforms_train = v2.Compose([
    #     v2.ToImage(),
    #     v2.ToDtype(torch.float32, scale=True),
    #     v2.Resize((224, 224), antialias=True),
    #     v2.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
    # ])

    transforms_test = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((224, 224), antialias=True),
        v2.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
    ])

    dataset_name = ["CUB", "CUB and FGVC-Aircraft", "FoodX"][DATASET]
    num_classes = [200, 200 + 100, 251][DATASET]

    cutmix = v2.CutMix(num_classes=num_classes)
    mixup = v2.MixUp(num_classes=num_classes)
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

    def collate_fn(batch):
        return cutmix_or_mixup(*default_collate(batch))

    if dataset_name == 'CUB':
        dataset_path = "/apps/local/shared/CV703/datasets/CUB/CUB_200_2011"

        train_dataset = CUBDataset(image_root_path=dataset_path, transform=transforms_train, split="train")
        test_dataset = CUBDataset(image_root_path=dataset_path, transform=transforms_test, split="test")

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

        class_names = train_dataset.classes

    elif dataset_name == 'CUB and FGVC-Aircraft':
        dataset_path_cub = "/apps/local/shared/CV703/datasets/CUB/CUB_200_2011"
        train_dataset_cub = CUBDataset(image_root_path=dataset_path_cub, transform=transforms_train, split="train")
        test_dataset_cub = CUBDataset(image_root_path=dataset_path_cub, transform=transforms_test, split="test")

        dataset_path_aircraft = "/apps/local/shared/CV703/datasets/fgvc-aircraft-2013b"
        train_dataset_aircraft = FGVCAircraft(root=dataset_path_aircraft, transform=transforms_train, train=True)
        test_dataset_aircraft = FGVCAircraft(root=dataset_path_aircraft, transform=transforms_test, train=False)

        concat_dataset_train = ConcatDataset([train_dataset_cub, train_dataset_aircraft])
        concat_dataset_test = ConcatDataset([test_dataset_cub, test_dataset_aircraft])

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
        dataset_path = "/apps/local/shared/CV703/datasets/FoodX/food_dataset"

        train_dataset = FOODDataset(data_dir=dataset_path, transform=transforms_train, split="train")
        test_dataset = FOODDataset(data_dir=dataset_path, transform=transforms_test, split="test")

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)


    #load model
    model = get_model(MODEL, PRETRAINED, num_classes)

    model.to(DEVICE)
    torch.compile(model)
    
    #load optimizer
    if LOSS == "CrossEntropyLoss":
        loss = torch.nn.CrossEntropyLoss()
    else:
        raise Exception("Loss not implemented")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    if LEARNING_SCHEDULER == "CosineAnnealingLR":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    else:
        lr_scheduler = None

    if LINEAR_PROBING:
        linear_probing_epochs = PROBING_EPOCHS
    else:
        linear_probing_epochs = None
     
    #train model
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
        linear_probing_epochs=linear_probing_epochs
    )

    train_summary = {
        "config": config,
        "results": results,
    }

    with open(save_dir + "/train_summary.json", "w") as f:
        json.dump(train_summary, f, indent=4)

    plot_results(results["train_loss"], results["val_loss"], "Loss", save_dir)
    plot_results(results["train_accucary"], results["val_accuracy"], "Accuracy", save_dir)

if __name__ == "__main__":
    main()


