import os
import random
import time
import datetime
import numpy as np
import albumentations as A
import torch
from torch.utils.data import DataLoader

from utils.utils import print_and_save, shuffling, epoch_time
from network.model_stage1 import HyperSegStage1
from utils.metrics import DiceBCELoss
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

from utils.run_engine_stage1 import load_data, train, evaluate, DATASET
def my_seeding(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    model_name = "HyperSeg_DG_stage1"

    dataset_name = 'BUSI'

    val_name = None

    seed = random.randint(0, 10000)

    my_seeding(seed)

    image_size = 256
    size = (image_size, image_size)
    batch_size = 8
    num_epochs = 200
    lr = 1e-4
    early_stopping_patience = 100

    backbone_pretrained_path = None
    resume_path = None


    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    folder_name = f"{model_name}_{dataset_name}_{val_name}_lr{lr}_{current_time}"

    # Use full path for the base directory and data path
    base_dir = "/gdata/aynulislam/SEGDATASET"
    data_path = os.path.join(base_dir, dataset_name)
    
    save_dir = os.path.join("/ghome/aynulislam/HyperSeg_DG/runfile", dataset_name, folder_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_log_path = os.path.join(save_dir, "train_log.txt")
    checkpoint_path = os.path.join(save_dir, "checkpoint.pth")

    train_log = open(train_log_path, "w")
    train_log.write("\n")
    train_log.close()

    datetime_object = str(datetime.datetime.now())
    print_and_save(train_log_path, datetime_object)
    print("")

    hyperparameters_str = f"Image Size: {image_size}\nBatch Size: {batch_size}\nLR: {lr}\nEpochs: {num_epochs}\n"
    hyperparameters_str += f"Early Stopping Patience: {early_stopping_patience}\n"
    hyperparameters_str += f"Seed: {seed}\n"
    print_and_save(train_log_path, hyperparameters_str)

    """ Data augmentation: Transforms """
    transform = A.Compose([
        A.Rotate(limit=90, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.CoarseDropout(p=0.3, max_holes=10, max_height=32, max_width=32)
    ])

    """ Dataset """
    (train_x, train_y), (valid_x, valid_y) = load_data(data_path, val_name)
    train_x, train_y = shuffling(train_x, train_y)
    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    print_and_save(train_log_path, data_str)

    """ Dataset and loader """
    train_dataset = DATASET(train_x, train_y, (image_size, image_size), transform=transform)
    valid_dataset = DATASET(valid_x, valid_y, (image_size, image_size), transform=None)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8
    )

    """ Model """
    device = torch.device('cuda')
    model = HyperSegStage1(backbone_pretrained_path=backbone_pretrained_path)

    if resume_path:
        checkpoint = torch.load(resume_path, map_location='cpu')
        model.load_state_dict(checkpoint)

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=30, verbose=True)
    loss_fn = DiceBCELoss()
    loss_name = "BCE Dice Loss"
    data_str = f"Optimizer: Adam\nLoss: {loss_name}\n"
    print_and_save(train_log_path, data_str)


    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    data_str = f"Number of parameters: {num_params/1000000}M\n"
    print_and_save(train_log_path, data_str)


    """ Training the model """
    best_valid_metrics = 0.0
    early_stopping_count = 0

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, train_metrics = train(model, train_loader, optimizer, loss_fn, device)
        valid_loss, valid_metrics = evaluate(model, valid_loader, loss_fn, device)
        scheduler.step(valid_loss)

        if valid_metrics[0] > best_valid_metrics:
            data_str = f"Valid mIoU improved from {best_valid_metrics:2.4f} to {valid_metrics[0]:2.4f}. Saving checkpoint: {checkpoint_path}"
            print_and_save(train_log_path, data_str)

            best_valid_metrics = valid_metrics[0]
            torch.save(model.state_dict(), checkpoint_path)
            early_stopping_count = 0


        elif valid_metrics[0] < best_valid_metrics:
            early_stopping_count += 1

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n"
        data_str += f"\tTrain Loss: {train_loss:.4f} - mIoU: {train_metrics[0]:.4f} - F1: {train_metrics[1]:.4f} - Recall: {train_metrics[2]:.4f} - Precision: {train_metrics[3]:.4f}\n"
        data_str += f"\t Val. Loss: {valid_loss:.4f} - mIoU: {valid_metrics[0]:.4f} - F1: {valid_metrics[1]:.4f} - Recall: {valid_metrics[2]:.4f} - Precision: {valid_metrics[3]:.4f}\n"
        print_and_save(train_log_path, data_str)

        if early_stopping_count == early_stopping_patience:
            data_str = f"Early stopping: validation loss stops improving from last {early_stopping_patience} continously.\n"
            print_and_save(train_log_path, data_str)
            break
