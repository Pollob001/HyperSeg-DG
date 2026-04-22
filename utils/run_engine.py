import os
import random
import time
import datetime
import numpy as np
import albumentations as A
import cv2
from glob import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils.utils import seeding, create_dir, print_and_save, shuffling, epoch_time, calculate_metrics, mask_to_bbox
from utils.metrics import calculate_metric_percase
from tqdm import tqdm


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch.nn.functional as F



def load_names(path, file_path):
    f = open(file_path, "r")
    data = f.read().split("\n")[:-1]
    
    images = [os.path.join(path, "images", name) + ext for name in data for ext in [".jpg", ".png"] if os.path.exists(os.path.join(path, "images", name) + ext)]
    masks = [os.path.join(path, "masks", name) + ext for name in data for ext in [".png", ".jpg"] if os.path.exists(os.path.join(path, "masks", name) + ext)]
    
    return images, masks


# def load_names(path, file_path):
#     f = open(file_path, "r")
#     data = f.read().split("\n")[:-1]
#     images = [os.path.join(path, "images", name) + ".jpg" for name in data]
#     masks = [os.path.join(path, "masks", name) + ".png" for name in data]
#     # images = [os.path.join(path, "images", name) + ".png" for name in data]
#     # masks = [os.path.join(path, "masks", name) + ".png" for name in data]
#     return images, masks


def load_data(path,val_name=None):
    train_names_path = f"{path}/train.txt"
    # valid_names_path = f"{path}/val.txt"
    if val_name is None:
        valid_names_path = f"{path}/val.txt"
    else:
        valid_names_path = f"{path}/val_{val_name}.txt"

    train_x, train_y = load_names(path, train_names_path)
    valid_x, valid_y = load_names(path, valid_names_path)

    return (train_x, train_y), (valid_x, valid_y)

class DATASET(Dataset):
    def __init__(self, images_path, masks_path, size, transform=None):
        super().__init__()

        self.images_path = images_path
        self.masks_path = masks_path
        self.transform = transform
        self.n_samples = len(images_path)
        self.size = size

    def __getitem__(self, index):
        """ Reading Image & Mask """
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        background = mask.copy()
        background = 255 - background

        """ Applying Data Augmentation """
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask, background=background)
            image = augmentations["image"]
            mask = augmentations["mask"]
            background = augmentations["background"]

        """ Image """
        image = cv2.resize(image, self.size)
        image = np.transpose(image, (2, 0, 1))
        image = image / 255.0

        """ Mask """
        mask = cv2.resize(mask, self.size)
        mask = np.expand_dims(mask, axis=0)
        mask = mask / 255.0

        """ Background """
        background = cv2.resize(background, self.size)
        background = np.expand_dims(background, axis=0)
        background = background / 255.0

        return image, (mask, background)

    def __len__(self):
        return self.n_samples


def complementary_loss(prob_fg, prob_bg, prob_uc):
    loss = (prob_fg * prob_bg).sum() + (prob_fg * prob_uc).sum() + (prob_bg * prob_uc).sum()
    num_pixels = prob_fg.size(0) * prob_fg.size(2) * prob_fg.size(3)  # B * H * W
    normalized_loss = loss / num_pixels
    return normalized_loss

def train(model, loader, optimizer, loss_fn, device):
    model.train()

    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0

    for i, ((x), (y1, y2)) in enumerate(tqdm(loader, desc="Training", total=len(loader))):
        x = x.to(device, dtype=torch.float32)
        y1 = y1.to(device, dtype=torch.float32)
        y2 = y2.to(device, dtype=torch.float32)

        optimizer.zero_grad()

        mask_pred, fg_pred, bg_pred, uc_pred = model(x)

        loss_mask = loss_fn(mask_pred, y1)
        loss_fg = loss_fn(fg_pred, y1)
        loss_bg = loss_fn(bg_pred, y2)

        beta1 = 1 / (torch.tanh(fg_pred.sum() / (fg_pred.shape[2] * fg_pred.shape[3])) + 1e-15)
        beta2 = 1 / (torch.tanh(bg_pred.sum() / (bg_pred.shape[2] * bg_pred.shape[3])) + 1e-15)
        beta1 = beta1.to(device)
        beta2 = beta2.to(device)
        preds = torch.stack([fg_pred, bg_pred, uc_pred], dim=1)
        probs = F.softmax(preds, dim=1)
        prob_fg, prob_bg, prob_uc = probs[:, 0], probs[:, 1], probs[:, 2]

        loss_comp = complementary_loss(prob_fg, prob_bg, prob_uc)
        loss_comp = loss_comp.to(device)
        loss = loss_mask + beta1 * loss_fg + beta2 * loss_bg + loss_comp
        loss.backward()

        optimizer.step()
        epoch_loss += loss.item()

        """ Calculate the metrics """
        batch_jac = []
        batch_f1 = []
        batch_recall = []
        batch_precision = []

        for yt, yp in zip(y1, mask_pred):
            score = calculate_metrics(yt, yp)
            batch_jac.append(score[0])
            batch_f1.append(score[1])
            batch_recall.append(score[2])
            batch_precision.append(score[3])

        epoch_jac += np.mean(batch_jac)
        epoch_f1 += np.mean(batch_f1)
        epoch_recall += np.mean(batch_recall)
        epoch_precision += np.mean(batch_precision)

    epoch_loss = epoch_loss / len(loader)
    epoch_jac = epoch_jac / len(loader)
    epoch_f1 = epoch_f1 / len(loader)
    epoch_recall = epoch_recall / len(loader)
    epoch_precision = epoch_precision / len(loader)

    return epoch_loss, [epoch_jac, epoch_f1, epoch_recall, epoch_precision]


def evaluate(model, loader, loss_fn, device):
    model.eval()

    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0
    epoch_asd = 0.0
    asd_count = 0

    with torch.no_grad():
        for i, ((x), (y1, y2)) in enumerate(tqdm(loader, desc="Evaluation", total=len(loader))):
            x = x.to(device, dtype=torch.float32)
            y1 = y1.to(device, dtype=torch.float32)
            y2 = y2.to(device, dtype=torch.float32)

            mask_pred, fg_pred, bg_pred, uc_pred = model(x)

            loss_mask = loss_fn(mask_pred, y1)
            loss_fg = loss_fn(fg_pred, y1)
            loss_bg = loss_fn(bg_pred, y2)

            beta1 = 1 / (torch.tanh(fg_pred.sum() / (fg_pred.shape[2] * fg_pred.shape[3])) + 1e-15)
            beta2 = 1 / (torch.tanh(bg_pred.sum() / (bg_pred.shape[2] * bg_pred.shape[3])) + 1e-15)
            beta1 = beta1.to(device)
            beta2 = beta2.to(device)

            preds = torch.stack([fg_pred, bg_pred, uc_pred], dim=1)
            probs = F.softmax(preds, dim=1)
            prob_fg, prob_bg, prob_uc = probs[:, 0], probs[:, 1], probs[:, 2]

            loss_comp = complementary_loss(prob_fg, prob_bg, prob_uc)
            loss_comp = loss_comp.to(device)

            loss = loss_mask + beta1 * loss_fg + beta2 * loss_bg + loss_comp

            epoch_loss += loss.item()

            """ Calculate the metrics """
            batch_jac = []
            batch_f1 = []
            batch_recall = []
            batch_precision = []
            batch_asd = []

            for yt, yp in zip(y1, mask_pred):
                score = calculate_metrics(yt, yp)
                batch_jac.append(score[0])
                batch_f1.append(score[1])
                batch_recall.append(score[2])
                batch_precision.append(score[3])
                try:
                    yt_bin = (yt.detach().cpu().numpy().squeeze() > 0.5).astype(np.uint8)
                    yp_bin = (yp.detach().cpu().numpy().squeeze() > 0.5).astype(np.uint8)
                    _, _, _, asd_score = calculate_metric_percase(yp_bin, yt_bin)
                    batch_asd.append(float(asd_score))
                except Exception:
                    # ASD can fail on degenerate masks; skip those samples.
                    pass

            epoch_jac += np.mean(batch_jac)
            epoch_f1 += np.mean(batch_f1)
            epoch_recall += np.mean(batch_recall)
            epoch_precision += np.mean(batch_precision)
            if len(batch_asd) > 0:
                epoch_asd += np.mean(batch_asd)
                asd_count += 1

        epoch_loss = epoch_loss / len(loader)
        epoch_jac = epoch_jac / len(loader)
        epoch_f1 = epoch_f1 / len(loader)
        epoch_recall = epoch_recall / len(loader)
        epoch_precision = epoch_precision / len(loader)
        epoch_dsc = epoch_f1
        epoch_asd = epoch_asd / asd_count if asd_count > 0 else 0.0

        return epoch_loss, [epoch_jac, epoch_f1, epoch_recall, epoch_precision, epoch_dsc, epoch_asd]
