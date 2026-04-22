import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import torch.nn.functional as F

from utils.utils import calculate_metrics
from utils.metrics import calculate_metric_percase


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


def load_names(file_path):
    images = []
    masks = []
    # Infer base_dir one level above the DomainX folder
    domain_dir = os.path.dirname(file_path)
    base_dir = os.path.dirname(domain_dir)
    with open(file_path, "r") as f:
        lines = f.read().splitlines()
        for line in lines:
            if line.strip():
                img_path, mask_path = line.strip().split()

                def resolve(p):
                    if os.path.isabs(p):
                        return p
                    if p.startswith("Domain"):
                        return os.path.normpath(os.path.join(base_dir, p))
                    return os.path.normpath(os.path.join(domain_dir, p))

                images.append(resolve(img_path))
                masks.append(resolve(mask_path))
    return images, masks


def load_data(path, domain_num=None):
    train_names_path = f"{path}/Domain{domain_num}_train.list"
    valid_names_path = f"{path}/Domain{domain_num}_test.list"

    train_x, train_y = load_names(train_names_path)
    valid_x, valid_y = load_names(valid_names_path)
    return train_x, train_y, valid_x, valid_y


def load_combined_data(d1_path, d2_path, d3_path, d4_path):
    # D1, D2, D3 -> train/val; D4 -> test
    d1_train_x, d1_train_y, d1_val_x, d1_val_y = load_data(d1_path, domain_num=4)
    d2_train_x, d2_train_y, d2_val_x, d2_val_y = load_data(d2_path, domain_num=2)
    d3_train_x, d3_train_y, d3_val_x, d3_val_y = load_data(d3_path, domain_num=3)
    d4_train_x, d4_train_y, d4_val_x, d4_val_y = load_data(d4_path, domain_num=1)

    train_x = d1_train_x + d2_train_x + d3_train_x
    train_y = d1_train_y + d2_train_y + d3_train_y
    val_x = d1_val_x + d2_val_x + d3_val_x
    val_y = d1_val_y + d2_val_y + d3_val_y
    test_x = d4_train_x + d4_val_x
    test_y = d4_train_y + d4_val_y
    return train_x, train_y, val_x, val_y, test_x, test_y


class DATASET(Dataset):
    def __init__(self, images_path, masks_path, size, transform=None):
        super().__init__()
        self.images_path = images_path
        self.masks_path = masks_path
        self.transform = transform
        self.n_samples = len(images_path)
        self.size = size

    def __getitem__(self, index):
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        background = 255 - mask.copy()

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask, background=background)
            image = augmentations["image"]
            mask = augmentations["mask"]
            background = augmentations["background"]

        image = cv2.resize(image, self.size)
        image = np.transpose(image, (2, 0, 1)).astype(np.float32) / 255.0

        mask = cv2.resize(mask, self.size)
        mask = np.expand_dims(mask, axis=0).astype(np.float32) / 255.0

        background = cv2.resize(background, self.size)
        background = np.expand_dims(background, axis=0).astype(np.float32) / 255.0

        return torch.from_numpy(image), (torch.from_numpy(mask), torch.from_numpy(background))

    def __len__(self):
        return self.n_samples


def complementary_loss(prob_fg, prob_bg, prob_uc):
    loss = (prob_fg * prob_bg).sum() + (prob_fg * prob_uc).sum() + (prob_bg * prob_uc).sum()
    num_pixels = prob_fg.size(0) * prob_fg.size(2) * prob_fg.size(3)
    return loss / num_pixels


def train(model, loader, optimizer, loss_fn, device):
    model.train()
    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0
    epoch_asd = 0.0

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

        loss_comp = complementary_loss(prob_fg, prob_bg, prob_uc).to(device)
        loss = loss_mask + beta1 * loss_fg + beta2 * loss_bg + loss_comp
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        batch_jac, batch_f1, batch_recall, batch_precision, batch_asd = [], [], [], [], []
        for yt, yp in zip(y1, mask_pred):
            score = calculate_metrics(yt, yp)
            batch_jac.append(score[0])
            batch_f1.append(score[1])
            batch_recall.append(score[2])
            batch_precision.append(score[3])
            try:
                yp_np = yp[0].detach().cpu().numpy()
                yt_np = yt[0].detach().cpu().numpy()
                yp_bin = (yp_np > 0.5).astype(np.uint8)
                yt_bin = (yt_np > 0.5).astype(np.uint8)
                if yp_bin.sum() == 0 and yt_bin.sum() == 0:
                    asd_score = 0.0
                else:
                    _, _, _, asd_score = calculate_metric_percase(yp_bin, yt_bin)
                batch_asd.append(asd_score)
            except Exception:
                batch_asd.append(0.0)

        epoch_jac += np.mean(batch_jac)
        epoch_f1 += np.mean(batch_f1)
        epoch_recall += np.mean(batch_recall)
        epoch_precision += np.mean(batch_precision)
        epoch_asd += np.mean(batch_asd)

    epoch_loss /= len(loader)
    epoch_jac /= len(loader)
    epoch_f1 /= len(loader)
    epoch_recall /= len(loader)
    epoch_precision /= len(loader)
    epoch_asd /= len(loader)
    return epoch_loss, [epoch_jac, epoch_f1, epoch_recall, epoch_precision, epoch_asd]


def evaluate(model, loader, loss_fn, device):
    model.eval()
    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0
    epoch_asd = 0.0

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

            loss_comp = complementary_loss(prob_fg, prob_bg, prob_uc).to(device)
            loss = loss_mask + beta1 * loss_fg + beta2 * loss_bg + loss_comp
            epoch_loss += loss.item()

            batch_jac, batch_f1, batch_recall, batch_precision, batch_asd = [], [], [], [], []
            for yt, yp in zip(y1, mask_pred):
                score = calculate_metrics(yt, yp)
                batch_jac.append(score[0])
                batch_f1.append(score[1])
                batch_recall.append(score[2])
                batch_precision.append(score[3])
                try:
                    yp_np = yp[0].detach().cpu().numpy()
                    yt_np = yt[0].detach().cpu().numpy()
                    yp_bin = (yp_np > 0.5).astype(np.uint8)
                    yt_bin = (yt_np > 0.5).astype(np.uint8)
                    if yp_bin.sum() == 0 and yt_bin.sum() == 0:
                        asd_score = 0.0
                    else:
                        _, _, _, asd_score = calculate_metric_percase(yp_bin, yt_bin)
                    batch_asd.append(asd_score)
                except Exception:
                    batch_asd.append(0.0)

            epoch_jac += np.mean(batch_jac)
            epoch_f1 += np.mean(batch_f1)
            epoch_recall += np.mean(batch_recall)
            epoch_precision += np.mean(batch_precision)
            epoch_asd += np.mean(batch_asd)

        epoch_loss /= len(loader)
        epoch_jac /= len(loader)
        epoch_f1 /= len(loader)
        epoch_recall /= len(loader)
        epoch_precision /= len(loader)
        epoch_asd /= len(loader)
        return epoch_loss, [epoch_jac, epoch_f1, epoch_recall, epoch_precision, epoch_asd]


def test(model, loader, loss_fn, device):
    model.eval()
    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0
    epoch_asd = 0.0

    with torch.no_grad():
        for i, ((x), (y1, y2)) in enumerate(tqdm(loader, desc="Testing", total=len(loader))):
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

            loss_comp = complementary_loss(prob_fg, prob_bg, prob_uc).to(device)
            loss = loss_mask + beta1 * loss_fg + beta2 * loss_bg + loss_comp
            epoch_loss += loss.item()

            batch_jac, batch_f1, batch_recall, batch_precision, batch_asd = [], [], [], [], []
            for yt, yp in zip(y1, mask_pred):
                score = calculate_metrics(yt, yp)
                batch_jac.append(score[0])
                batch_f1.append(score[1])
                batch_recall.append(score[2])
                batch_precision.append(score[3])
                try:
                    yp_np = yp[0].detach().cpu().numpy()
                    yt_np = yt[0].detach().cpu().numpy()
                    yp_bin = (yp_np > 0.5).astype(np.uint8)
                    yt_bin = (yt_np > 0.5).astype(np.uint8)
                    if yp_bin.sum() == 0 and yt_bin.sum() == 0:
                        asd_score = 0.0
                    else:
                        _, _, _, asd_score = calculate_metric_percase(yp_bin, yt_bin)
                    batch_asd.append(asd_score)
                except Exception:
                    batch_asd.append(0.0)

            epoch_jac += np.mean(batch_jac)
            epoch_f1 += np.mean(batch_f1)
            epoch_recall += np.mean(batch_recall)
            epoch_precision += np.mean(batch_precision)
            epoch_asd += np.mean(batch_asd)

        epoch_loss /= len(loader)
        epoch_jac /= len(loader)
        epoch_f1 /= len(loader)
        epoch_recall /= len(loader)
        epoch_precision /= len(loader)
        epoch_asd /= len(loader)
        return epoch_loss, [epoch_jac, epoch_f1, epoch_recall, epoch_precision, epoch_asd]


