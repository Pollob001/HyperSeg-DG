import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from operator import add
import numpy as np
import cv2
from tqdm import tqdm
import torch
from network.model import HyperSegStage2
from utils.utils import create_dir, seeding
from utils.utils import calculate_metrics
from utils.run_engine import load_data
from utils.metrics import calculate_metric_percase


def process_mask(y_pred):
    y_pred = y_pred[0].cpu().numpy()
    y_pred = np.squeeze(y_pred, axis=0)
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.int32)
    y_pred = y_pred * 255
    y_pred = np.array(y_pred, dtype=np.uint8)
    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=2)
    return y_pred


def process_edge(y_pred):
    y_pred = y_pred[0].cpu().numpy()
    y_pred = np.squeeze(y_pred, axis=0)

    y_pred = y_pred > 0.001
    y_pred = y_pred.astype(np.int32)
    y_pred = y_pred * 255
    y_pred = np.array(y_pred, dtype=np.uint8)
    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=2)
    return y_pred


def print_score(metrics_score, num_samples):
    jaccard = metrics_score[0] / num_samples
    f1 = metrics_score[1] / num_samples
    recall = metrics_score[2] / num_samples
    precision = metrics_score[3] / num_samples
    acc = metrics_score[4] / num_samples
    f2 = metrics_score[5] / num_samples
    asd = metrics_score[6] / num_samples

    print(
        f"Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f} - Acc: {acc:1.4f} - F2: {f2:1.4f} - ASD: {asd:1.4f}")


def evaluate(model, save_path, test_x, test_y, size):
    """Evaluate model on test dataset and calculate metrics including ASD"""
    metrics_score_1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # [jaccard, f1, recall, precision, acc, f2, asd]
    num_samples = len(test_x)

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        name = y.split("/")[-1].split(".")[0]

        """ Image """
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        image = cv2.resize(image, size)
        save_img = image.copy()
        image = np.transpose(image, (2, 0, 1))
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        image = image.to(device)

        """ Mask """
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, size)
        save_mask = mask.copy()
        save_mask = np.expand_dims(save_mask, axis=-1)
        save_mask = np.concatenate([save_mask, save_mask, save_mask], axis=2)
        mask = np.expand_dims(mask, axis=0)
        mask = mask / 255.0
        mask = np.expand_dims(mask, axis=0)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)
        mask = mask.to(device)

        with torch.no_grad():
            mask_pred, fg_pred, bg_pred, uc_pred = model(image)
            p1 = mask_pred

            """ Evaluation metrics """
            # Calculate standard metrics (jaccard, f1, recall, precision, acc, f2)
            score_1 = calculate_metrics(mask, p1)
            
            # Calculate medpy metrics (dc, jc, hd, asd) - we need ASD
            # medpy expects 2D numpy arrays, so convert and squeeze tensors
            try:
                # Convert tensors to numpy arrays: p1 and mask are [1, 1, H, W]
                # Remove batch and channel dimensions to get [H, W]
                p1_np = p1[0, 0].detach().cpu().numpy()  # Direct indexing: [1,1,H,W] -> [H,W]
                mask_np = mask[0, 0].detach().cpu().numpy()  # Direct indexing: [1,1,H,W] -> [H,W]
                
                # Ensure they are 2D arrays
                assert p1_np.ndim == 2, f"p1_np should be 2D, got shape {p1_np.shape}"
                assert mask_np.ndim == 2, f"mask_np should be 2D, got shape {mask_np.shape}"
                    
                _, _, _, asd_score = calculate_metric_percase(p1_np, mask_np)
            except Exception as e:
                print(f"Warning: ASD calculation failed for {name}: {e}")
                import traceback
                traceback.print_exc()
                asd_score = 0.0
            
            # Accumulate metrics - preserve ASD at index 6
            # score_1 has 6 elements, metrics_score_1 has 7 (including ASD)
            for idx in range(len(score_1)):
                metrics_score_1[idx] += score_1[idx]
            metrics_score_1[6] += asd_score  # Accumulate ASD
            
            # Process and save mask
            p1_processed = process_mask(p1)
            cv2.imwrite(f"{save_path}/mask/{name}.jpg", p1_processed)

    # Print and save results
    print_score(metrics_score_1, num_samples)

    with open(f"{save_path}/result.txt", "w") as file:
        file.write(f"Jaccard: {metrics_score_1[0] / num_samples:1.4f}\n")
        file.write(f"F1: {metrics_score_1[1] / num_samples:1.4f}\n")
        file.write(f"Recall: {metrics_score_1[2] / num_samples:1.4f}\n")
        file.write(f"Precision: {metrics_score_1[3] / num_samples:1.4f}\n")
        file.write(f"Acc: {metrics_score_1[4] / num_samples:1.4f}\n")
        file.write(f"F2: {metrics_score_1[5] / num_samples:1.4f}\n")
        file.write(f"ASD: {metrics_score_1[6] / num_samples:1.4f}\n")



if __name__ == "__main__":
    """ Seeding """

    dataset_name = 'CVC'

    seeding(42)
    size = (256, 256)
    """ Load the checkpoint """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HyperSegStage2(256, 256)
    model = model.to(device)
    checkpoint_path = "/ghome/aynulislam/HyperSeg_DG/runfile/CVC/CVC_None_lr0.0001_20251119-023813/checkpoint.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    """ Test dataset """
    path = "/gdata/aynulislam/SEGDATASET/{}/".format(dataset_name)
    (train_x, train_y), (test_x, test_y) = load_data(path)
    # Combine path lists
    all_paths = train_x + test_x
    # Combine label lists
    all_labels = train_y + test_y

    save_path = f"/ghome/aynulislam/HyperSeg_DG/runfile/CVC/CVC_None_lr0.0001_20251119-023813/{dataset_name}/MyModel"

    create_dir(f"{save_path}/mask")
    evaluate(model, save_path, all_paths, all_labels, size)
