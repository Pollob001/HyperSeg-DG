import os
import sys
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm

from wmamba import wmamba_s

def main():
    device = torch.device("cuda")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    train_dataset = datasets.ImageFolder(root="/gdata1/alif/ImageNet/train",
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    flower_list = train_dataset.class_to_idx
    # Reverse the dictionary to save index -> class_name
    cla_dict = {index: class_name for class_name, index in flower_list.items()}

    # Write the dictionary to JSON file
    json_str = json.dumps(cla_dict, indent=4)
    with open('/ghome/alif/MedMamba224x224/class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 256
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 16])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root="/gdata1/alif/ImageNet/val",
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)
    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    # Initialize the model
    num_classes = len(train_dataset.classes)
    net = wmamba_s(num_classes=num_classes)

    # If multiple GPUs are available, use DataParallel
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        net = nn.DataParallel(net)  # Wrap the model for multi-GPU usage

    net.to(device)  # Transfer the model to the appropriate device (GPU or CPU)

    # Define the loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    max_patience = 20

    patience = 0
    best_acc = -float('inf')
    epochs = 300

    model_name = "swinmamba_s"
    save_path = os.path.join("/ghome/alif/MedMamba224x224/swinmamba_s", "{}Net.pth".format(model_name))
    train_steps = len(train_loader)

    # Open a text file to save epoch information
    log_file = open('/ghome/alif/MedMamba224x224/swinmamba_s/training_log.txt', 'w')
    log_file.write("Epoch\tTrain_Loss\tVal_Accuracy\n")  # Write header to log file

    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' % 
              (epoch + 1, running_loss / train_steps, val_accurate))

        # Log epoch results to the file
        log_file.write(f"{epoch + 1}\t{running_loss / train_steps:.3f}\t{val_accurate:.3f}\n")

        # Early stopping logic
        if val_accurate > best_acc:
            patience = 0
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
        else:
            patience += 1
            if patience> max_patience:
                print("Early stopping triggered")
                break

    # Close the log file after training is complete
    log_file.close()
    print('Finished Training')


if __name__ == '__main__':
    main()
