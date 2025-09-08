import os
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision import tv_tensors
from torchvision import transforms
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from constants import *

# from the official torch repo: https://github.com/pytorch/vision/tree/main/gallery/transforms/helpers.py
def plot(imgs, row_title=None, bbox_width=3, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            boxes = None
            masks = None
            if isinstance(img, tuple):
                img, target = img
                if isinstance(target, dict):
                    boxes = target.get("boxes")
                    masks = target.get("masks")
                elif isinstance(target, tv_tensors.BoundingBoxes):
                    boxes = target

                    # Conversion necessary because draw_bounding_boxes() only
                    # work with this specific format.
                    if tv_tensors.is_rotated_bounding_format(boxes.format):
                        boxes = v2.ConvertBoundingBoxFormat("xyxyxyxy")(boxes)
                else:
                    raise ValueError(f"Unexpected target type: {type(target)}")
            img = F.to_image(img)
            if img.dtype.is_floating_point and img.min() < 0:
                # Poor man's re-normalization for the colors to be OK-ish. This
                # is useful for images coming out of Normalize()
                img -= img.min()
                img /= img.max()

            img = F.to_dtype(img, torch.uint8, scale=True)
            if boxes is not None:
                img = draw_bounding_boxes(img, boxes, colors="yellow", width=bbox_width)
            if masks is not None:
                img = draw_segmentation_masks(img, masks.to(torch.bool), colors=["green"] * masks.shape[0], alpha=.65)

            ax = axs[row_idx, col_idx]
            ax.imshow(img.permute(1, 2, 0).numpy(), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()


#-----------------------custom functions implemented by my own-----------------------
def print_samples(dataset_name, dataset, dataset_labels, num_rows):
    ver_fig_size = num_rows * 2.5
    fig, axes = plt.subplots(num_rows,4, figsize=(10, ver_fig_size), subplot_kw={'xticks': [], 'yticks': []})
    
    print(f"\n\n-------------------------------------------------- Samples from {dataset_name} --------------------------------------------------")
    
    for i, ax in enumerate(axes.flat):
        ax.imshow(plt.imread(dataset[i]))
        ax.set_title(dataset_labels[i])
    
    plt.tight_layout()
    plt.show()


def print_class_samples(class_name, img_paths, num_rows):
    ver_fig_size = num_rows * 2.5
    fig, axes = plt.subplots(num_rows,4, figsize=(10, ver_fig_size), subplot_kw={'xticks': [], 'yticks': []})
    
    print(f"\n\n--------------------------------------- Randomly selected samples from {class_name} ---------------------------------------")

    class_img_paths = [path for path in img_paths if class_name in str(path)] 

    np.random.seed(13)
    random_indices = np.random.randint(0, len(class_img_paths), 4*num_rows)
    
    for i, ax in enumerate(axes.flat):
        ax.imshow(plt.imread(class_img_paths[random_indices[i]]))
        ax.set_title(class_name)
    
    plt.tight_layout()
    plt.show()
    
    
def save_model(model, path_to_save):
    torch.save(model.to("cpu").state_dict(), path_to_save)
    print(f'Saved model to {path_to_save}')


def load_model(model, path_to_model, device):
    model.load_state_dict(torch.load(path_to_model, weights_only = True))
    model.to(device)
    return model


def data_to_device(data, device):
    return data.to(device)


def train_valid_result_from_training(epoch, path_to_train_result):
    # read train result csv
    result_df = pd.read_csv(path_to_train_result)

    # retrieve loss and acc from training and validation
    train_loss, train_acc, valid_loss, valid_acc = result_df.loc[result_df['epoch'] == epoch, ['train_loss', 'train_acc', 'valid_loss', 'valid_acc']].values[0]
    
    return train_acc, train_loss, valid_acc, valid_loss


def get_eval_df(name, model, epoch_filename, path_to_train_result, test_loader, path_to_custom_cnn, device):
    epoch_custom_cnn = get_epoch_from_csv(epoch_filename)
    train_acc, train_loss, valid_acc, valid_loss = train_valid_result_from_training(epoch_custom_cnn, path_to_train_result)
    test_acc, test_loss = compute_acc(test_loader, 'Test', False, model, path_to_custom_cnn, device)
    
    eval_df = pd.DataFrame({
        'model': [name],
        'train_acc': [train_acc],
        'train_loss': [train_loss],
        'valid_acc': [valid_acc],
        'valid_loss': [valid_loss],
        'test_acc': [test_acc],
        'test_loss': [test_loss]
    })

    eval_df.set_index('model', inplace = True)

    return eval_df
    
    
def compute_acc(dataloader, loader_type, to_print, model, path_to_model, device):
    running_loss, correct, total_sample_len = 0., 0, 0
    
    model = load_model(model, path_to_model, device)
    
    # turn on eval mode
    model.eval() 

    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total_sample_len += labels.size(0)
            
    acc = (100 * correct) / total_sample_len  
    loss = running_loss / total_sample_len

    if to_print: print(f'{loader_type} Accuracy: {round(acc,3)}% | {loader_type} Loss: {round(loss,3)}')
    else: return acc, loss
    

def save_epoch_to_csv(epoch_value, filename):
    df = pd.DataFrame({'best_epoch': [epoch_value]})

    # define path to save and save it to csv
    path_to_save = f'{DIR_TO_BEST_EPOCH}/{filename}'
    df.to_csv(path_to_save, index=False)

    print(f"\nBest epoch {epoch_value} saved to {path_to_save}")
    
    
def get_epoch_from_csv(filename):
    epoch_df = pd.read_csv(f'{DIR_TO_BEST_EPOCH}/{filename}')
    return epoch_df.iloc[0, 0]


def plot_train_valid_result(path_to_train_result):
    # read in csv file to df
    result_df = pd.read_csv(path_to_train_result)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 6))

    # lineplots of train and valid loss
    ax1.plot(result_df['epoch'], result_df['train_loss'], label = 'Training Loss', color = 'royalblue')
    ax1.plot(result_df['epoch'], result_df['valid_loss'], label = 'Validation Loss', color = 'darkorange')
    ax1.set_title('Training Loss & Validation Loss Across Epochs')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # lineplots of train and valid acc
    ax2.plot(result_df['epoch'], result_df['train_acc'], label = 'Training Accuracy', color = 'royalblue')
    ax2.plot(result_df['epoch'], result_df['valid_acc'], label = 'Validation Accuracy', color = 'darkorange')
    ax2.set_title('Training Accuracy & Validation Accuracy Across Epochs')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.show()


def get_labels(test_loader, model, path_to_model, device):
    # load model
    model = load_model(model, path_to_model, device)
    
    # turn on eval mode
    model.eval() 

    # define lists to save labels
    target_labels = []
    predicted_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            # move data to mps
            images, labels = images.to(device), labels.to(device)

            # get prdictions
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # add to label lists
            target_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

    return target_labels, predicted_labels


def draw_conf_mat(test_loader, model, path_to_model, device, classes):            
    target_labels, predicted_labels = get_labels(test_loader, model, path_to_model, device)

    cm = confusion_matrix(target_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = classes)
    
    fig, ax = plt.subplots(figsize=(6, 6))  
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    plt.title("Confusion Matrix")
    plt.show()


def plot_all_acc_result(dir_to_train_result):
    csv_files = [os.path.join(dir_to_train_result, f) for f in os.listdir(dir_to_train_result) if f.endswith(".csv")]
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 6))
    axes = axes.flatten()

    for i, (ax, csv_file) in enumerate(zip(axes, csv_files), start = 1):
        result_df = pd.read_csv(csv_file)
        
        # lineplots of train and valid loss
        ax.plot(result_df['epoch'], result_df['train_acc'], label = 'Training Accuracy', color = 'royalblue')
        ax.plot(result_df['epoch'], result_df['valid_acc'], label = 'Validation Accuracy', color = 'darkorange')
        ax.set_title(f'Model {i}')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Accuracy')
        ax.legend()

    axes[5].axis('off')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.suptitle('Training Accuracy & Validation Accuracy', fontsize=13)
    plt.show()


def plot_all_loss_result(dir_to_train_result):
    csv_files = [os.path.join(dir_to_train_result, f) for f in os.listdir(dir_to_train_result) if f.endswith(".csv")]
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 6))
    axes = axes.flatten()

    for i, (ax, csv_file) in enumerate(zip(axes, csv_files), start = 1):
        result_df = pd.read_csv(csv_file)
        
        # lineplots of train and valid loss
        ax.plot(result_df['epoch'], result_df['train_loss'], label = 'Training Loss', color = 'royalblue')
        ax.plot(result_df['epoch'], result_df['valid_loss'], label = 'Validation Loss', color = 'darkorange')
        ax.set_title(f'Model {i}')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend()

    axes[5].axis('off')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.suptitle('Training Loss & Validation Loss', fontsize=13)
    plt.show()


def draw_saliency_map(model, test_ds, indices, classes, device, title):
    # move model to mps
    model.to(device)
    
    # freeze parameters 
    for param in model.parameters():
        param.requires_grad = False

    # turn on eval mode
    model.eval()

    # compute len of indices, and get n_rows
    n_images = len(indices)
    n_cols = 4  
    n_rows = math.ceil(n_images * 2 / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows))
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        image, label = test_ds[idx]

        # undo normalization for display
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        inv_normalize = transforms.Normalize(
            mean=[-m/s for m, s in zip(mean, std)],
            std=[1/s for s in std]
        )

        # img to display 
        image_disp = inv_normalize(image).clamp(0, 1)

        img = image.unsqueeze(0).to(device)
        img.requires_grad = True

        # get model prediction
        predicted = model(img)
        pred_class = predicted.argmax(dim=1).item()

        model.zero_grad()
        score = predicted[0, pred_class]
        score.backward()

        saliency_map, _ = torch.max(torch.abs(img.grad[0]), dim=0)
        saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-10)

        # plot original image
        axes[2*i].imshow(image_disp.cpu().numpy().transpose(1, 2, 0))
        axes[2*i].set_title(f"Target: {classes[label]} | Predicted: {classes[pred_class]}")
        axes[2*i].axis('off')

        # plot saliency map 
        axes[2*i + 1].imshow(saliency_map.cpu().numpy(), cmap='plasma')
        axes[2*i + 1].set_title('Saliency Map')
        axes[2*i + 1].axis('off')

    for j in range(2*n_images, len(axes)):
        axes[j].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle(title, fontsize = 16) # set super title
    plt.show()
