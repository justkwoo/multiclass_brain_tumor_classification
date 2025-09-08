import copy
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from tqdm.notebook import tqdm
from helpers import *

def train(model, train_loader, valid_loader, device, learning_rate, epochs, patience, path_to_save):
    # define optimizer and loss func
    optimizer = optim.Adam(model.parameters(), learning_rate)
    criterion = nn.CrossEntropyLoss()

    # move model to mps 
    model.to(device)

    # define df to store training and validation result
    result_df = pd.DataFrame(columns=['epoch', 'train_loss', 'train_acc', 'valid_loss', 'valid_acc'])
    
    # params for early stopping
    prev_valid_loss, best_epoch = float("inf"), 0
    best_model_weights = None
    patience_counter = 0

    for epoch in range(epochs):
        
        # turn on train mode
        model.train()

        running_loss, correct, total_sample_len = 0.0, 0, 0

        for images, labels in tqdm(train_loader):
            images, labels = data_to_device(images, device), data_to_device(labels, device)

            # reset gradients
            optimizer.zero_grad()

            # forward pass
            outputs = model(images)

            # compute loss + update weights
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # training stat
            running_loss += loss.item() * images.size(0)
            _, predicted_label = torch.max(outputs, 1)
            correct += (predicted_label == labels).sum().item()
            total_sample_len += labels.size(0)

        train_loss = running_loss / total_sample_len
        train_acc = 100. * correct / total_sample_len
        
        valid_acc, valid_loss = compute_valid_acc(model, valid_loader, device, criterion)

        # store train acc and loss, valid acc and loss
        result_df.loc[epoch] = [epoch + 1, train_loss, train_acc, valid_loss, valid_acc]

        print(f'Epoch: {epoch + 1}/{epochs}')
        print(f'Training Accuracy {round(train_acc, 3)}% | Traning Loss: {round(train_loss, 3)}')
        print(f'Validation Accuracy {round(valid_acc, 3)}% | Validation Loss: {round(valid_loss, 3)}')

        if valid_loss < prev_valid_loss:
            # update best valid loss and epoch
            prev_valid_loss = valid_loss
            best_epoch = epoch + 1

            # reset patience counter
            patience_counter = 0
            
            # save best model weights
            best_model_weights = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

        prev_valid_loss = valid_loss

    print('\nFinished training')

    if best_model_weights is not None:
        print(f'Best model at epoch {best_epoch}')
        
        # load best weights and save model
        model.load_state_dict(best_model_weights)
        save_model(model, path_to_save)

    # return result_df
    return result_df, best_epoch

    
def compute_valid_acc(model, valid_loader, device, criterion):
    # turn on eval mode
    model.eval()
    
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    valid_acc = 100. * correct / total
    valid_loss = running_loss / total 
    
    return valid_acc, valid_loss