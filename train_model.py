import numpy as np

import os 
from glob import glob
import torch

from monai.transforms import (
    Compose,
    LoadImaged,
    ToTensord,
    AddChanneld,
    ScaleIntensityRanged,
    Resized
    )
from monai.data import Dataset, DataLoader 
from monai.losses import DiceLoss 

def prepare(data_path, img_size):

    ''' 
        Function prepares the dictionaries with labeled data for testing and training. 
        Performs pre-processing transformations on data converts them to tensors. 

        Returns train_loader and test_loader to be used with further training and testing functions
    '''
    
    train_images = sorted(glob(os.path.join(data_path, 'training_data', 'images', '*.nii.gz')))
    train_labels = sorted(glob(os.path.join(data_path, 'training_data', 'labels', '*.nii.gz')))

    test_images = sorted(glob(os.path.join(data_path, 'test_data', 'images', '*.nii.gz')))
    test_labels = sorted(glob(os.path.join(data_path, 'test_data', 'labels', '*.nii.gz')))

    train_files = [{'image':image, 'label':label} for image, label in zip(train_images, train_labels)]
    test_files = [{'image':image, 'label':label} for image, label in zip(test_images, test_labels)]

    train_transforms = Compose( 
        [
            LoadImaged(keys=['image', 'label']),
            AddChanneld(keys=['image', 'label']),
            ScaleIntensityRanged(keys='image', a_min=0, a_max=1500, b_min=0.0, b_max=1.0, clip=True), # improving contrast of an image: 'a' values should be tuned depending on used data
            Resized(keys=['image', 'label'], spatial_size=[img_size, img_size, 128]),
            ToTensord(keys=['image', 'label'])
        ]
    )

    test_transforms = Compose( 
        [
            LoadImaged(keys=['image', 'label']),
            AddChanneld(keys=['image', 'label']),
            ScaleIntensityRanged(keys='image', a_min=0, a_max=1500, b_min=0.0, b_max=1.0, clip=True), # improving contrast of an image: 'a' values should be tuned depending on used data
            Resized(keys=['image', 'label'], spatial_size=[img_size, img_size, 128]),
            ToTensord(keys=['image', 'label'])
        ]
    )

    train_ds = Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=1)

    test_ds = Dataset(data=test_files, transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=1)

    return train_loader, test_loader

def dice_metric(predicted, target):

    '''
        Function performs dice metric calculation to be used for model optimization.

        Returns a value of dice metric 
    '''

    dice_value = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)
    value = 1 - dice_value(predicted, target).item()
    return value

def train(train_loader, test_loader, model, max_epochs, device, optimizer, loss_function, model_path):

    '''
        Function trains the model, using provided labeled data, to achieve the lowest mean dice loss. 

        Saves model with the best dice metric to "best_metric_model.pth" file. 
        Saves calulated dice metric values to coresponding '*.npy' files. 
    '''

    test_interval = 1

    best_metric = -1
    best_metric_epoch = -1
    save_loss_train = []
    save_loss_test = []
    save_metric_train = []
    save_metric_test = []
    outputs = []

    for epoch in range(max_epochs):

        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        train_epoch_loss = 0
        train_step = 0
        epoch_metric_train = 0

        for batch_data in train_loader:
            
            train_step += 1

            volume = batch_data["image"]
            label = batch_data["label"]
            label = label != 0
            volume, label = (volume.to(device), label.to(device))
            
            optimizer.zero_grad()
            outputs = model(volume)
            
            train_loss = loss_function(outputs, label)
            
            train_loss.backward()
            optimizer.step()

            train_epoch_loss += train_loss.item()
            print(
                f"{train_step}/{len(train_loader) // train_loader.batch_size}, "
                f"Train_loss: {train_loss.item():.4f}")

            train_metric = dice_metric(outputs, label)
            epoch_metric_train += train_metric
            print(f'Train_dice: {train_metric:.4f}')

        print('-'*20)
        
        train_epoch_loss /= train_step
        print(f'Epoch_loss: {train_epoch_loss:.4f}')
        save_loss_train.append(train_epoch_loss)
        np.save(os.path.join(model_path, 'loss_train.npy'), save_loss_train)
        
        epoch_metric_train /= train_step
        print(f'Epoch_metric: {epoch_metric_train:.4f}')

        save_metric_train.append(epoch_metric_train)
        np.save(os.path.join(model_path, 'metric_train.npy'), save_metric_train)

        if (epoch + 1) % test_interval == 0:

            model.eval()
            with torch.no_grad():
                test_epoch_loss = 0
                test_metric = 0
                epoch_metric_test = 0
                test_step = 0

                for test_data in test_loader:

                    test_step += 1

                    test_volume = test_data["image"]
                    test_label = test_data["label"]
                    test_label = test_label != 0
                    test_volume, test_label = (test_volume.to(device), test_label.to(device),)
                    
                    test_outputs = model(test_volume)
                    
                    test_loss = loss_function(outputs, test_label)
                    test_epoch_loss += test_loss.item()
                    test_metric = dice_metric(test_outputs, test_label)
                    epoch_metric_test += test_metric                    
                
                test_epoch_loss /= test_step
                print(f'test_loss_epoch: {test_epoch_loss:.4f}')
                save_loss_test.append(test_epoch_loss)
                np.save(os.path.join(model_path, 'loss_test.npy'), save_loss_test)

                epoch_metric_test /= test_step
                print(f'test_dice_epoch: {epoch_metric_test:.4f}')
                save_metric_test.append(epoch_metric_test)
                np.save(os.path.join(model_path, 'metric_test.npy'), save_metric_test)

                if epoch_metric_test > best_metric:
                    best_metric = epoch_metric_test
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(
                        model_path, "best_metric_model.pth"))
                
                print(
                    f"current epoch: {epoch + 1} current mean dice: {test_metric:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}"
                )

    print(
        f"train completed, best_metric: {best_metric:.4f} "
        f"at epoch: {best_metric_epoch}")

