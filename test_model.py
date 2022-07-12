import torch
import os 

import matplotlib.pyplot as plt
import numpy as np

from monai.utils import first
from monai.inferers import sliding_window_inference


def test(test_loader, model, model_path, img_size, device, s_min, s_max):
    
    '''
        Function performs testing of provided model using data of first patient from 'test_data' directory. 

        Plots tested images with corresponding labels and predicted segmentation results. 
    '''

    model.load_state_dict(torch.load(os.path.join(model_path, "best_metric_model.pth")))
    sw_batch_size = 4
    roi_size = (img_size, img_size, 128)
    with torch.no_grad():
        test_patient = first(test_loader)
        t_volume = test_patient['image']
        
        test_outputs = sliding_window_inference(t_volume.to(device), roi_size, sw_batch_size, model)
            
        for i in range(s_min, s_max):
            plt.figure(figsize=(20, 10))
            plt.subplot(1, 3, 1)
            plt.title(f"Slice {i}: Image")
            plt.imshow(test_patient["image"][0, 0, :, :, i], cmap="gray")
            plt.subplot(1, 3, 2)
            plt.title(f"Slice {i}: Label")
            plt.imshow(test_patient["label"][0, 0, :, :, i] != 0)
            plt.subplot(1, 3, 3)
            plt.title(f"Slice {i}: Predicted segmentation result")
            plt.imshow(test_outputs.detach().cpu()[0, 1, :, :, i])
            plt.show()

def show_stats(model_path):

    '''
        Function plots dice metric values calculated for provided model. 
    '''

    train_loss = np.load(os.path.join(model_path, 'loss_train.npy'))
    train_metric = np.load(os.path.join(model_path, 'metric_train.npy'))
    test_loss = np.load(os.path.join(model_path, 'loss_test.npy'))
    test_metric = np.load(os.path.join(model_path, 'metric_test.npy'))

    plt.figure(figsize=(20, 10))
    plt.subplot(2, 2, 1)
    plt.title("Train dice loss")
    x = [i + 1 for i in range(len(train_loss))]
    y = train_loss
    plt.xlabel("Epoch")
    plt.plot(x, y)

    plt.subplot(2, 2, 2)
    plt.title("Train metric DICE")
    x = [i + 1 for i in range(len(train_metric))]
    y = train_metric
    plt.xlabel("Epoch")
    plt.plot(x, y)

    plt.subplot(2, 2, 3)
    plt.title("Test dice loss")
    x = [i + 1 for i in range(len(test_loss))]
    y = test_loss
    plt.xlabel("Epoch")
    plt.plot(x, y)

    plt.subplot(2, 2, 4)
    plt.title("Test metric DICE")
    x = [i + 1 for i in range(len(test_metric))]
    y = test_metric
    plt.xlabel("Epoch")
    plt.plot(x, y)

    plt.show()