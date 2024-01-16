import torch

from matplotlib import pyplot as plt
import numpy as np
import sklearn
from scipy.interpolate import interp1d
from scipy.integrate import quad
import cv2

def plot_results(train_data, val_data, label, save_dir):

    plt.figure(figsize=(6, 6))
    plt.title(label)
    plt.xlabel('Epoch')
    plt.ylabel(label)
    plt.plot(train_data, label=f'Train {label}')
    plt.plot(val_data, label=f'Validation {label}')
    plt.legend()
    plt.savefig(save_dir + f'/{label}.png')

def plot_several_results(all_results, variable_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    i = 0
    for key, value in all_results.items():
        if i == 0: color = 'blue'
        elif i == 1: color = 'orange'
        elif i == 2: color = 'green'
        else : color = 'red'

        ax1.plot(value['train_loss'], label=f'{variable_name}={key}, train loss', color=color, )
        ax1.plot(value['val_loss'], label=f'{variable_name}={key}, val loss', color=color, linestyle='dashed')
        ax1.legend()

        ax2.plot(value['train_acc'], label=f'{variable_name}={key}, train acc', color=color, )
        ax2.plot(value['val_acc'], label=f'{variable_name}={key}, val acc', color=color, linestyle='dashed')
        ax2.legend()

        i+=1

    plt.show()

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model
