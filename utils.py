import os
import shutil

import cv2
import glob
import numpy as np
import pickle as pkl
from tqdm import tqdm
import matplotlib.pyplot as plt


def organize_data(base_path='/home/user/PycharmProjects/Chess_Piece_Recognition/data/'):
    """
    Arranges data in train, test and validation folders
    :param base_path: path to one level above dataset folder
    :returns: True if No error else returns False
    .. note:: expected file structure inside base path
        -Chess/
            |-Rook/
            |-Knight/
            |-Bishop/
            |-King/
            |-Pawn/
            |-Queen/
    """
    try:
        BASE_PATH = base_path
        SPLITS = ['train/', 'test/', 'validation/']

        for split in SPLITS:
            os.mkdir(BASE_PATH + split)

        for dir1 in os.listdir(BASE_PATH + 'Chess'):
            if os.path.isdir(BASE_PATH + 'Chess/' + dir1):
                for split in SPLITS:
                    os.mkdir(BASE_PATH + split + dir1)
                files = list(glob.glob(BASE_PATH + 'Chess/' + dir1 + '/*'))
                train_l = round(len(files) * 0.7)
                val_l = round(len(files) * 0.8)
                train_share = files[:train_l]
                val_share = files[train_l:val_l]
                test_share = files[val_l:]
                for split, share in zip(SPLITS, [train_share, test_share, val_share]):
                    for file in share:
                        shutil.copy(file, BASE_PATH + split + dir1)
        return True
    except Exception as e:
        print(e)
        return False


def plot_history(history=None):
    """
    plots history of training acc and loss
    :param history: history object from training
    """
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and Validation accuracy')
    plt.legend()

    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.legend()
    plt.show()


def pickle_data(base_path='data/'):
    CLASS_MAP = {'Bishop': 0, 'King': 1, 'Knight': 2, 'Pawn': 3, 'Queen': 4, 'Rook': 5}
    train_val = []
    train_lab = []
    val_val = []
    val_lab = []
    test_val = []
    test_lab = []

    for dir1 in tqdm(os.listdir(base_path)):
        print(dir1)
        for dir2 in tqdm(os.listdir(base_path + dir1)):
            files = list(glob.glob(base_path + dir1 + '/' + dir2 + '/*'))
            for f in files:
                img = cv2.imread(f)
                if img is not None:
                    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    if dir1 == 'train':
                        train_val.append(img_gray)
                        train_lab.append(CLASS_MAP[dir2])
                    elif dir1 == 'validation':
                        val_val.append(img_gray)
                        val_lab.append(CLASS_MAP[dir2])
                    elif dir1 == 'test':
                        test_val.append(img_gray)
                        test_lab.append(CLASS_MAP[dir2])
    train_val = np.array(train_val, dtype=float)
    train_lab = np.array(train_lab)
    val_val = np.array(val_val, dtype=float)
    val_lab = np.array(val_lab)
    test_val = np.array(test_val, dtype=float)
    test_lab = np.array(test_lab)

    with open('train_data.pkl', 'wb') as f:
        pkl.dump(train_val, f)
        pkl.dump(train_lab, f)

    with open('validation_data.pkl', 'wb') as f:
        pkl.dump(val_val, f)
        pkl.dump(val_lab, f)

    with open('test_data.pkl', 'wb') as f:
        pkl.dump(test_val, f)
        pkl.dump(test_lab, f)


if __name__ == '__main__':
    organize_data()
    pickle_data()
