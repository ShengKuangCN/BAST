import os
import wave
import numpy as np
from scipy import signal
from datetime import datetime
import random
from sklearn.preprocessing import StandardScaler
from conf import *


def load_dataset(env='RI', train=True, azimuth=None, raw_path=None, converted_path=None):
    """
    This function exists to
        (1) convert the raw sound waves from 'raw_path' to spectrogram if the spectrogram file(s) dose not exist.
     or (2) load the spectrogram file(s) if exists in the path 'converted_path'.

    :param env: the name of acoustic environment,
        'RI01': the anechoic environment only,
        'RIO2': the reverberation environment only,
        'RI': both environments.
    :param train: load whether the training dataset or the BAST dataset,
        True: training dataset (75%),
        False: BAST dataset (25%).
    :param azimuth:
        None: all data from 0 to 350 (interval = 10 degrees)
        0: the data spatialized to the azimuth of 0 degree,
        10: the data spatialized to the azimuth of 10 degree,
        ...
        350: the data spatialized to the azimuth of 350 degree.
    :param raw_path: the path of raw sound waves. There should be two separate sub-folders under this path,
    i.e. 'raw_path/train/' and 'raw_path/BAST/'.
    :param converted_path: the path to save/load the spectrogram file(s). Two sub-folders will be automatically created
    if they do not exist, i.e. 'converted_path/train/' and 'converted_path/BAST/'.

    :return: x, y
        x: the spectrogram object with a size of n * 2 * 129 * 61 ([n_sample]*[left&right]*[height]*[time]),
        y: the x-y coordinates with a size of n * 2.
    """
    if train:
        data_dir = raw_path + 'train' + os.path.sep
        converted_data_path = converted_path + 'train' + os.path.sep
    else:
        data_dir = raw_path + 'test' + os.path.sep
        converted_data_path = converted_path + 'test' + os.path.sep
    if azimuth is not None:
        arr = np.arange(0, 360, 10)
        assert azimuth in arr, 'azimuth should be one of {}'.format(arr)
    data_path = converted_data_path + env + ('.npz' if azimuth is None else '_{:03}.npz'.format(azimuth))
    print('The spectrogram will be saved to: ' + data_path)
    if not os.path.exists(data_path):
        t_data_list = os.listdir(data_dir)
        data_list = []
        for f_name in t_data_list:
            if env in f_name:
                if azimuth is not None:
                    if int(f_name[1:4]) == azimuth:
                        data_list.append(f_name)
                else:
                    data_list.append(f_name)
        data_list.sort()
        dataset_x = np.zeros((len(data_list), 2, 8000))
        dataset_y = np.zeros((len(data_list), 2))
        length = len(data_list)
        for i_name in range(length):
            f = wave.open(data_dir + data_list[i_name], 'rb')
            y = int(data_list[i_name][1:4])
            params = f.getparams()
            n_channels, samp_width, frame_rate, n_frames = params[:4]
            str_data = f.readframes(n_frames)
            f.close()
            wave_data = np.frombuffer(str_data, dtype=np.int16)
            wave_data.shape = -1, 2
            wave_data = wave_data.T
            dataset_x[i_name, :, :] = wave_data
            y_coordinate_x = np.cos(2 * np.pi * y / 360)
            y_coordinate_y = np.sin(2 * np.pi * y / 360)
            dataset_y[i_name, 0] = y_coordinate_x
            dataset_y[i_name, 1] = y_coordinate_y
            print('\r[{}] Loading sound waves: {} ({:.2f}%)'.format(datetime.now(), i_name + 1, 100 * (i_name + 1) / length), end='')
        print('\r[{}] Loading sound waves: {} ({:.2f}%)'.format(datetime.now(), i_name + 1, 100 * (i_name + 1) / length))

        spectrogram_x = np.zeros((len(data_list), 2, 129, 61))
        for idx in range(dataset_x.shape[0]):
            x1 = dataset_x[idx, 0, :]
            x2 = dataset_x[idx, 1, :]
            _, _, spectrogram1 = signal.spectrogram(x1, frame_rate, nperseg=256, noverlap=256 // 2)
            _, _, spectrogram2 = signal.spectrogram(x2, frame_rate, nperseg=256, noverlap=256 // 2)
            spectrogram_x[idx, 0, :, :] = spectrogram1
            spectrogram_x[idx, 1, :, :] = spectrogram2
            print('\r[{}] Converting sound to spectrogram: {} ({:.2f}%)'.format(datetime.now(), idx + 1,
                                                                       100 * (idx + 1) / dataset_x.shape[0]), end='')
        print('\r[{}] Converting sound to spectrogram: {} ({:.2f}%)'.format(datetime.now(), idx + 1,
                                                                   100 * (idx + 1) / dataset_x.shape[0]))
        spectrogram_x = 20 * np.log10(spectrogram_x)
        print('\r[{}] Saving data: {}'.format(datetime.now(), data_path))
        np.savez(data_path, spectrogram_x, dataset_y)
        return spectrogram_x, dataset_y
    else:
        d = np.load(data_path, allow_pickle=True)
        print('\r[{}] Loading converted data: {}'.format(datetime.now(), data_path))
        return d['arr_0'], d['arr_1']


def norm_image(data):
    """Normalize the spectrogram"""
    scaler = StandardScaler()
    data = scaler.fit_transform(data.reshape(data.shape[0], -1).T).T.reshape(data.shape)
    return data


class DataManager:
    """
    This class aims at splitting the input dataset into two subsets as training and BAST datasets.
    """
    def __init__(self, x, y):
        """
        :param x: x
        :param y: y
        """
        self.x = x
        self.y = y
        self.n_files_per_angle = np.sum(y == 0)
        self.n_directions = int(x.shape[0] / self.n_files_per_angle)

    def split_data_large(self, percent_list):
        """
        :param percent_list: the percentage of training ans validation dataset, default [0.75, 0.25].
        :return: [training_x, training_y, validation_x, validation_y]
        """
        self.x = norm_image(self.x)
        idx_per_angle = np.arange(self.n_files_per_angle)
        random.shuffle(idx_per_angle)

        n_training_per_angle = int(percent_list[0] * self.n_files_per_angle)
        n_validation_per_angle = int(percent_list[1] * self.n_files_per_angle)

        idx_training_per_angle = idx_per_angle[:n_training_per_angle]
        idx_validation_per_angle = idx_per_angle[n_training_per_angle:n_training_per_angle + n_validation_per_angle]

        training_x = np.zeros((idx_training_per_angle.shape[0] * self.n_directions, *self.x.shape[1:]), dtype=np.float32)
        validation_x = np.zeros((idx_validation_per_angle.shape[0] * self.n_directions, *self.x.shape[1:]), dtype=np.float32)

        training_y = np.zeros((idx_training_per_angle.shape[0] * self.n_directions, *self.y.shape[1:]), dtype=np.float32)
        validation_y = np.zeros((idx_validation_per_angle.shape[0] * self.n_directions, *self.y.shape[1:]), dtype=np.float32)

        for i in range(self.n_directions):
            training_x[i * n_training_per_angle:(i + 1) * n_training_per_angle, :, :, :] = self.x[idx_training_per_angle + i * self.n_files_per_angle, :, :, :]
            training_y[i * n_training_per_angle:(i + 1) * n_training_per_angle, :] = self.y[idx_training_per_angle + i * self.n_files_per_angle, :]

            validation_x[i * n_validation_per_angle:(i + 1) * n_validation_per_angle, :, :, :] = self.x[idx_validation_per_angle + i * self.n_files_per_angle, :, :, :]
            validation_y[i * n_validation_per_angle:(i + 1) * n_validation_per_angle, :] = self.y[idx_validation_per_angle + i * self.n_files_per_angle, :]

        tr_indices = np.arange(training_y.shape[0])
        val_indices = np.arange(validation_y.shape[0])
        np.random.shuffle(tr_indices)
        np.random.shuffle(val_indices)

        training_x = training_x[tr_indices]
        training_y = training_y[tr_indices]
        validation_x = validation_x[val_indices]
        validation_y = validation_y[val_indices]

        print('[{}], Training Dataset: {}, {}'.format(datetime.now(), training_x.shape, training_y.shape))
        print('[{}], Validation Dataset: {}, {}'.format(datetime.now(), validation_x.shape, validation_y.shape))

        return training_x, training_y, validation_x, validation_y


if __name__ == '__main__':
    x, y = load_dataset('RI', train=False, azimuth=None, raw_path=DATA_DIR, converted_path=CONVERTED_DATA_DIR)



