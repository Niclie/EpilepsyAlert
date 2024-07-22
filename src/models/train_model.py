import sys
import os
sys.path.append(os.path.abspath('.')) # to import src package
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import keras
from keras import layers
import numpy as np
from scripts.run_preprocessing import get_dataset


def get_uncompiled_model():
    model = keras.Sequential([
        keras.Input(shape = (1280,23)),

        layers.Conv2D(32, (3, 2), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 2), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 2), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 2), activation='relu'),

        layers.BatchNormalization()
    ])

    return model


def get_compiled_model():
    model = get_uncompiled_model()
    model.compile(
        optimizer = "rmsprop",
        loss = "binary_crossentropy",
        metrics = ["accuracy"],
    )
    return model


def get_tf_dataset(patient_id):
    dataset = get_dataset([patient_id], True)[0]
    print(dataset.keys())
    # trainig_dataset = tf.data.Dataset.from_tensor_slices((dataset['trainig_dataset'], dataset['trainig_label']))
    # test_dataset = tf.data.Dataset.from_tensor_slices((dataset['test_dataset'], dataset['test_label']))

    print('ok')

    return


def main():
    get_tf_dataset('chb01')

    return

if __name__ == '__main__':
    main()