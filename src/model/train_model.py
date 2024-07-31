import sys
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
sys.path.append(os.path.abspath('.'))
import keras


def get_uncompiled_cnn_model(input_shape):
    """
    Creates an uncompiled CNN model.

    Args:
        input_shape (tuple): shape of the input data.

    Returns:
        keras.models.Model: uncompiled CNN model.
    """

    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(1, activation="sigmoid")(gap)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)


def get_uncompiled_cnn_rnn_model(input_shape):
    """
    Creates an uncompiled CNN-RNN model.

    Args:
        input_shape (tuple): shape of the input data.

    Returns:
        keras.models.Model: uncompiled CNN-RNN model.
    """

    input_layer = keras.layers.Input(input_shape)

    # CNN layers
    conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    # RNN layer
    rnn = keras.layers.LSTM(64, return_sequences=False)(conv3)

    dense = keras.layers.Dense(64, activation='relu')(rnn)
    dropout = keras.layers.Dropout(0.5)(dense)

    output_layer = keras.layers.Dense(1, activation="sigmoid")(dropout)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)


def train_model(model, training_data, training_label, out_path, epochs=500, batch_size=32):
    """
    Trains a given model using the provided training data and labels.

    Args:
        model (keras.Model): model to be trained.
        training_data (array-like): input training data.
        training_label (array-like): corresponding training labels.
        out_path (str): path to save the trained model.
        epochs (int, optional): number of epochs to train the model. Defaults to 500.
        batch_size (int, optional): batch size for training. Defaults to 32.

    Returns:
        dictionary: training history object.
    """
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            f'{out_path}.keras', save_best_only=True, monitor="val_loss", verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=50, verbose=1)
    ]

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        training_data,
        training_label,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        validation_split=0.2,
        verbose=1
    )

    return history