import keras

def build_resnet(input_shape):
    """
    Creates a ResNet model.

    Args:
        input_shape (tuple): shape of the input data.

    Returns:
        keras.models.Model: ResNet model.
    """
    input_layer = keras.layers.Input(input_shape)
    
    # First
    conv1 = keras.layers.Conv1D(filters=64, kernel_size=8, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)
    
    conv2 = keras.layers.Conv1D(filters=128, kernel_size=5, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)
    
    conv3 = keras.layers.Conv1D(filters=128, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)
    
    # Second
    conv4 = keras.layers.Conv1D(filters=64, kernel_size=8, padding="same")(conv3)
    conv4 = keras.layers.BatchNormalization()(conv4)
    conv4 = keras.layers.ReLU()(conv4)
    
    conv5 = keras.layers.Conv1D(filters=128, kernel_size=5, padding="same")(conv4)
    conv5 = keras.layers.BatchNormalization()(conv5)
    conv5 = keras.layers.ReLU()(conv5)
    
    conv6 = keras.layers.Conv1D(filters=128, kernel_size=3, padding="same")(conv5)
    conv6 = keras.layers.BatchNormalization()(conv6)
    conv6 = keras.layers.ReLU()(conv6)
    
    # Third
    conv7 = keras.layers.Conv1D(filters=64, kernel_size=8, padding="same")(conv6)
    conv7 = keras.layers.BatchNormalization()(conv7)
    conv7 = keras.layers.ReLU()(conv7)
    
    conv8 = keras.layers.Conv1D(filters=128, kernel_size=5, padding="same")(conv7)
    conv8 = keras.layers.BatchNormalization()(conv8)
    conv8 = keras.layers.ReLU()(conv8)
    
    conv9 = keras.layers.Conv1D(filters=128, kernel_size=3, padding="same")(conv8)
    conv9 = keras.layers.BatchNormalization()(conv9)
    conv9 = keras.layers.ReLU()(conv9)
    
    # Global Average Pooling
    gap = keras.layers.GlobalAveragePooling1D()(conv9)
    
    # Output
    output_layer = keras.layers.Dense(1, activation="sigmoid")(gap)
    
    return keras.models.Model(inputs=input_layer, outputs=output_layer)


def build_lstm(input_shape):
    """
    Builds an LSTM model.

    Args:
        input_shape (tuple): shape of the input data.

    Returns:
        keras.models.Model: LSTM model.
    """
    model = keras.Sequential()

    # LSTM
    model.add(keras.layers.LSTM(128, input_shape=input_shape))

    # Dropout
    model.add(keras.layers.Dropout(0.5))

    # Fully connected
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    return model


def build_cnn(input_shape):
    """
    Builds a CNN model.

    Args:
        input_shape (tuple): shape of the input data.

    Returns:
        keras.models.Model: CNN model.
    """
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=128, kernel_size=8, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=128, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(1, activation="sigmoid")(gap)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)


def train(model, training_data, training_label, out_path, epochs = 500, batch_size = 32, early_stopping = 50):
    """
    Trains a given model using the provided training data and labels.

    Args:
        model (keras.Model): model to be trained.
        training_data (array-like): input training data.
        training_label (array-like): corresponding training labels.
        out_path (str): path to save the trained model.
        epochs (int, optional): number of epochs to train the model. Defaults to 500.
        batch_size (int, optional): batch size for training. Defaults to 32.
        early_stopping (int, optional): number of epochs to wait before early stopping. Defaults to 50.

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
            monitor="val_loss", patience = early_stopping, verbose=1)
    ]

    model.compile(
        optimizer = "adam",
        loss = "binary_crossentropy",
        metrics = ["accuracy"]
    )

    history = model.fit(
        training_data,
        training_label,
        batch_size = batch_size,
        epochs = epochs,
        callbacks = callbacks,
        validation_split = 0.2,
        verbose = 1
    )

    return history