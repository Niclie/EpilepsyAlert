import tensorflow as tf
def build_mlp(input_shape):
    return tf.keras.models.Sequential([
        tf.keras.layers.Input(input_shape),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(300, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    
# def build_cnn(input_shape):
#     """
#     Builds a CNN model.

#     Args:
#         input_shape (tuple): shape of the input data.

#     Returns:
#         keras.models.Model: CNN model.
#     """
#     return tf.keras.models.Sequential([
#             tf.keras.layers.Input(input_shape),
            
#             tf.keras.layers.Conv1D(filters=128, kernel_size=8, padding="same"),
#             tf.keras.layers.BatchNormalization(),
#             tf.keras.layers.ReLU(),
            
#             tf.keras.layers.Conv1D(filters=256, kernel_size=5, padding="same"),
#             tf.keras.layers.BatchNormalization(),
#             tf.keras.layers.ReLU(),
            
#             tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding="same"),
#             tf.keras.layers.BatchNormalization(),
#             tf.keras.layers.ReLU(),
            
#             tf.keras.layers.GlobalAveragePooling1D(),
#             tf.keras.layers.Dense(1, activation="sigmoid")
#         ])

# def build_cnn(input_shape):
#     return tf.keras.Sequential([
#         tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(1280, 23, 1)),
#         tf.keras.layers.MaxPooling2D((2, 2)),
        
#         tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
#         tf.keras.layers.MaxPooling2D((2, 2)),
        
#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(32, activation='relu'),
        
#         tf.keras.layers.Dense(1, activation='sigmoid')
#     ])

def build_cnn(input_shape):
    """
    Builds a CNN model.

    Args:
        input_shape (tuple): shape of the input data.

    Returns:
        keras.models.Model: CNN model.
    """
    return tf.keras.models.Sequential([
            tf.keras.layers.Input(input_shape),
            
            tf.keras.layers.Conv1D(filters=128, kernel_size=8, padding="same"),
            # tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(0.5),
            
            # tf.keras.layers.Conv1D(filters=4, kernel_size=16, padding="same"),
            # tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.ReLU(),
            # tf.keras.layers.Dropout(0.5),
            
            # tf.keras.layers.Conv1D(filters=128, kernel_size=8, padding="same"),
            # tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.ReLU(),
            
            # tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding="same"),
            # tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.ReLU(),
            
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])


def build_lstm(input_shape):
    """
    Builds an LSTM model.

    Args:
        input_shape (tuple): shape of the input data.

    Returns:
        keras.models.Model: LSTM model.
    """
    return tf.keras.models.Sequential([
        tf.keras.layers.Input(input_shape),
        tf.keras.layers.LSTM(128),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])


def train(model, training_data, training_label, out_path, file_name, optimizer = "adam", epochs = 500, batch_size = 32, early_stopping = 50):
    """
    Trains a given model using the provided training data and labels.

    Args:
        model (keras.Model): model to be trained.
        training_data (array-like): input training data.
        training_label (array-like): corresponding training labels.
        out_path (str): path to save the trained model.
        optimizer #TODO:aggiungi documentazione
        epochs (int, optional): number of epochs to train the model. Defaults to 500.
        batch_size (int, optional): batch size for training. Defaults to 32.
        early_stopping (int, optional): number of epochs to wait before early stopping. Defaults to 50.

    Returns:
        dictionary: training history object.
    """
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            f'{out_path}/{file_name}.keras', save_best_only=True, monitor="val_loss", verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience = early_stopping, verbose=1)
    ]

    model.compile(
        optimizer = optimizer,
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