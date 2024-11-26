import tensorflow as tf


def cnn(training_data, training_label, out_path, file_name, epochs=500, batch_size=64, early_stopping=50):
    """
    Train a Convolutional Neural Network model on the given data.

    Args:
        training_data (np.array): Data to train the model on.
        training_label (np.array): Labels for the training data.
        out_path (str): Path to save the model.
        file_name (str): Name of the file to save the model.
        epochs (int, optional): _description_. Defaults to 500.
        batch_size (int, optional): _description_. Defaults to 64.
        early_stopping (int, optional): _description_. Defaults to 50.

    Returns:
        history: Training history.
    """
    model = tf.keras.models.Sequential([
            tf.keras.layers.Input(training_data[0].shape),
            
            tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Conv1D(filters=4, kernel_size=32, padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])
    
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
        optimizer='adam',
        loss = 'binary_crossentropy',
        metrics = ['accuracy']
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