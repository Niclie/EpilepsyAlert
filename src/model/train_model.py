from keras.api.models import Sequential
from keras.api.layers import Input, Dense, Conv1D, Dropout, GlobalAveragePooling1D, BatchNormalization, Activation
from keras.api.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


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
    model = Sequential([
        Input(shape=training_data[0].shape),

        Conv1D(filters=64, kernel_size=16, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.3),

        Conv1D(filters=32, kernel_size=8, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.2),

        GlobalAveragePooling1D(),

        Dense(64, activation='relu'),
        Dropout(0.4),
        Dense(32, activation='relu'),
        Dropout(0.3),

        Dense(1, activation='sigmoid')
    ])

    callbacks = [
        ModelCheckpoint(
            f'{out_path}/{file_name}.keras', save_best_only=True, monitor="val_loss", verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
        ),
        EarlyStopping(
            monitor="val_loss", patience=early_stopping, verbose=1)
    ]

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
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