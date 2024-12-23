from keras.api.models import Sequential
from keras.api.layers import Input, Dense, Conv1D, Dropout, GlobalAveragePooling1D, BatchNormalization, Activation
from keras.api.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


def load_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),

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

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_model(model, training_data, training_label, out_path, batch_size, epochs, early_stopping):
    callbacks = [
        ModelCheckpoint(
            f'{out_path}.keras', save_best_only=True, monitor='val_loss', verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=20, min_lr=0.0001
        ),
        EarlyStopping(
            monitor='val_loss', patience=early_stopping, verbose=1)
    ]

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