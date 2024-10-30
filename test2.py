from scripts.run_preprocessing import get_dataset
from src.visualization import visualize

import pandas as pd
import keras
import tensorflow as tf

patient_id = 'chb01'
dataset = get_dataset(patient_id)
y_train = dataset['train_labels']
y_test = dataset['test_labels']

x_train = pd.read_csv(f'{patient_id}_x_train.csv')
x_test = pd.read_csv(f'{patient_id}_x_test.csv')




# model = keras.Sequential([
#     keras.layers.Dense(32, activation='relu', input_shape=(x_train.shape[1],)),  # Primo livello hidden
#     keras.layers.Dense(16, activation='relu'),  # Secondo livello hidden
#     keras.layers.Dense(1, activation='sigmoid')  # Livello di output per classificazione binaria
# ])

model = keras.models.Sequential([
        keras.layers.Input(shape=(x_train.shape[1],)),
            
        keras.layers.Dense(512, activation='relu' ),
        # keras.layers.Dropout(0.1),
        
        # keras.layers.Dense(500, activation='relu'),
        # keras.layers.Dropout(0.2),
        
        # keras.layers.Dense(500, activation='relu'),
        # keras.layers.Dropout(0.2),
        
        # keras.layers.Dense(500, activation='relu'),
        # keras.layers.Dropout(0.3),
        
        keras.layers.Dense(1, activation='sigmoid')
    ])


callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        f'{patient_id}.keras', save_best_only=True, monitor="val_loss", verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience = 100, verbose=1)
]
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, callbacks=callbacks, epochs=500, batch_size=32, validation_split=0.2, shuffle=True)

visualize.plot_all_metrics(history, f'{patient_id}', patient_id)

model = keras.models.load_model(f'{patient_id}.keras')

evaluation = model.evaluate(x_test, y_test, return_dict = True)
print(evaluation)