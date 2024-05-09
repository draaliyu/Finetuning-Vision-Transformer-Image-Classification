# finetune_model.py
import tensorflow as tf
import tensorflow_addons as tfa
from vit_keras import vit

IMAGE_SIZE = 224
EPOCHS = 2


def build_model(num_classes):
    vit_model = vit.vit_b32(
        image_size=IMAGE_SIZE,
        activation='softmax',
        pretrained=True,
        include_top=False,
        pretrained_top=False,
        classes=num_classes
    )

    model = tf.keras.Sequential([
        vit_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(11, activation=tfa.activations.gelu),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ], name='vision_transformer')

    return model


def compile_model(model):
    learning_rate = 1e-4
    optimizer = tfa.optimizers.RectifiedAdam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2),
                  metrics=['accuracy'])
    return model


def train_model(model, train_dataset, val_dataset):
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                                     factor=0.2,
                                                     patience=2,
                                                     verbose=1,
                                                     min_delta=1e-4,
                                                     min_lr=1e-6,
                                                     mode='max')

    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                     min_delta=1e-4,
                                                     patience=5,
                                                     mode='max',
                                                     restore_best_weights=True,
                                                     verbose=1)

    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='./model.hdf5',
                                                      monitor='val_accuracy',
                                                      verbose=1,
                                                      save_best_only=True,
                                                      save_weights_only=True,
                                                      mode='max')

    callbacks = [earlystopping, reduce_lr, checkpointer]

    history = model.fit(train_dataset,
                        validation_data=val_dataset,
                        epochs=EPOCHS,
                        callbacks=callbacks)

    model.save_weights('model_weights.h5')

    return history
