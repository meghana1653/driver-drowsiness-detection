
from tensorflow.keras import layers, models, Input

def simple_resnet(input_shape=(64, 64, 3), num_classes=4):
    def residual_block(x, filters):
        shortcut = x
        x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([shortcut, x])
        x = layers.Activation('relu')(x)
        return x

    inputs = Input(shape=input_shape)

    x = layers.Conv2D(32, (7, 7), strides=2, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = residual_block(x, 32)
    x = residual_block(x, 32)

    x = layers.Conv2D(64, (3, 3), strides=2, padding='same', activation='relu')(x)
    x = residual_block(x, 64)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model
