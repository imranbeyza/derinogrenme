import tensorflow as tf

# Veri seti
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

# Channel ekle
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# NiN block
def nin_block(num_channels, kernel_size, strides, padding):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(num_channels, kernel_size,
                               strides=strides, padding=padding, activation='relu'),
        tf.keras.layers.Conv2D(num_channels, 1, activation='relu'),
        tf.keras.layers.Conv2D(num_channels, 1, activation='relu')
    ])

# NiN modeli
def build_model():
    model = tf.keras.Sequential([
        nin_block(32, 3, 1, 'same'),
        tf.keras.layers.MaxPool2D(2),

        nin_block(64, 3, 1, 'same'),
        tf.keras.layers.MaxPool2D(2),

        nin_block(128, 3, 1, 'same'),
        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

model = build_model()

# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(x_train, y_train, epochs=3, batch_size=64)

# Test
model.evaluate(x_test, y_test)