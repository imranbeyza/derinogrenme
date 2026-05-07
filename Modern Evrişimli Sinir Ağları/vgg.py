import tensorflow as tf

# Veri seti
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Normalize + reshape
x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# VGG block
def vgg_block(num_convs, num_channels):
    blk = tf.keras.models.Sequential()
    for _ in range(num_convs):
        blk.add(tf.keras.layers.Conv2D(num_channels, 3, padding='same', activation='relu'))
    blk.add(tf.keras.layers.MaxPool2D(2))
    return blk


conv_arch = ((1, 32), (1, 64), (2, 128))

def vgg():
    net = tf.keras.models.Sequential()

    for (num_convs, num_channels) in conv_arch:
        net.add(vgg_block(num_convs, num_channels))

    net.add(tf.keras.layers.Flatten())
    net.add(tf.keras.layers.Dense(128, activation='relu'))
    net.add(tf.keras.layers.Dropout(0.3))
    net.add(tf.keras.layers.Dense(10, activation='softmax'))

    return net

model = vgg()

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