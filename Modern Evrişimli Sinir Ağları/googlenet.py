import tensorflow as tf

# Veri
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# Inception Block
class Inception(tf.keras.layers.Layer):
    def __init__(self, c1, c2, c3, c4):
        super().__init__()
        self.p1 = tf.keras.layers.Conv2D(c1, 1, activation='relu')

        self.p2_1 = tf.keras.layers.Conv2D(c2[0], 1, activation='relu')
        self.p2_2 = tf.keras.layers.Conv2D(c2[1], 3, padding='same', activation='relu')

        self.p3_1 = tf.keras.layers.Conv2D(c3[0], 1, activation='relu')
        self.p3_2 = tf.keras.layers.Conv2D(c3[1], 5, padding='same', activation='relu')

        self.p4_1 = tf.keras.layers.MaxPool2D(3, strides=1, padding='same')
        self.p4_2 = tf.keras.layers.Conv2D(c4, 1, activation='relu')

    def call(self, x):
        p1 = self.p1(x)
        p2 = self.p2_2(self.p2_1(x))
        p3 = self.p3_2(self.p3_1(x))
        p4 = self.p4_2(self.p4_1(x))
        return tf.concat([p1, p2, p3, p4], axis=-1)

# model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPool2D(2),

    Inception(16, (16, 24), (8, 8), 8),
    tf.keras.layers.MaxPool2D(2),

    Inception(32, (32, 48), (8, 16), 16),

    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(x_train, y_train, epochs=3, batch_size=64)

# Test
model.evaluate(x_test, y_test)