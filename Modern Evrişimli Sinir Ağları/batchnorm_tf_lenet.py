import tensorflow as tf

# =========================
# VERİ
# =========================
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# =========================
# MODEL (LeNet + TF BatchNorm)
# =========================
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(6, 5, input_shape=(28,28,1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('sigmoid'),
    tf.keras.layers.AvgPool2D(2),

    tf.keras.layers.Conv2D(16, 5),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('sigmoid'),
    tf.keras.layers.AvgPool2D(2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(120),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('sigmoid'),

    tf.keras.layers.Dense(84),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('sigmoid'),

    tf.keras.layers.Dense(10, activation='softmax')
])

# =========================
# COMPILE
# =========================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# =========================
# TRAIN
# =========================
history = model.fit(
    x_train, y_train,
    epochs=3,
    batch_size=64,
    validation_data=(x_test, y_test)
)

# =========================
# TEST
# =========================
model.evaluate(x_test, y_test)
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='test acc')
plt.xlabel("epoch")
plt.legend()
plt.grid()
plt.show()