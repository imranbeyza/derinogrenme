import tensorflow as tf

# =========================
# VERİ
# =========================
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

# channel ekle
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# =========================
# BatchNorm (Custom - DOĞRU)
# =========================
class BatchNorm(tf.keras.layers.Layer):
    def __init__(self, momentum=0.9, eps=1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps

    def build(self, input_shape):
        dim = input_shape[-1]

        self.gamma = self.add_weight(shape=(dim,), initializer="ones", trainable=True)
        self.beta = self.add_weight(shape=(dim,), initializer="zeros", trainable=True)

        self.moving_mean = self.add_weight(shape=(dim,), initializer="zeros", trainable=False)
        self.moving_var = self.add_weight(shape=(dim,), initializer="ones", trainable=False)

    def call(self, x, training=False):

        # 🔥 Conv vs Dense ayrımı
        if len(x.shape) == 4:
            axes = [0, 1, 2]   # Conv
        else:
            axes = [0]         # Dense

        if training:
            mean = tf.reduce_mean(x, axis=axes)
            var = tf.reduce_mean(tf.square(x - mean), axis=axes)

            self.moving_mean.assign(self.momentum * self.moving_mean + (1 - self.momentum) * mean)
            self.moving_var.assign(self.momentum * self.moving_var + (1 - self.momentum) * var)
        else:
            mean = self.moving_mean
            var = self.moving_var

        x_norm = (x - mean) / tf.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta

# =========================
# MODEL (LeNet + BN)
# =========================
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(6, 5, input_shape=(28,28,1)),
    BatchNorm(),
    tf.keras.layers.Activation('sigmoid'),
    tf.keras.layers.AvgPool2D(2),

    tf.keras.layers.Conv2D(16, 5),
    BatchNorm(),
    tf.keras.layers.Activation('sigmoid'),
    tf.keras.layers.AvgPool2D(2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(120),
    BatchNorm(),
    tf.keras.layers.Activation('sigmoid'),

    tf.keras.layers.Dense(84),
    BatchNorm(),
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
model.fit(x_train, y_train, epochs=3, batch_size=64)

# =========================
# TEST
# =========================
model.evaluate(x_test, y_test)

# =========================
# GAMMA & BETA
# =========================
bn_layer = model.layers[1]

print("Gamma:", bn_layer.gamma.numpy())
print("Beta:", bn_layer.beta.numpy())