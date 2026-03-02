import pickle
import numpy as np
import os
import matplotlib.pyplot as plt


class CIFAR10KNN:

    def __init__(self, dataset_dir, num_train=5000, num_test=500):
        self.dataset_dir = dataset_dir
        self.num_train = num_train
        self.num_test = num_test

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    # CIFAR-10 BATCH OKUMA
    def load_batch(self, file):
        with open(file, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
            X = batch[b'data']
            y = batch[b'labels']
            return X, y

    # DATASET YÜKLE
    def load_data(self):

        X_train = []
        y_train = []

        for i in range(1, 6):
            X, y = self.load_batch(
                os.path.join(self.dataset_dir, f"data_batch_{i}")
            )
            X_train.append(X)
            y_train.extend(y)

        X_train = np.concatenate(X_train)
        y_train = np.array(y_train)

        X_test, y_test = self.load_batch(
            os.path.join(self.dataset_dir, "test_batch")
        )

        self.X_train = X_train[:self.num_train].astype(np.float32)
        self.y_train = y_train[:self.num_train]

        self.X_test = X_test[:self.num_test].astype(np.float32)
        self.y_test = y_test[:self.num_test]

        print("Train:", self.X_train.shape)
        print("Test:", self.X_test.shape)

    # MESAFE
    def compute_distance(self, X, x, metric="L2"):

        if metric == "L1":
            return np.sum(np.abs(X - x), axis=1)

        return np.sqrt(np.sum((X - x) ** 2, axis=1))

    # TAHMİN
    def predict(self, X, k=1, metric="L2"):

        num_test = X.shape[0]
        preds = np.zeros(num_test)

        for i in range(num_test):
            distances = self.compute_distance(self.X_train, X[i], metric)

            idx = np.argsort(distances)[:k]
            labels = self.y_train[idx]

            values, counts = np.unique(labels, return_counts=True)
            preds[i] = values[np.argmax(counts)]

        return preds

    # ACCURACY
    def accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)

    # TRAIN + TEST ACCURACY (grafik için gerekli)
    def evaluate_k_values(self, k_values):

        results = {}

        for k in k_values:
            print(f"k={k} test ediliyor...")

            # train accuracy (subset ile hızlı)
            train_pred = self.predict(self.X_train[:200], k)
            train_acc = self.accuracy(self.y_train[:200], train_pred)

            # test accuracy
            test_pred = self.predict(self.X_test, k)
            test_acc = self.accuracy(self.y_test, test_pred)

            results[k] = (train_acc, test_acc)

        return results

    # TABLO
    def print_results_table(self, k_values, results):

        print("\n===== SONUÇ TABLOSU =====")
        print("k\tTrain\tTest")

        for k in k_values:
            train_acc, test_acc = results[k]
            print(f"{k}\t{train_acc:.4f}\t{test_acc:.4f}")

    # TRAIN vs TEST GRAFİK
    def plot_train_test(self, k_values, results):

        train_scores = [results[k][0] * 100 for k in k_values]
        test_scores = [results[k][1] * 100 for k in k_values]

        plt.figure()
        plt.plot(k_values, train_scores, marker='o', label="Train Accuracy")
        plt.plot(k_values, test_scores, marker='s', label="Test Accuracy")

        plt.xlabel("k")
        plt.ylabel("Accuracy (%)")
        plt.title("kNN Train vs Test Accuracy")
        plt.legend()
        plt.grid(True)
        plt.show()

    # SINIF BAZINDA ACCURACY
    def class_wise_accuracy(self, y_pred):

        num_classes = 10
        acc = np.zeros(num_classes)

        for c in range(num_classes):
            idx = (self.y_test == c)
            if np.sum(idx) > 0:
                acc[c] = np.mean(y_pred[idx] == self.y_test[idx])

        return acc


# ÇALIŞTIRMA
if __name__ == "__main__":

    dataset_path = r"C:\PythonProject2\cifar-10-batches-py"

    model = CIFAR10KNN(dataset_path)
    model.load_data()

    k_values = [1, 3, 5, 7]

    results = model.evaluate_k_values(k_values)

    model.print_results_table(k_values, results)

    model.plot_train_test(k_values, results)

    # class-wise accuracy örnek
    y_pred = model.predict(model.X_test, k=3)
    class_acc = model.class_wise_accuracy(y_pred)

    print("\nSınıf bazında accuracy:")
    print(class_acc)