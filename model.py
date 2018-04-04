import numpy as np
import keras


def hello_world():
    print("hello world, my name is charles.")


def fft_features(data):
    hamming = np.hamming(len(data))

    f_x = np.abs(np.fft.fft(data[:, 0])) ** 2
    f_x = f_x * hamming
    f_x = f_x[1:]

    f_y = np.abs(np.fft.fft(data[:, 1])) ** 2
    f_y = f_y * hamming
    f_y = f_y[1:]

    f_z = np.abs(np.fft.fft(data[:, 2])) ** 2
    f_z = f_z * hamming
    f_z = f_z[1:]

    f_t = data[:, 3]
    f_t = f_t[1:]

    f = np.array([max(x_i, y_i, z_i) for x_i, y_i, z_i in zip(f_x, f_y, f_z)])
    f_mean = np.mean(f)
    f_median = np.median(f)
    f_std = np.std(f)
    f_sum = np.sum(f)
    f_min = np.min(f)
    f_max = np.max(f)
    f_range = f_max - f_min
    f_der = np.diff(f) / np.diff(f_t)
    f_ratios = f / f_sum
    f_peaks = -np.partition(-f, 5)[:5]

    f_features_extended = [f_mean, f_median, f_std, f_sum, f_min, f_max, f_range]
    f_features_extended.extend(f_der)
    f_features_extended.extend(f_ratios)
    f_features_extended.extend(f_peaks)

    data_fft_features = np.array([f_features_extended])

    return data_fft_features


model = None


def load_model():
    global model
    model = keras.models.load_model("D:\\p_eaglesense\\eaglesense\\models\\v2\\3-gesture-xyz-nn-overlap.h5")


def predict(X):
    y_pred = model.predict(np.array([X]))
    y_output = np.array([np.argmax(i) for i in y_pred])
    return y_output
