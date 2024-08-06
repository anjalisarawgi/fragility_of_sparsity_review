import pandas as pd
import numpy as np
from keras.datasets import mnist
import os


(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("x_train.shape", x_train.shape)
print("y_train.shape", y_train.shape)

x_train_flattened = x_train.reshape(-1, 28 * 28)
print("x_train_flattened.shape", x_train_flattened.shape)
x_test_flattened = x_test.reshape(-1, 28 * 28)
print("x_test_flattened.shape", x_test_flattened.shape)

x_train_flattened = x_train_flattened.astype('float32') / 255.0
x_test_flattened = x_test_flattened.astype('float32') / 255.0

df_train = pd.DataFrame(x_train_flattened)
df_train['label'] = y_train

df_test = pd.DataFrame(x_test_flattened)
df_test['label'] = y_test

os.makedirs('Data/MNIST', exist_ok=True)

# Save DataFrames to CSV files
df_train.to_csv('Data/MNIST/mnist_train.csv', index=False)
df_test.to_csv('Data/MNIST/mnist_test.csv', index=False)

print("CSV files have been saved to the 'Data/MNIST/' directory.")