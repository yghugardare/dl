# dl

Boston - 
```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from sklearn import preprocessing

(X_train, Y_train), (X_test, Y_test) = keras.datasets.boston_housing.load_data()

print("Training data shape:", X_train.shape)
print("Test data shape:", X_test.shape)
print("Train output data shape:", Y_train.shape)
print("Actual Test output data shape:", Y_test.shape)

##Normalize the data

X_train=preprocessing.normalize(X_train)
X_test=preprocessing.normalize(X_test)

#Model Building

X_train[0].shape
model = Sequential()
model.add(Dense(128,activation='relu',input_shape= X_train[0].shape))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(1))

model.summary()

model.compile(loss='mse',optimizer='rmsprop',metrics=['mae'])

history = model.fit(X_train,Y_train,epochs=100,batch_size=1,verbose=1,validation_data=(X_test,Y_test))

results = model.evaluate(X_test, Y_test)
print(results)

```

## fASHION -

Absolutely! Let's analyze this code that trains a neural network to classify clothing images from the Fashion-MNIST dataset.

**Deep Learning Concepts/Libraries**

* **TensorFlow/Keras:** A powerful deep learning framework.  You're using the high-level Keras API to build the model.
* **Fashion-MNIST:** A common dataset of clothing images (grayscale, 28x28 pixels) with labels corresponding to clothing types.
* **Image Classification:** The task of assigning a class label to an image.
* **Convolutional Neural Networks (CNNs):** Specialized neural networks often used for image tasks, but not used in this simple example.

**Code Breakdown**

**1.  Import Libraries**

```python
from tensorflow import keras
import numpy as np 
import matplotlib.pyplot as plt 
```

*  **`keras`:**  Provides tools for building and training neural networks.
*  **`numpy`:**  Enables numerical computations (arrays, etc.).
*  **`matplotlib`:**  Lets you  visualize the results.

**2. Load/Preprocess Data**

```python
fashion_mnist = keras.datasets.fashion_mnist
(train_img, train_labels), (test_img, test_labels) = fashion_mnist.load_data()
train_img = train_img / 255.0
test_img = test_img / 255.0
```

* **Loads the Fashion-MNIST dataset:** Splits it into training and test sets.
* **Normalization:** Scales pixels values to be between 0 and 1 for better numerical stability during training. 

**3. Build the Model**

```python
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), 
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
```

* **Sequential Model:**  Indicates a simple linear stack of layers.
* **Flatten:** Takes the 2D image and converts it into a 1D array to feed into dense layers.
* **Dense Layers:** Fully connected layers. The first has 128 neurons with 'relu' activation; the output layer has 10 (number of classes) with 'softmax' activation to output probabilities.

**4.  Compile the Model**

```python
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

* **Optimizer:** 'adam' is a popular optimization algorithm for updating weights.
* **Loss:** 'sparse_categorical_crossentropy' is suitable when labels are integers (not one-hot encoded).
* **Metrics:** 'accuracy' is tracked during training.

**5. Train the Model**

```python
model.fit(train_img, train_labels, epochs=10)
```

* **`model.fit`:** Initiates the training process. The model learns patterns from the training data over 10 'epochs' (iterations over the entire dataset). 

**6. Evaluate the Model**

```python
test_loss, test_acc = model.evaluate(test_img, test_labels)
print("accuracy of tessting: ",test_acc)
```

* **`model.evaluate`:** Calculates the loss and accuracy on the held-out test set.

**7. Make Predictions and Visualize**

```python
# ... (See detailed explanation below)
```

* **`model.predict`:** Generates predictions (probabilities for each class) on the test images.
* **Visualization:** The code creates a grid of images, displaying their true labels and the prediction probabilities.

Let me know if you would like a deeper explanation of a particular section or concept!



```py
from tensorflow import keras
import numpy as np 
import matplotlib.pyplot as plt 

fashion_mnist = keras.datasets.fashion_mnist
(train_img, train_labels), (test_img, test_labels) = fashion_mnist.load_data()

train_img = train_img / 255.0
test_img = test_img / 255.0

model = keras.Sequential([keras.layers.Flatten(input_shape=(28, 28)), 
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', 
    metrics=['accuracy'])

model.fit(train_img, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_img, test_labels)
print("accuracy of tessting: ",test_acc)

predictions = model.predict(test_img)

predicted_labels = np.argmax(predictions, axis=1)

num_rows = 5 
num_cols = 5
num_imgs = num_rows*num_cols

plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_imgs):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plt.imshow(test_img[1], cmap='gray')
    plt.axis("off")
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plt.bar(range(10), predictions[i])
    plt.xticks(range(10))
    plt.ylim([0,1])
    plt.tight_layout()
    plt.title(f"predicted_labels: {predicted_labels[i]}")
    plt.show()
```

imdb  - 


Absolutely! Let's break down the provided code, focusing on the key concepts and libraries involved in this basic movie review classification task.

**Deep Learning Concepts/Libraries**

* **Keras:** A high-level deep learning API providing user-friendly ways to build and train neural networks.
* **IMDB Dataset:** A built-in dataset in Keras containing movie reviews labeled as positive or negative (sentiment classification).
* **Neural Network:** The core computational model used for learning patterns in the data.
* **Layers:** Building blocks of a neural network. Here, you're using 'Dense' (fully connected) layers.
* **Activation Functions:** Non-linear functions (like 'relu', 'sigmoid') applied to outputs of layers, enabling the network to learn complex patterns.

**Code Breakdown**

**1. Data Loading**

```python
from keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
```

* **Loads the IMDB dataset:** Includes movie reviews (text) and labels (0 for negative, 1 for positive).
* **`num_words=10000`:** Limits the vocabulary to the 10000 most frequent words.

**2. Data Exploration**

```python
max([max(sequence) for sequence in train_data]) 
```

* **Finds the maximum word index:** Explores the range of word indices present in the reviews after limiting the vocabulary.

**3. Preprocessing**

```python
word_index = imdb.get_word_index()
reverse_word_index = dict([(val, key) for (key, val) in word_index.items()])
decoded_review = ' '.join([reverse_word_index.get(i-3, '?') for i in train_data[0]])

def vectorize(sequences, dimension=10000):
    # ... (see function explanation below)
```

* **`word_index`:** A dictionary mapping words to their integer indices.
* **`reverse_word_index`:**  A reverse mapping from integer indices to words.
* **`decoded_review`:**  Reconstructs a sample review from its word indices (likely for understanding the data).
* **`vectorize`:** This function converts lists of word indices (reviews) into one-hot encoded vectors, a suitable format for the neural network.  

**4. Model Creation**

```python
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
```

* **`Sequential` model**:  A linear stack of layers.
* **Dense Layers:** Fully connected layers, each with 16 units (neurons).
* **Activation:** 'relu' for the hidden layers, 'sigmoid' for the output (suitable for binary classification).

**5. Compilation**

```python
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
```

* **`loss`:** Loss function for optimization (binary cross-entropy for binary problems).
* **`optimizer`:** Algorithm for updating model weights (RMSProp is a common choice).
* **`metrics`:** 'accuracy' is tracked during training.

**6. Training and Validation**

```python
# ... (split data into training/validation sets) 
history = model.fit(partial_x, partial_y, epochs=20, batch_size=512, validation_data=(x_val, y_val))
```

* **`model.fit`:** Trains the model, adjusting weights based on the training data.
* **`epochs`:**  Number of passes through the entire training set.
* **`batch_size`:** How many samples are used per update.
* **`validation_data`:**  Set aside for monitoring performance during training.

**7. Evaluation**

```python
results = model.evaluate(x_test, y_test) 
print(results)
```

* **`model.evaluate`:** Calculates the loss and accuracy on the unseen test data. 

Let me know if you want a deeper dive into any specific aspect! 


