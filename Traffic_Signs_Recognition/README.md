
# Self-Driving Car Engineer Nanodegree

## Deep Learning

## Project: Build a Traffic Sign Recognition Classifier

---
### Updates in the second submission


I added 7 more captured images at the last section. And I realized that in the previous submission, when I tested the model on the captured images, I forgot to normalize the images. So I normalized the test images this time and I got better results than before.

---

In this notebook, a template is provided for you to implement your functionality in stages which is required to successfully complete this project. If additional code is required that cannot be included in the notebook, be sure that the Python code is successfully imported and included in your submission, if necessary. Sections that begin with **'Implementation'** in the header indicate where you should begin your implementation for your project. Note that some sections of implementation are optional, and will be marked with **'Optional'** in the header.

In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.

>**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

---

## Step 1: Dataset Exploration

Visualize the German Traffic Signs Dataset. This is open ended, some suggestions include: plotting traffic signs images, plotting the count of each sign, etc. Be creative!


The pickled data is a dictionary with 4 key/value pairs:

- features -> the images pixel values, (width, height, channels)
- labels -> the label of the traffic sign
- sizes -> the original width and height of the image, (width, height)
- coords -> coordinates of a bounding box around the sign in the image, (x1, y1, x2, y2)


```python
# Load pickled data
import pickle

# TODO: fill this in based on where you saved the training and testing data
training_file = "./lab_2_data/train.p"
testing_file = "./lab_2_data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']
```


```python
### To start off let's do a basic data summary.
import cv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer
from IPython.display import display, Image
%matplotlib inline

# TODO: number of training examples
n_train = X_train.shape[0]

# TODO: number of testing examples
n_test = X_test.shape[0]

# TODO: what's the shape of an image?
image_shape = X_train[0].shape

# TODO: how many classes are in the dataset
n_classes = len(np.unique(y_test))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
```

    Number of training examples = 39209
    Number of testing examples = 12630
    Image data shape = (32, 32, 3)
    Number of classes = 43



```python
# Print out Signs corresponding to the class number.
for i in range(n_classes):
    print("Class:",i)
    plt.imshow(X_train[np.argwhere(y_train==i)[20][0]])
    plt.show()
```

    Class: 0



![png](output_4_1.png)


    Class: 1



![png](output_4_3.png)


    Class: 2



![png](output_4_5.png)


    Class: 3



![png](output_4_7.png)


    Class: 4



![png](output_4_9.png)


    Class: 5



![png](output_4_11.png)


    Class: 6



![png](output_4_13.png)


    Class: 7



![png](output_4_15.png)


    Class: 8



![png](output_4_17.png)


    Class: 9



![png](output_4_19.png)


    Class: 10



![png](output_4_21.png)


    Class: 11



![png](output_4_23.png)


    Class: 12



![png](output_4_25.png)


    Class: 13



![png](output_4_27.png)


    Class: 14



![png](output_4_29.png)


    Class: 15



![png](output_4_31.png)


    Class: 16



![png](output_4_33.png)


    Class: 17



![png](output_4_35.png)


    Class: 18



![png](output_4_37.png)


    Class: 19



![png](output_4_39.png)


    Class: 20



![png](output_4_41.png)


    Class: 21



![png](output_4_43.png)


    Class: 22



![png](output_4_45.png)


    Class: 23



![png](output_4_47.png)


    Class: 24



![png](output_4_49.png)


    Class: 25



![png](output_4_51.png)


    Class: 26



![png](output_4_53.png)


    Class: 27



![png](output_4_55.png)


    Class: 28



![png](output_4_57.png)


    Class: 29



![png](output_4_59.png)


    Class: 30



![png](output_4_61.png)


    Class: 31



![png](output_4_63.png)


    Class: 32



![png](output_4_65.png)


    Class: 33



![png](output_4_67.png)


    Class: 34



![png](output_4_69.png)


    Class: 35



![png](output_4_71.png)


    Class: 36



![png](output_4_73.png)


    Class: 37



![png](output_4_75.png)


    Class: 38



![png](output_4_77.png)


    Class: 39



![png](output_4_79.png)


    Class: 40



![png](output_4_81.png)


    Class: 41



![png](output_4_83.png)


    Class: 42



![png](output_4_85.png)



```python
### Data exploration visualization goes here.
### Feel free to use as many code cells as needed.
import random

### Print out the number of each class from the train set
count_train = []
for i in range(n_classes):
    count = len(y_train[y_train==i])
    print("Number of Class", i, ":", count)
    count_train.append(count)
```

    Number of Class 0 : 210
    Number of Class 1 : 2220
    Number of Class 2 : 2250
    Number of Class 3 : 1410
    Number of Class 4 : 1980
    Number of Class 5 : 1860
    Number of Class 6 : 420
    Number of Class 7 : 1440
    Number of Class 8 : 1410
    Number of Class 9 : 1470
    Number of Class 10 : 2010
    Number of Class 11 : 1320
    Number of Class 12 : 2100
    Number of Class 13 : 2160
    Number of Class 14 : 780
    Number of Class 15 : 630
    Number of Class 16 : 420
    Number of Class 17 : 1110
    Number of Class 18 : 1200
    Number of Class 19 : 210
    Number of Class 20 : 360
    Number of Class 21 : 330
    Number of Class 22 : 390
    Number of Class 23 : 510
    Number of Class 24 : 270
    Number of Class 25 : 1500
    Number of Class 26 : 600
    Number of Class 27 : 240
    Number of Class 28 : 540
    Number of Class 29 : 270
    Number of Class 30 : 450
    Number of Class 31 : 780
    Number of Class 32 : 240
    Number of Class 33 : 689
    Number of Class 34 : 420
    Number of Class 35 : 1200
    Number of Class 36 : 390
    Number of Class 37 : 210
    Number of Class 38 : 2070
    Number of Class 39 : 300
    Number of Class 40 : 360
    Number of Class 41 : 240
    Number of Class 42 : 240



```python
### Print out the number of each class from the test set
count_test = []
for i in range(n_classes):
    count = len(y_test[y_test==i])
    print("Number of Class ", i, ":", count)
    count_test.append(count)
```

    Number of Class  0 : 60
    Number of Class  1 : 720
    Number of Class  2 : 750
    Number of Class  3 : 450
    Number of Class  4 : 660
    Number of Class  5 : 630
    Number of Class  6 : 150
    Number of Class  7 : 450
    Number of Class  8 : 450
    Number of Class  9 : 480
    Number of Class  10 : 660
    Number of Class  11 : 420
    Number of Class  12 : 690
    Number of Class  13 : 720
    Number of Class  14 : 270
    Number of Class  15 : 210
    Number of Class  16 : 150
    Number of Class  17 : 360
    Number of Class  18 : 390
    Number of Class  19 : 60
    Number of Class  20 : 90
    Number of Class  21 : 90
    Number of Class  22 : 120
    Number of Class  23 : 150
    Number of Class  24 : 90
    Number of Class  25 : 480
    Number of Class  26 : 180
    Number of Class  27 : 60
    Number of Class  28 : 150
    Number of Class  29 : 90
    Number of Class  30 : 150
    Number of Class  31 : 270
    Number of Class  32 : 60
    Number of Class  33 : 210
    Number of Class  34 : 120
    Number of Class  35 : 390
    Number of Class  36 : 120
    Number of Class  37 : 60
    Number of Class  38 : 690
    Number of Class  39 : 90
    Number of Class  40 : 90
    Number of Class  41 : 60
    Number of Class  42 : 90



```python
# Plot the graph showing the count in the Train Data set
plt.plot(count_train)
plt.title("Number of Classes in Train Data Set")
plt.show()
```


![png](output_7_0.png)



```python
# Plot the graph showing the count in the Test Data set
plt.plot(count_test)
plt.title("Number of Classes in Test Data Set")
plt.show()
```


![png](output_8_0.png)


The count of each class is irregular. Some of them are very high, and the others are very low. I expect that this irregularity of the data might cause some trouble when I proceed the deep learning on this data.

----

## Step 2: Design and Test a Model Architecture

Design and implement a deep learning model that learns to recognize traffic signs. Train and test your model on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

There are various aspects to consider when thinking about this problem:

- Your model can be derived from a deep feedforward net or a deep convolutional network.
- Play around preprocessing techniques (normalization, rgb to grayscale, etc)
- Number of examples per label (some have more than others).
- Generate fake data.

Here is an example of a [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). It's not required to be familiar with the approach used in the paper but, it's good practice to try to read papers like these.

### Implementation

Use the code cell (or multiple code cells, if necessary) to implement the first step of your project. Once you have completed your implementation and are satisfied with the results, be sure to thoroughly answer the questions that follow.


```python
### Preprocess the data here.
### Feel free to use as many code cells as needed.

# Implement Min-Max scaling for image data
def normalize(image_data):
    a = 0.01
    b = 0.99
    color_min = 0.0
    color_max = 255.0
    
    return a + ( ( (image_data - color_min) * (b - a) )/(color_max - color_min))

# Normalize train features and test features
train_features = normalize(X_train)
test_features = normalize(X_test)
```


```python
# Looking at the original image and normalized image
plt.subplot(121)
plt.imshow(X_train[15])
plt.title("Original Image")
plt.subplot(122)
plt.title("Normalized Image")
plt.imshow(train_features[15])
plt.show()
```


![png](output_13_0.png)



```python
# Trake out Green Colors
if train_features.shape[3] != 2:
    train_features = np.copy(train_features[:,:,:,[0,2]])
    
if test_features.shape[3] != 2:
    test_features = np.copy(test_features[:,:,:,[0,2]])

# Print out new shapes
print(train_features.shape)
print(test_features.shape)
```

    (39209, 32, 32, 2)
    (12630, 32, 32, 2)


The features are all normalized to have well-conditioned data set.


```python
# Turn labels into numbers and apply One-Hot Encoding
encoder = LabelBinarizer()
encoder.fit(y_train)
train_labels = encoder.transform(y_train)
test_labels = encoder.transform(y_test)

# Change to float32, so it can be multiplied against the features in TensorFlow, which are float32
# This is the script from https://github.com/udacity/CarND-TensorFlow-Lab/blob/master/lab.ipynb
train_labels = train_labels.astype(np.float32)
test_labels = test_labels.astype(np.float32)

print('Labels One-Hot Encoded')
```

    Labels One-Hot Encoded


### Question 1 

_Describe the techniques used to preprocess the data._

**Answer:**

First, I normalized the image data using min max scaling in order to have well-conditioned data set. When I looked at traffic signs, I noticed that there are no green colored signs, so I decided to take the green out of the color channels. Then, I one-hot encoded the labels sets. Now, labels will be represented as column vectors instead of digits.


```python
### Generate data additional (if you want to!)
### and split the data into training/validation/testing sets here.
### Feel free to use as many code cells as needed.
```


```python
### Generate data to have regular data set.

# Check the maximum number of counts of the classes.
max_count = np.max(count_train)
max_count
```




    2250




```python
import random

def affine_transform(img):
    """
    Applies affine transfrom to the image to get a
    distorted image that is different from the original one.
    """
    points_before = np.float32([[10,10], [10,20], [20,10]])
    points_after = np.float32([[10+int(6*(random.random() - 0.5)), 10+int(6*(random.random() - 0.5))], 
                               [10+int(6*(random.random() - 0.5)), 20+int(6*(random.random() - 0.5))],
                               [20+int(6*(random.random() - 0.5)), 10+int(6*(random.random() - 0.5))]])
    M = cv2.getAffineTransform(points_before, points_after)
    rows, cols, ch = img.shape
    dst = cv2.warpAffine(img, M, (cols,rows))
    return dst

img = train_features[10]
dst = affine_transform(img)
plt.subplot(121)
plt.imshow(img[:,:,0])
plt.title('Input')
plt.subplot(122)
plt.imshow(dst[:,:,0])
plt.title('Output')
```




    <matplotlib.text.Text at 0x1262ad0f0>




![png](output_21_1.png)



```python
# Applying affine transformation to each class image

# Copy the train features to the new train set
train_features2 = np.copy(train_features)
y_train2 = np.copy(y_train)

# Print the shape of train_features2 and y_train2 before transformation
print("Before Transformation")
print("train_feature2 shape:", train_features2.shape)
print("y_train2 shape:", y_train2.shape)
```

    Before Transformation
    train_feature2 shape: (39209, 32, 32, 2)
    y_train2 shape: (39209,)



```python
# Iterate over the classes and append the distorted images 
# The transformation will be applied if the count is under 500.
# If the iteration is done, then all of the classes will have
# at least 500 images.

#Progress bar
images_pbar = tqdm(range(n_classes), desc='Progress', unit='class')

num = 500
for cls in images_pbar:
    if count_train[cls] < num:
        for count in range(num - count_train[cls]):
            rand = int(100*random.random())
            dst = affine_transform(train_features[np.argwhere(y_train==cls)[rand]][0])
            train_features2 = np.append(train_features2, [dst], axis=0)
            y_train2 = np.append(y_train2, [cls], axis=0)
```

    Progress: 100%|██████████| 43/43 [25:57<00:00, 85.41s/class]



```python
# Print the shape of train_features2 and y_train2 after transformation
print("\nAfter Transformation")
print("train_feature2 shape:", train_features2.shape)
print("y_train2 shape:", y_train2.shape)
```

    
    After Transformation
    train_feature2 shape: (42739, 32, 32, 2)
    y_train2 shape: (42739,)



```python
# Count the classes from the feature set after transformation
count_train2 = []
for i in range(n_classes):
    count = len(y_train2[y_train2==i])
    count_train2.append(count)

# Plot the graph showing the count in the Train Data set
plt.plot(count_train2)
plt.title("Number of Classes in Train Data After Transformation")
plt.axis([0, 42, 0, 2400])
plt.show()
```


![png](output_25_0.png)



```python
# Apply One-Hot Encoding to the new labels
train_labels2 = encoder.transform(y_train2)

# Change to float32
train_labels2 = train_labels2.astype(np.float32)

print(train_labels2.shape)
```

    (42739, 43)



```python
# Get randomized datasets for training and validation
train_features2, valid_features, train_labels2, valid_labels = train_test_split(
    train_features2,
    train_labels2,
    test_size=0.05,
    random_state=832289)

print('Training features and labels randomized and split.')
```

    Training features and labels randomized and split.



```python
import os

# Save the data for easy access
pickle_file = 'trafficSigns.pickle'
if not os.path.isfile(pickle_file):
    print('Saving data to pickle file...')
    try:
        with open('trafficSigns.pickle', 'wb') as pfile:
            pickle.dump(
                {
                    'train_dataset': train_features2,
                    'train_labels': train_labels2,
                    'valid_dataset': valid_features,
                    'valid_labels': valid_labels,
                    'test_dataset': test_features,
                    'test_labels': test_labels,
                },
                pfile, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise

print('Data cached in pickle file.')
```

    Saving data to pickle file...
    Data cached in pickle file.


---
# Checkpoint

All of the progress is saved. So when I return I can start from here.


```python
%matplotlib inline

# Load the modules
import pickle
import math

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt

# Reload the data
pickle_file = 'trafficSigns.pickle'
with open(pickle_file, 'rb') as f:
    pickle_data = pickle.load(f)
    train_features = pickle_data['train_dataset']
    train_labels = pickle_data['train_labels']
    valid_features = pickle_data['valid_dataset']
    valid_labels = pickle_data['valid_labels']
    test_features = pickle_data['test_dataset']
    test_labels = pickle_data['test_labels']
    del pickle_data  # Free up memory

print('Data and modules loaded.')
print("train_features size:", train_features.shape)
print("train_labels size:", train_labels.shape)
print("valid_features size:", valid_features.shape)
print("valid_labels size:", valid_labels.shape)
print("test_features size:", test_features.shape)
print("test_labels size:", test_labels.shape)
```

    Data and modules loaded.
    train_features size: (40602, 32, 32, 2)
    train_labels size: (40602, 43)
    valid_features size: (2137, 32, 32, 2)
    valid_labels size: (2137, 43)
    test_features size: (12630, 32, 32, 2)
    test_labels size: (12630, 43)



```python
# number of training examples
n_train = train_features.shape[0]

# number of valid examples
n_valid = valid_features.shape[0]

# number of testing examples
n_test = test_features.shape[0]

# what's the shape of an image?
image_shape = train_features[0].shape

# how many classes are in the dataset
n_classes = train_labels.shape[1]

print("n_train:", n_train)
print("n_valid:", n_valid)
print("n_test:", n_test)
print("image shape:", image_shape)
print("n_classes:", n_classes)
```

    n_train: 40602
    n_valid: 2137
    n_test: 12630
    image shape: (32, 32, 2)
    n_classes: 43


### Question 2

_Describe how you set up the training, validation and testing data for your model. If you generated additional data, why?_

**Answer:**

I decided to generate addiotional data because I believe that the irregularity of the data will cause some issue when I proceed the deep learning to the data set. I used the affine transformations to the randomly chosen images of those classes that contain less than **800** images. For example, if a class *i* contains 700 images, then 100 images are created by affine transformations. As a result, the count of images of each class is at least 800.

After generating addtional data, the train data set was splitted into train and validation set using **train_test_split** from **sklearn** library. I used train_size = **5 %**. Then, I saved these data to pickle files so that I can use them immediately whenever I come back to this jupyter notebook file, rather than running all of the previous scripts from the beginning.


```python
### Define your architecture here.
### Feel free to use as many code cells as needed.
features_count = image_shape[0] * image_shape[1] * image_shape[2] # 32 x 32 x 3
labels_count = len(train_labels[0])

print("features count:", features_count)
print("labels count:", labels_count)
```

    features count: 2048
    labels count: 43



```python
# Reformat the matrix
def reformat(matrix):
    return matrix.reshape((-1, features_count))

train_features = reformat(train_features)
valid_features = reformat(valid_features)
test_features = reformat(test_features)

# Print out shapes
print(train_features.shape)
print(valid_features.shape)
print(test_features.shape)
```

    (40602, 2048)
    (2137, 2048)
    (12630, 2048)


---
# Multilayer Convolutional Network

I will employ the multilayer convolutional network to train the data.


```python
# parameters
depth_conv1 = 32
depth_conv2 = 64
num_hidden = 128

# Set the features and labels tensors
x = tf.placeholder(tf.float32, shape=[None, features_count])
y_ = tf.placeholder(tf.float32, shape=[None, labels_count])

# Weight Initialization
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# Convolutional and Pooling
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

### First Convolutional Layer
W_conv1 = weight_variable([5, 5, image_shape[2], depth_conv1])
b_conv1 = bias_variable([depth_conv1])

x_image = tf.reshape(x, [-1,image_shape[0],image_shape[1],image_shape[2]])

# Applying Relu function
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

### Densely Connected Layer
reduced_size = image_shape[0]//4 * image_shape[1]//4* depth_conv1
W_fc1 = weight_variable([reduced_size, num_hidden])
b_fc1 = bias_variable([num_hidden])

shape = h_pool1.get_shape().as_list()
h_pool1_flat = tf.reshape(h_pool1, [-1, shape[1] * shape[2] * shape[3]])
h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)

### Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

### Readout Layer
W_fc2 = weight_variable([num_hidden, labels_count])
b_fc2 = bias_variable([labels_count])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# softmax function
prediction = tf.nn.softmax(y_conv)

# Loss and Cross entropy
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))

# Determine if the predictions are correct
is_correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

# Calculate the accuracy of the predictions
accuracy = tf.reduce_mean(tf.cast(is_correct_prediction, tf.float32))

# Feed dicts for training, validation, and test session
train_feed_dict = {x: train_features, y_: train_labels, keep_prob: 1.0}
valid_feed_dict = {x: valid_features, y_: valid_labels, keep_prob: 1.0}
test_feed_dict = {x: test_features, y_: test_labels, keep_prob: 1.0}

print("variables set.")
```

    variables set.


### Question 3

_What does your final architecture look like? (Type of model, layers, sizes, connectivity, etc.)  For reference on how to build a deep neural network using TensorFlow, see [Deep Neural Network in TensorFlow
](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/b516a270-8600-4f93-a0a3-20dfeabe5da6/concepts/83a3a2a2-a9bd-4b7b-95b0-eb924ab14432) from the classroom._


**Answer:**

Below is the summary of my architecture.

        - Input
        - Convolutional Layer with depth 32
        - ReLU layer
        - Max Pool 2 x 2
        - Dense Layer (input size: 8 * 8 * 32, output size: 128)
        - Dropout (Probability 0.5)
        - Dense Layer (input size: 128, output size: 43)

I used *Dropout* because the size of the data is huge so the course recommended to use *Dropout* to avoid **overfitting**. 

At first, I tried to use two convolutional layers, but it takes too much time and the result was not very satisfactory. When I reduced the number of convolutional layer to just one and reran it, the overall speed increased drastically and the accuracy was very good which was about **90%**.


```python
### Train your model here.
### Feel free to use as many code cells as needed.
```


```python
# Find the best parameters for each configuration
epochs = 10
batch_size = 60
learning_rate = 0.03

# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)    

# Create an operation that initializes all variables
init = tf.initialize_all_variables()

# The accuracy measured against the validation set
validation_accuracy = 0.0

# Measurements use for graphing loss and accuracy
log_batch_step = 50
batches = []
loss_batch = []
train_acc_batch = []
valid_acc_batch = []

with tf.Session() as session:
    session.run(init)
    batch_count = int(math.ceil(len(train_features)/batch_size))

    for epoch_i in range(epochs):
        
        # Progress bar
        batches_pbar = tqdm(range(batch_count), desc='Epoch {:>2}/{}'.format(epoch_i+1, epochs), unit='batches')
        
        # The training cycle
        for batch_i in batches_pbar:
            # Get a batch of training features and labels
            batch_start = batch_i*batch_size
            batch_features = train_features[batch_start:batch_start + batch_size]
            batch_labels = train_labels[batch_start:batch_start + batch_size]
            
            # Run optimizer and get loss
            _, l = session.run(
                [optimizer, loss],
                feed_dict={x: batch_features, y_: batch_labels, keep_prob: 0.5})
            
            # Log every 50 batches
            if not batch_i % log_batch_step:
                # Calculate Training and Validation accuracy
                training_accuracy = session.run(accuracy, feed_dict=train_feed_dict)
                validation_accuracy = session.run(accuracy, feed_dict=valid_feed_dict)

                # Log batches
                previous_batch = batches[-1] if batches else 0
                batches.append(log_batch_step + previous_batch)
                loss_batch.append(l)
                train_acc_batch.append(training_accuracy)
                valid_acc_batch.append(validation_accuracy)

        # Check accuracy against Validation data
        validation_accuracy = session.run(accuracy, feed_dict=valid_feed_dict)

loss_plot = plt.subplot(211)
loss_plot.set_title('Loss')
loss_plot.plot(batches, loss_batch, 'g')
loss_plot.set_xlim([batches[0], batches[-1]])
acc_plot = plt.subplot(212)
acc_plot.set_title('Accuracy')
acc_plot.plot(batches, train_acc_batch, 'r', label='Training Accuracy')
acc_plot.plot(batches, valid_acc_batch, 'b', label='Validation Accuracy')
acc_plot.set_ylim([0, 1.0])
acc_plot.set_xlim([batches[0], batches[-1]])
acc_plot.legend(loc=4)
plt.tight_layout()
plt.show()

print('Validation accuracy at {}'.format(validation_accuracy))
```

    Epoch  1/10: 100%|██████████| 677/677 [02:13<00:00,  5.07batches/s]
    Epoch  2/10: 100%|██████████| 677/677 [02:12<00:00,  5.10batches/s]
    Epoch  3/10: 100%|██████████| 677/677 [02:12<00:00,  5.12batches/s]
    Epoch  4/10: 100%|██████████| 677/677 [02:11<00:00,  5.14batches/s]
    Epoch  5/10: 100%|██████████| 677/677 [02:11<00:00,  5.13batches/s]
    Epoch  6/10: 100%|██████████| 677/677 [02:12<00:00,  5.12batches/s]
    Epoch  7/10: 100%|██████████| 677/677 [02:12<00:00,  5.13batches/s]
    Epoch  8/10: 100%|██████████| 677/677 [02:12<00:00,  5.11batches/s]
    Epoch  9/10: 100%|██████████| 677/677 [02:11<00:00,  5.15batches/s]
    Epoch 10/10: 100%|██████████| 677/677 [02:13<00:00,  5.07batches/s]



![png](output_42_1.png)


    Validation accuracy at 0.9082826375961304


# Measuing Accuracy Against The Test Set


```python
# The accuracy measured against the test set
test_accuracy = 0.0

with tf.Session() as session:
    
    session.run(init)
    batch_count = int(math.ceil(len(train_features)/batch_size))

    for epoch_i in range(epochs):
        
        # Progress bar
        batches_pbar = tqdm(range(batch_count), desc='Epoch {:>2}/{}'.format(epoch_i+1, epochs), unit='batches')
        
        # The training cycle
        for batch_i in batches_pbar:
            # Get a batch of training features and labels
            batch_start = batch_i*batch_size
            batch_features = train_features[batch_start:batch_start + batch_size]
            batch_labels = train_labels[batch_start:batch_start + batch_size]

            # Run optimizer
            _ = session.run(optimizer, feed_dict={x: batch_features, y_: batch_labels, keep_prob: 1})

        # Check accuracy against Test data
        test_accuracy = session.run(accuracy, feed_dict=test_feed_dict)


assert test_accuracy >= 0.80, 'Test accuracy at {}, should be equal to or greater than 0.80'.format(test_accuracy)
print('Nice Job! Test Accuracy is {}'.format(test_accuracy))
```

    Epoch  1/10: 100%|██████████| 677/677 [00:19<00:00, 35.12batches/s]
    Epoch  2/10: 100%|██████████| 677/677 [00:16<00:00, 41.39batches/s]
    Epoch  3/10: 100%|██████████| 677/677 [00:16<00:00, 41.04batches/s]
    Epoch  4/10: 100%|██████████| 677/677 [00:16<00:00, 41.01batches/s]
    Epoch  5/10: 100%|██████████| 677/677 [00:16<00:00, 40.83batches/s]
    Epoch  6/10: 100%|██████████| 677/677 [00:17<00:00, 37.85batches/s]
    Epoch  7/10: 100%|██████████| 677/677 [00:16<00:00, 40.02batches/s]
    Epoch  8/10: 100%|██████████| 677/677 [00:16<00:00, 41.06batches/s]
    Epoch  9/10: 100%|██████████| 677/677 [00:16<00:00, 40.92batches/s]
    Epoch 10/10: 100%|██████████| 677/677 [00:16<00:00, 40.94batches/s]


    Nice Job! Test Accuracy is 0.8361045122146606


### Question 4

_How did you train your model? (Type of optimizer, batch size, epochs, hyperparameters, etc.)_


**Answer:**

I used *Gradient Descent* as my optimizer, and I set *epoch = 10*, *batch size = 60*, and *learning rate = 0.03*. I found that learning rate 0.01 to be inefficient and slow for this data set and using a greater rate showed better results. I didn't increase the number of *epoch* since the accuracy graph did not increase after 6000 batches.

## Question 5


_What approach did you take in coming up with a solution to this problem?_

**Answer:**

It took me several days to come up with this architecture. Initially, I used a basic layer with a lot of epochs and small learning rates. I experimented with it and realized that the maximum accuracy I can get is about **77%**. I believe that the minimum loss from the basic architecture is about that level. I decided to complicate the architecture using numerous techniques such as convolutional layers, ReLU, Dropout, and so on. It takes some time to figure out how to write down scripts for those layers and I confronted a lot of errors while doing it. Throughout trials and erros over several days, I found the most efficient - the most accurate and fastest - deep learning architecture.

---

## Step 3: Test a Model on New Images

Take several pictures of traffic signs that you find on the web or around you (at least five), and run them through your classifier on your computer to produce example results. The classifier might not recognize some local signs but it could prove interesting nonetheless.

You may find `signnames.csv` useful as it contains mappings from the class id (integer) to the actual sign name.

### Implementation

Use the code cell (or multiple code cells, if necessary) to implement the first step of your project. Once you have completed your implementation and are satisfied with the results, be sure to thoroughly answer the questions that follow.


```python
### Load the images and plot them here.
### Feel free to use as many code cells as needed.
```


```python
import os
import cv2

# List the images from the saved directory
img_files = os.listdir('./new_signs/')

test_images = np.zeros((1,32,32,2))

# Show each image
for i in img_files:
    image = './new_signs/' + i
    img = plt.imread(image)
    img = cv2.resize(img, (32,32))
    plt.imshow(img)
    plt.show()
    
    # Append to the test_images array
    test_images = np.append(test_images, [img[:,:,[0,2]]], axis=0)
    
# Remove the zero matrix at index 0
test_images = test_images[1:]
```


![png](output_52_0.png)



![png](output_52_1.png)



![png](output_52_2.png)



![png](output_52_3.png)



![png](output_52_4.png)



![png](output_52_5.png)



![png](output_52_6.png)



```python
# List the images from the saved directory
img_files = os.listdir('./new_signs2/')

# Show each image
for i in img_files:
    image = './new_signs2/' + i
    print(i)
    img = plt.imread(image)
    img = cv2.resize(img, (32,32))
    plt.imshow(img)
    plt.show()
    
    # Append to the test_images array
    test_images = np.append(test_images, [img[:,:,[0,2]]], axis=0)
```

    0.jpg



![png](output_53_1.png)


    13.jpg



![png](output_53_3.png)


    17.JPG



![png](output_53_5.png)


    19.jpg



![png](output_53_7.png)


    2.jpg



![png](output_53_9.png)


    23.jpg



![png](output_53_11.png)


    9.jpg



![png](output_53_13.png)



```python
# Implement Min-Max scaling for image data
def normalize(image_data):
    a = 0.01
    b = 0.99
    color_min = 0.0
    color_max = 255.0
    
    return a + ( ( (image_data - color_min) * (b - a) )/(color_max - color_min))

# Normalize train features and test features
test_images = normalize(test_images)
```


```python
# Print out the test images shape
print("test images shape:", test_images.shape)
```

    test images shape: (14, 32, 32, 2)



```python
from sklearn.preprocessing import LabelBinarizer

# Saving labels
test_images_labels = np.array([[39],[28],[9],[17],[23],[14],[13],[0],[13],[17],[19],[2],[23],[9]])

# Turn labels into numbers and apply One-Hot Encoding
encoder = LabelBinarizer()
encoder.fit(test_labels)
test_images_labels = encoder.transform(test_images_labels)

# Change to float32, so it can be multiplied against the features in TensorFlow, which are float32
test_images_labels = test_images_labels.astype(np.float32)

print('Labels One-Hot Encoded')
```

    Labels One-Hot Encoded



```python
# Reshaping test images
test_images = np.reshape(test_images, [-1,32*32*2])

print(test_images.shape)
print("reshape successful")
```

    (14, 2048)
    reshape successful


### Question 6

_Choose five candidate images of traffic signs and provide them in the report. Are there any particular qualities of the image(s) that might make classification difficult? It would be helpful to plot the images in the notebook._



**Answer:**

I found the photos of traffic signs taken by other people from internet. Since the data that I used was 32 by 32 in numpy array, I reshaped all of the photos to 32 by 32. As a result, the photos became blurry and difficult to identify. When you look at the last picture, the word *"yield"* is not recognizable at all.


```python
### Run the predictions here.
### Feel free to use as many code cells as needed.
```


```python
test_feed_dict = {x: test_images, y_: test_images_labels, keep_prob: 1.0}

# Find the best parameters for each configuration
epochs = 20
batch_size = 30
learning_rate = 0.005

# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)    

# Create an operation that initializes all variables
init = tf.initialize_all_variables()

# The accuracy measured against the test set
test_accuracy = 0.0

with tf.Session() as session:
    
    session.run(init)
    batch_count = int(math.ceil(len(train_features)/batch_size))

    for epoch_i in range(epochs):
        
        # Progress bar
        batches_pbar = tqdm(range(batch_count), desc='Epoch {:>2}/{}'.format(epoch_i+1, epochs), unit='batches')
        
        # The training cycle
        for batch_i in batches_pbar:
            # Get a batch of training features and labels
            batch_start = batch_i*batch_size
            batch_features = train_features[batch_start:batch_start + batch_size]
            batch_labels = train_labels[batch_start:batch_start + batch_size]

            # Run optimizer
            _ = session.run(optimizer, feed_dict={x: batch_features, y_: batch_labels, keep_prob: 0.5})
            

        # Check accuracy against Test data
        test_accuracy, yy_ = session.run([accuracy, y_conv], feed_dict=test_feed_dict)
    print(np.argmax(yy_, axis=1))
        
print('Nice Job! Test Accuracy is {}'.format(test_accuracy))
```

    Epoch  1/20: 100%|██████████| 1354/1354 [00:17<00:00, 78.22batches/s]
    Epoch  2/20: 100%|██████████| 1354/1354 [00:17<00:00, 75.72batches/s]
    Epoch  3/20: 100%|██████████| 1354/1354 [00:18<00:00, 73.76batches/s]
    Epoch  4/20: 100%|██████████| 1354/1354 [00:17<00:00, 75.47batches/s]
    Epoch  5/20: 100%|██████████| 1354/1354 [00:17<00:00, 75.89batches/s]
    Epoch  6/20: 100%|██████████| 1354/1354 [00:17<00:00, 75.72batches/s]
    Epoch  7/20: 100%|██████████| 1354/1354 [00:18<00:00, 74.17batches/s]
    Epoch  8/20: 100%|██████████| 1354/1354 [00:17<00:00, 75.40batches/s]
    Epoch  9/20: 100%|██████████| 1354/1354 [00:17<00:00, 76.21batches/s]
    Epoch 10/20: 100%|██████████| 1354/1354 [00:17<00:00, 76.23batches/s]
    Epoch 11/20: 100%|██████████| 1354/1354 [00:19<00:00, 69.71batches/s]
    Epoch 12/20: 100%|██████████| 1354/1354 [00:18<00:00, 75.10batches/s]
    Epoch 13/20: 100%|██████████| 1354/1354 [00:18<00:00, 74.34batches/s]
    Epoch 14/20: 100%|██████████| 1354/1354 [00:18<00:00, 75.10batches/s]
    Epoch 15/20: 100%|██████████| 1354/1354 [00:17<00:00, 75.25batches/s]
    Epoch 16/20: 100%|██████████| 1354/1354 [00:18<00:00, 75.19batches/s]
    Epoch 17/20: 100%|██████████| 1354/1354 [00:18<00:00, 74.93batches/s]
    Epoch 18/20: 100%|██████████| 1354/1354 [00:18<00:00, 74.99batches/s]
    Epoch 19/20: 100%|██████████| 1354/1354 [00:18<00:00, 74.83batches/s]
    Epoch 20/20: 100%|██████████| 1354/1354 [00:18<00:00, 74.82batches/s]

    [13 11  1 12 11 14 12 35 13 17 21 37 23  9]
    Nice Job! Test Accuracy is 0.3571428656578064


    


### Question 7

_Is your model able to perform equally well on captured pictures or a live camera stream when compared to testing on the dataset?_


**Answer:**

Unfortunately, the model didn't perform very well with the captured pictures. The model performed on the dataset very well since the sizes of traffic sign images from the dataset are maximized on 32 by 32 pixels. In the set of captured pictures, the sizes of traffic signs are not maximized and they contain a lot of background images. For example when the model tried to identify the first test image, it sometimes focused on yellow, which led to the wrong label, and sometimes focused on blue, which led to the correct label. Since the model started from the random numbers, it can lead to two different outcomes depending on the random values.

Another reason that might explain the low accuracy on the captured pictures is that I removed *Green* channel from all of the images. When looking at the image of "speed limit 20", we can see that the image has the best quality: it is not blurry, the number 20 can be recognized clearly, and the background contains no noise. However, the model could not recognize the sign correctly; furthermore, the correct label is not one of top 5 predictions. The only plausible explanation for this error is the lack of color channels.


```python
### Visualize the softmax probabilities here.
### Feel free to use as many code cells as needed.
```


```python
test_result = np.argmax(yy_, axis=1)

for i in range(len(test_result)):
    plt.subplot(121)
    plt.title("Result {}".format(test_result[i]))
    plt.imshow(np.reshape(train_features[np.argwhere(np.argmax(train_labels, axis=1)==test_result[i])[0]],[32,32,2])[:,:,1])
    plt.subplot(122)
    plt.title("Actual {}".format(np.argmax(test_images_labels[i])))
    print(test_images[0].shape)
    plt.imshow(np.reshape(test_images[i],[32,32,2])[:,:,0])
    plt.show()
    plt.plot(yy_[i])
    top5 = np.argsort(yy_[i])[::-1][:5]
    for k in top5:
        plt.plot(k, yy_[i][k], 'ro')
        plt.annotate(k, xy=(k,yy_[i][k]))
    plt.title("Prediction on traffic sign # {}".format(np.argmax(test_images_labels[i])))
    plt.show()

```

    (2048,)



![png](output_65_1.png)



![png](output_65_2.png)


    (2048,)



![png](output_65_4.png)



![png](output_65_5.png)


    (2048,)



![png](output_65_7.png)



![png](output_65_8.png)


    (2048,)



![png](output_65_10.png)



![png](output_65_11.png)


    (2048,)



![png](output_65_13.png)



![png](output_65_14.png)


    (2048,)



![png](output_65_16.png)



![png](output_65_17.png)


    (2048,)



![png](output_65_19.png)



![png](output_65_20.png)


    (2048,)



![png](output_65_22.png)



![png](output_65_23.png)


    (2048,)



![png](output_65_25.png)



![png](output_65_26.png)


    (2048,)



![png](output_65_28.png)



![png](output_65_29.png)


    (2048,)



![png](output_65_31.png)



![png](output_65_32.png)


    (2048,)



![png](output_65_34.png)



![png](output_65_35.png)


    (2048,)



![png](output_65_37.png)



![png](output_65_38.png)


    (2048,)



![png](output_65_40.png)



![png](output_65_41.png)


### Question 8

*Use the model's softmax probabilities to visualize the **certainty** of its predictions, [`tf.nn.top_k`](https://www.tensorflow.org/versions/r0.11/api_docs/python/nn.html#top_k) could prove helpful here. Which predictions is the model certain of? Uncertain? If the model was incorrect in its initial prediction, does the correct prediction appear in the top k? (k should be 5 at most)*


**Answer:**

The model recognized only few traffic signs correctly. We have 9 cases where the actual sign labels are listed in top 5 predictions and 5 cases where the actual sign labels are not listed in top 5 predictions. Looking at the probability graphs above, the top 5 predictions are close together, which means that the model is uncertain about its prediction.  

Even though the model did not accurately recognize the traffic signs, it did a good job recognizing the shape of the traffic signs. There are only **3** cases when the model did not recognize the shapes of traffic signs: #39, #17, and #13. There are common things between these 3 captured images. First, the sizes of the signs are not maximized. Second, there are too much of background noises.

### Question 9
_If necessary, provide documentation for how an interface was built for your model to load and classify newly-acquired images._


**Answer:**

Deep Neural Network Architecture: https://www.tensorflow.org/versions/r0.11/tutorials/mnist/pros/index.html

Traffic Signs Pictures: http://luxadventure.blogspot.com/2014/02/driving-me-crazy-red-lights-roundabouts_11.html


> **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  \n",
    "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.
