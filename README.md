Question 1 (35 points)
Image classification with linear classifiers and neural networks. In this exercise, you
will implement a linear classifier for a simple image classification problem, using the KuzushijiMNIST dataset, which contains handwritten cursive images of 10 characters from the Hiragana
writing system (used for Japanese). Examples of images in this dataset are shown in Figure 1.


Please do not use any machine learning library such as scikit-learn or similar for
this exercise; just plain linear algebra (the numpy library is fine). Python skeleton code
is provided (hw1-q1.py).

In order to complete this exercise, you will need to download the Kuzushiji-MNIST dataset.
You can do this by running the following command in the homework directory:
python download_kuzushiji_mnist.py.

1. (a) (10 points) Implement the update_weights method of the Perceptron class in hw1-q1.py.
Then train 20 epochs of the perceptron on the training set and report its performance
on the validation and test set. Plot the accuracies as a function of the epoch number.
You can do this with the command
python hw1-q1.py perceptron
(b) (10 points) Repeat the same exercise using logistic regression instead (without regularization), using stochastic gradient descent as your training algorithm. Set a fixed
learning rate η = 0:001. This can be solved by implementing the update_weights
method in the LogisticRegression class. You can do this with the command
python hw1-q1.py logistic_regression
2. Now, you will implement a multi-layer perceptron (a feed-forward neural network) again
using as input the original feature representation (i.e. simple independent pixel values).
1Figure 1: Examples of images from the Kuzushiji-MNIST dataset.
(a) (5 points) Justify briefly why multi-layer perceptrons with non-linear activations are
more expressive than the simple perceptron implemented above, and what kind of
limitations they overcome for this particular task. Is this still the case if the activation
function of the multi-layer perceptron is linear?
(b) (10 points) Without using any neural network toolkit, implement a multi-layer
perceptron with a single hidden layer to solve this problem, including the gradient
backpropagation algorithm which is needed to train the model. Use 200 hidden units,
a relu activation function for the hidden layers, and a multinomial logistic loss (also
called cross-entropy) in the output layer. Don’t forget to include bias terms in your
hidden units. Train the model with stochastic gradient descent with a learning rate
of 0.001. Initialize biases with zero vectors and values in weight matrices with wij ∼
N (µ; σ2) with µ = 0:1 and σ2 = 0:12 (hint: use numpy.random.normal). Run your code
with the command
python hw1-q1.py mlp
Question 2 (35 points)
Image classification with an autodiff toolkit. In the previous question, you had to write
gradient backpropagation by hand. This time, you will implement the same system using a
deep learning framework with automatic differentiation. Pytorch skeleton code is provided
(hw1-q2.py) but if you feel more comfortable with a different framework, you are free to use it
instead.
1. (10 points) Implement a linear model with logistic regression, using stochastic gradient descent as your training algorithm (use a batch size of 1). Train your model for 20 epochs and
tune the learning rate on your validation data, using the following values: {0:001; 0:01; 0:1}.
Report the best configuration (in terms of final validation accuracy) and plot two things: the
training loss and the validation accuracy, both as a function of the epoch number. Report
the final accuracy on the test set.
In the skeleton code, you will need to implement the method train_batch() and the class
LogisticRegression’s __init__() and forward() methods.
2. (15 points) Implement a feed-forward neural network with a single layer, using dropout
regularization. Make sure to include all the hyperparameters and training/model design
choices shown in Table 1. Use the values presented in the table as default. Tune each of
these hyperparameters while leaving the remaining at their default value:
• The learning rate: {0:001; 0:01; 0:1}.
Page 2• The hidden size: {100; 200}.
• The dropout probability: {0:3; 0:5}.
• The activation function: relu and tanh.
Number of Epochs 20
Learning Rate 0.01
Hidden Size 100
Dropout 0.3
Batch Size 16
Activation ReLU
Optimizer SGD
Table 1: Default hyperparameters.
Report your best configuration, make similar plots as in the previous question, and report
the final test accuracy.
In the skeleton code, you will need to implement the class FeedforwardNetwork’s __init__()
and forward() methods.
3. (10 points) Using the same hyperparameters as in Table 1, increase the model to 2 and 3
layers. Report your best configuration, make similar plots as in the previous question, and
report the final test accuracy. (Note: in the real world, you would need to do hyperparameter
tuning for the different network architectures, but this is not required for this assignment.)



## **Image Classification Model**
This project, originally an evaluation component for the Deep Learning course (2022/2023), talking place in Instituto Superior Técnico, University of Lisbon, aimed to **implement a linear classifier for a simple image classification problem**. More specifically, this poject utilises the KuzushijiMNIST dataset, which contains handwritten cursive images of 10 characters from the Hiragana writing system (used for Japanese). Examples of images in this dataset are shown in Figure 1.

<p align="center">
  <img src="https://github.com/user-attachments/assets/598e0554-0dff-4928-82bc-ea1ffdb41e92"/>
</p>

The following document indicates how to access and utilise the source code. It also contains a brief analysis of the implementation and results, referring to the [official report](https://github.com/zorrocrisis/NaturalLanguageClassificationModel/blob/main/FinalReport.pdf) for more detailed information.

## **Quick Start**
This project's source files can be downloaded from this repository. They are divided into the following main files:
- ***reviews.py*** - contains the best final classification model, training it on the training set (*train.txt*) to classify the test set (*test_just_reviews.txt*) and producing a list of labels/classifications (*results.txt*).
- ***train_multiple_models.py*** - trains and tests the implemented machine learning classification models.
- ***fine_tuning.py*** - fine-tunes the machine learning classification models.
- ***bert.py*** and ***tcn_models.py*** - contains the deep learning classification models.

To run this poject, follow these steps:
1. Install the necessary dependencies:
     - pip install pandas
     - pip install nltk
     - pip install scikit-learn
  
2. Simply run whatever file you would like utilising a terminal. As an example:
     - python reviews.py
  
Feel free to change the test and training sets, as well as any other parameters you see fit.

## **Task Introduction**

## **Implementations**

## **Authors and Acknowledgements**
This project was developed by **[Miguel Belbute (zorrocrisis)](https://github.com/zorrocrisis)** and [Guilherme Pereira](https://github.com/the-Kob).

The skeleton code was supplied by Francisco Melo (fmelo@inesc-id.pt).
