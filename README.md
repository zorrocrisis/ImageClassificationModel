Question 1 (35 points)
3. Now, you will implement a multi-layer perceptron (a feed-forward neural network) again
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
This project, originally an evaluation component for the Deep Learning course (2022/2023), talking place in Instituto Superior Técnico, University of Lisbon, aimed to **implement a linear classifier for a simple image classification problem**. More specifically, this poject utilises the KuzushijiMNIST dataset (KMNIST dataset), which contains handwritten cursive images of 10 characters from the Hiragana writing system (used for Japanese). Examples of images in this dataset are shown below.

<p align="center">
  <img src="https://github.com/user-attachments/assets/598e0554-0dff-4928-82bc-ea1ffdb41e92"/>
</p>

The following document indicates how to access and utilise the source code. It also contains a brief analysis of the implementation and results.

## **Quick Start**

In order to complete this exercise, you will need to download the Kuzushiji-MNIST dataset.
You can do this by running the following command in the homework directory:
python download_kuzushiji_mnist.py.

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

## **Perceptron**
Firstly, a **Perceptron** class was implemented (*python hw1-q1.py perceptron*) and subsequently **trained for 20 epochs** on the training set (KMNIST dataset). The verified resulting validation and test accuracies were the following:

<p align="center">
  <img src="https://github.com/user-attachments/assets/6460c8d3-3d0c-4a1e-964b-a83856f8cb9d"/>
</p>

<p align="center">
  <i>Graph 1 - Validation and test accuracies of multi-class Perceptron</i>
</p>

## **Logistic Regression**
The same exercise was applied to a **logistic regression** class (*python hw1-q1.py logistic_regression*), without regularisation and utilising **stochastic gradient descent as its training algorithm** (this algorithm was written by hand - no libraries or auxiliary modules were employed). A learning rate of η = 0.001 was defined. The verified resulting validation and test accuracies were the following:

<p align="center">
  <img src="https://github.com/user-attachments/assets/a496fcc1-88cf-4551-869c-296c899581f6"/>
</p>

<p align="center">
  <i>Graph 2 - Validation and test accuracies of multi-class Logistic Regression</i>
</p>

Comparing the accuracies from Graph 1 and 2, it was verified that **the overall accuracy of the logistic regression was better, also producing more consistent results than the multi-class Perceptron**.

## **Multi-layer Perceptron**
Seeking to further improve the overall performance and expressiveness, a **multi-layer perceptron** (or a feed-forward neural network) was additionally developed. The intermediate (or hidden) layers continuously alter the input and propagate this information forward, ensuring the underlying data results in better representations. In essence, the increase in expressiveness is translated into a more complex, non-linear classifier.

More specifically, a **single hidden layer with 200 units and a relu activation function** was implemented. The **gradient backpropagation algorithm with a learning rate of 0.001** was developed (manually, without employing any libraries or auxiliary modules) to train the model, alongside a **multinomial logistic loss (also called cross-entropy) in the output layer**. The **biases were intialised with zero vectors**, whereas the **values in the weight matrices were initialised with N (µ; σ2), where µ = 0.1 and σ2 = 0.12**.

This model can be run with *python hw1-q1.py mlp* and its results are graphically represented here:

<p align="center">
  <img src="https://github.com/user-attachments/assets/bb56b417-b0bf-4ffc-8bdf-d009720baf33"/>
</p>

<p align="center">
  <i>Graph 3 - Evolution of validation and test accuracies in a multi-layer Perceptron (learning rate of 0.001)</i>
</p>

In comparison with Graph 1 and Graph 2, the values above demonstrate a clear **overall improvement regarding the model’s accuracy**.

Starting from around a 75% accuracy, **the multi-layer Perceptron immediately begins to learn with a considerable rate of accuracy per epoch** (although this rate decreases over time, the improvement is constant), **achieving a final accuracy of around 94% and 84% in the validation and test stages, respectively**. 

These outcomes are quite remarkable, especially taking into account the previously obtained results for the image classification models. Though the single-layer Perceptron starts its training with the same accuracy, it does not manage to learn at a consistent and noteworthy rate. The logistic regression, on the other hand, does showcase a continuous enhancement over the 20 epochs, despite it being slow progress and only reaching a final accuracy of about 82% and 70% in the validation and test stages. That being said, the multi-layer Perceptron improved its initial accuracy by around 20%, demonstrating much better and quicker (in epochs) results than the former models.

However, **one must also consider the computational performance of this version of the Perceptron** - whereas the other models took a few minutes to train, **the multi-layer Perceptron does require a longer learning time** (around 20 minutes, in the same computer as the other tests), which might not be ideal for every application.


## **Autodiff Toolkit**
In the previous models, the gradient backpropagation algorithm had been written by hand. That being said, a new system logisitc regrission classifier was developed (*python hw1-q2.py logistic_regression*), this time **employing a deep learning framework with automatic differentiation**, namely Pytorch.

More specifically, this new model functioned with **stochastic gradient descent with a batch size of 1** and with a **training duration of 20 epochs**, producing the following results for distinct values of learning rates:

<p align="center">
  <i>Table 1 - Effect of learning rate parameter in logistic regression</i>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/82650516-92aa-45eb-af48-db7edfa85d5d"/>
</p>



<p align="center">
  <img src="https://github.com/user-attachments/assets/a3dd2bf8-dbb6-4547-89bc-b8faee78516b"/>
  <img src="https://github.com/user-attachments/assets/5dd86603-f221-49f4-8670-c9bcd2883983"/>
  <img src="https://github.com/user-attachments/assets/96650339-d148-419c-a351-4d80749a3db7"/>
</p>

<p align="center">
  <i>Graph Set 1 -The resulting validation accuracies with increasing learning rates: 0.001, 0.01 and 0.1</i>
</p>

![validation_accuracy_0 001](https://github.com/user-attachments/assets/a3dd2bf8-dbb6-4547-89bc-b8faee78516b)

Graph Set 1 - From left to right, top to bottom we have the resulting validation accuracies with
increasing learning rates: 0.001, 0.01 and 0.1

![imagem](https://github.com/user-attachments/assets/82650516-92aa-45eb-af48-db7edfa85d5d)


From a brief analysis of the plotted data, we could report **the best configuration corresponded to the logistic regression classifier with a learning rate of 0.00**: not only does it have the best accuracies and final test loss, it seems to learn in a more consistent manner - the respective graphs have less sudden variations than the models with bigger learning rates. This can mean that bigger learning rates make the model “jump over” potential minima, thus explaining higher accuracies followed by considerable drops and an inconsistent evolution of these values.



3. (15 points) Implement a feed-forward neural network with a single layer, using dropout
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
4. (10 points) Using the same hyperparameters as in Table 1, increase the model to 2 and 3
layers. Report your best configuration, make similar plots as in the previous question, and
report the final test accuracy. (Note: in the real world, you would need to do hyperparameter
tuning for the different network architectures, but this is not required for this assignment.)


## **Implementations**

## **Authors and Acknowledgements**
This project was developed by **[Miguel Belbute (zorrocrisis)](https://github.com/zorrocrisis)** and [Guilherme Pereira](https://github.com/the-Kob).

The skeleton code was supplied by Francisco Melo (fmelo@inesc-id.pt).
