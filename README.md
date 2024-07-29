## **ImageClassificationModel**
This project, originally an evaluation component for the Deep Learning course (2022/2023), talking place in Instituto Superior Técnico, University of Lisbon, aimed to 

<p align="center">
  <img src="https://github.com/user-attachments/assets/970eb12e-2859-479c-89a4-824c4d121b0e"/>
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

The skeleton code was supplied by [Francisco Melo](fmelo@inesc-id.pt).
