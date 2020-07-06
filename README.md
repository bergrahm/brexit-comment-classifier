# Brexit Comment Classifier, Machine Learning with scikit-learn in Python

## Get Started

This repository contains files that make use of the sklearn API to classify whether comments found on the internet regarding Brexit are either for (1) or against (0) it.

Before we being, it's worth mentioning that the code was written in Python version 3.6, and thus to avoid unecessary errors it is recommended that you have 3.6 or later version installed. I've not tested if this walkthrough runs smoothly on older versions (I guess it should, but who knows).

Additionally, this repository makes use of the sklearn API and requires you to make use of it in order to run it successfully. So if you haven't done so already, try the following command in your terminal:
```
$ pip install sklearn
```
On successful installation, you now have access to the library that will be used for this implementation.

## Execute the Classifier
Ok. Now that you have both Python and the correct dependencies on your system you are ready to test the classifier. In this repository you will find three files:

- brexit.py
- train-data.tsv
- test-data.tsv

As you may suspect the brexit.py file is the executable file that makes use of the two tsv-files. The train-data.tsv contains the data on which the classifier will be trained on and tested onto the test-data.tsv file. In total the two files contain around 9000 documents where roughly 6700 are distributed into the train-data.tsv. 

If you want to try this classifier on your own data, you can move all entries from the test-data.tsv into train-data.tsv or just replace test-data.tsv with your own test data. Just be sure to note the format of the files; that the annotations are first and separated by a tab (hence, a tsv file).

To execute the classifier, first navigate to the folder containing the files in your system's terminal. Execute the following command:
```
$ python brexit.py
```
*Note: please observe that the script creates a new file named "preprocessed-brexit.tsv". This file contains a smaller training set with higher levels of agreement between annotators. This is the file on which the script trains after filtering out uncertain documents.*

Observe the confusion matrix provided to understand how well the classifier performed. If you used the data provided you should see around 79% accuracy for this set of data.

## About
The data has been crowd-sourced from various forums on the internet. There has been no sanitation for duplicate comments from the crowd and it could possibly have implications on how the model classifies the test set, though I choose to ignore this and blindly believe that it makes for stronger association in the f-score (hopefully all the annotations agree on the same thing). Finally, the annotations have been made in different rounds from different people so that bias where the comment was mined from is removed, context is a large factor to understanding whether some comments are for or against Brexit.

If you want to learn more about scikit-learn, you can visit them on their website and read the documentation here: https://scikit-learn.org/stable/
