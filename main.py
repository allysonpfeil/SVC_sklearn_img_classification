# import libraries
import os
from skimage.io import imread
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# get data from folder containing each classification of images
input_dir = 'path'

# define the two different categories of images from the folder
    # the folders inside input_dir's path must be named these two strings for it to work
categories = ['cat1', 'cat2']

# define open lists to store data in
data = []
labels = []

# list comprehension for reading the files and determining their assignment
data = [resize(imread(os.path.join(input_dir, category, file)), (8, 8)).flatten()
        for category_idx, category in enumerate(categories)
        for file in os.listdir(os.path.join(input_dir, category))]
labels = [category_idx for category_idx, category in enumerate(categories)
          for file in os.listdir(os.path.join(input_dir, category))]

# convert data to numpy array with the label attached
data = np.asarray(data)
labels = np.array(labels)

# split the data into training and testing groups 
    # training group is 80% of data; testing group is 20% of data
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# train the SVC image classifier
classifier = SVC()

# define the parameters 
    # gamma is a parameter for non linear hyperplanes
    # C is the penalty parameter of the error term
    # this program will train # of C * gamma models, and then determine the most accurate model
parameters = [{'gamma': [0.1, 1, 10, 100], 'C': [0.1, 1, 10, 100, 1000, 10000]}]
grid_search = GridSearchCV(classifier, parameters)

# fit the model
grid_search.fit(x_train, y_train)

# test the performance
best_estimator = grid_search.best_estimator_

# print to console the performance
y_prediction = best_estimator.predict(x_test)
score = accuracy_score(y_prediction, y_test)
print('{}% of samples were correctly classified'.format(str(score * 100)))
