import os
from skimage.io import imread
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

input_dir = 'path'

categories = ['cat1', 'cat2']

data = []
labels = []

data = [resize(imread(os.path.join(input_dir, category, file)), (8, 8)).flatten()
        for category_idx, category in enumerate(categories)
        for file in os.listdir(os.path.join(input_dir, category))]
labels = [category_idx for category_idx, category in enumerate(categories)
          for file in os.listdir(os.path.join(input_dir, category))]

data = np.asarray(data)
labels = np.array(labels)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

classifier = SVC()

parameters = [{'gamma': [0.1, 1, 10, 100], 'C': [0.1, 1, 10, 100, 1000, 10000]}]
grid_search = GridSearchCV(classifier, parameters)

grid_search.fit(x_train, y_train)

best_estimator = grid_search.best_estimator_

y_prediction = best_estimator.predict(x_test)
score = accuracy_score(y_prediction, y_test)
print('{}% of samples were correctly classified'.format(str(score * 100)))
