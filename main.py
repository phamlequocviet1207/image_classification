import os
import pickle
import numpy as np
from skimage.io import  imread
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import  accuracy_score


#prepare data
input_dir = 'D:\OPEN CV\Projects\ParkingLotDetectorAndCounter-20240703T130554Z-001\ParkingLotDetectorAndCounter\clf-data\clf-data'
categories = ['empty', 'not_empty']

data = []
labels = []
# print (enumerate(categories))

# enumerate examples:
# x = ('apple', 'banana', 'cherry')
# y = enumerate(x)
# y = [(0, 'apple'), (1, 'banana'), (2, 'cherry')]

for category_dix, category in enumerate(categories):
    print(category_dix, category)
    for file in os.listdir(os.path.join(input_dir, category)):
        img_path = os.path.join(input_dir, category, file)
        # print(img_path)
        img = imread(img_path)
        img = resize(img, (15,15))
        data.append(img.flatten())
        labels.append(category_dix)

data = np.asarray(data)
labels = np. asarray(labels)

# train / test split
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# train classifier
classifier = SVC()

parameter = [{'gamma': [0.01,0.001, 0.0001], 'C': [1, 10, 100, 1000]}]

grid_search = GridSearchCV(classifier, parameter)

grid_search.fit(x_train, y_train)

# test_performance
best_estimator = grid_search.best_estimator_

y_prediction = best_estimator.predict(x_test)

score = accuracy_score(y_prediction, y_test)

print('{}% of samples were correctly classified'.format(str(score*100)))

pickle.dump(best_estimator, open('./model.p', 'wb'))


