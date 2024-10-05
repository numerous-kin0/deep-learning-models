import codecademylib3_seaborn
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

data = [[0,0],[1,0],[0,1],[1,1]]

labels = [point[0] and point[1] for point in data]

x_values = [point[0] for point in data]
y_values = [point[1] for point in data]

plt.scatter(x_values,y_values,c=labels,cmap='bwr')
plt.show()

classifier = Perceptron()
classifier.max_iter = 40
classifier.random_state = 22
classifier.fit(data, labels)
print(classifier.score(data,labels))
print(classifier.decision_function([[0, 0], [1, 1], [0.5, 0.5]]))

x_values = np.linspace(0, 1, 100)
y_values = np.linspace(0, 1, 100)

point_grid = list(product(x_values, y_values))

distances = classifier.decision_function(point_grid) #decision function returns a list of values which indicate the distance of each ordered pair from the decision boundary

abs_distances = [abs(num) for num in distances]

distances_matrix = np.reshape(abs_distances, (100,100)) #Turns the absolute distance of abs_distances into a 2-dimensional list
print(distances_matrix)

heatmap = plt.pcolormesh(x_values,y_values,distances_matrix)

plt.colorbar(heatmap)
plt.show() #it does not display the decision boundary, figure out why
