from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#loading the dataset

my_dataset = datasets.load_digits()
_,axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, my_dataset.images, my_dataset.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)
# plt.imshow(my_dataset.images[0], interpolation='nearest')
# plt.show()
# print(type(my_dataset.images[0]))
n = len(my_dataset.images)
print(my_dataset.images.shape)
data = my_dataset.images.reshape((n,-1))
print(data.shape)

X_train, X_test, y_train, y_test = train_test_split(data, my_dataset.target, test_size = 0.2, shuffle=True)

model = linear_model.SGDClassifier()

model.fit(X_train,y_train)

predict = model.predict(X_test)

accuracy = accuracy_score(y_test,predict)

print(accuracy)

for ax, image, label in zip(axes, X_test, predict):
    ax.set_axis_off()
    ax.imshow(image.reshape((8,8)), cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)
plt.show()