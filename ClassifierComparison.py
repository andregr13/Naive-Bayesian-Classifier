import sklearn as sk
import numpy as np
import scipy as sp
import pandas as pd
from BayesClassifier import MyNBClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import timeit

# simple percent success accuracy metric
def accuracy(y, yhat):
    return np.mean(y == yhat)

def class_acc(y, yhat, c):
    return np.count_nonzero(yhat[y==c] == c)

# import dataset
df = pd.read_csv("wine.csv")

# split x and y
x = df.iloc[:,1:].to_numpy()
y = df.iloc[:,0].to_numpy()


print("Wine samples from cultivar 1:", np.count_nonzero(y == 1))
print("Wine samples from cultivar 2:", np.count_nonzero(y == 2))
print("Wine samples from cultivar 3:", np.count_nonzero(y == 3))
print("Total samples:", y.size)

# adjusting y from [1,2,3] to [0,1,2] to work with array indices
y = y-1

# scaler is not necessary for bayes classifiers but may make other types perform better
# scaler = StandardScaler()
# scaler = scaler.fit(x)

# set random_state=5 for reproducibility to show results in report
# the random state value was arbitrarily chosen
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=5)


cnt1 = np.count_nonzero(y_test == 0)
cnt2 = np.count_nonzero(y_test == 1)
cnt3 = np.count_nonzero(y_test == 2)

# common classification algorithms for comparison
ada = AdaBoostClassifier()
forest = RandomForestClassifier(random_state=5) #again set arbitrary random_state for reproducable results
knn = KNeighborsClassifier()
bnb = MultinomialNB()
gnb = GaussianNB()
bayes = MyNBClassifier()

class_names = np.array(["AdaBoost","Random Forest",
                        "K-Nearest Neighbors",
                        "Multinomial Naive Bayes","Gaussian Naive Bayes",
                        "Handwritten Binary Naive Bayes"])
classifiers = np.array([ada, forest, knn, bnb, gnb, bayes])
predictions = np.empty((classifiers.size,y_test.size))
accuracies = np.zeros(classifiers.size)


for i in range(classifiers.size):
    start = timeit.default_timer() # time to check efficiency
    
    # fit each classifier to train data and predict on test data
    classifiers[i].fit(x_train,y_train)
    predictions[i] = classifiers[i].predict(x_test)
    accuracies[i] = accuracy(predictions[i], y_test)
    
    end = timeit.default_timer()-start # stop time
    
    # print results
    print("-----------------------------------------")
    print(class_names[i])
    print("Cultivar 1: %s/%s, Cultivar 2: %s/%s Cultivar 3 %s/%s"
          % (class_acc(y_test,predictions[i],0),cnt1,
             class_acc(y_test,predictions[i],1),cnt2,
             class_acc(y_test,predictions[i],2),cnt3))
    print("\nAccuracy = ", accuracies[i])
    print("\nRuntime = ", end)
    print("\n----------------------------------------")
    
    
# x_plt, y_plt = make_classification(n_features=2, n_redundant=0, n_informative=5, random_state=1, n_clusters_per_class=1)
# rng = np.random.RandomState(2)
# x_plt += 2 * rng.uniform(size=x_plt.shape)
# linearly_separable = (X, y)

# datasets = [make_moons(noise=0.3, random_state=0),
#             make_circles(noise=0.2, factor=0.5, random_state=1),
#             linearly_separable]