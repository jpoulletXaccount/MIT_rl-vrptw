
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import mixture
import joblib

ev = "cvrptw"
# df = pd.read_csv("/Users/jpoullet/Documents/MIT/Thesis/ML6867_project/data_UPS/list_tuesday_Central_area_" + ev +".csv")
# print(df.columns)
# df = df.drop(columns=['Latitude', 'Longitude','service_time'])
# print(df.columns)
# X = df.to_numpy()
# print(X)

# fit a Gaussian Mixture Model, n_components corresponds to K
# clf = mixture.GaussianMixture(n_components=13, covariance_type='full')
# clf.fit(X)

# output the mixture
# joblib.dump(clf, ev +'.joblib')

# read the mixture
clf = joblib.load( ev + '.joblib')

#Sample generation. Labels are not useful for us, they refer to the gaussian component used to generate sample
sample_X, _ = clf.sample(n_samples=1000)
x_unif = np.random.uniform(0,1,size=1000)
y_unif = np.random.uniform(0,1,size=1000)

plt.scatter(x_unif,y_unif,label = 'train instances')
plt.scatter(sample_X[:,0],sample_X[:,1],label = 'test instances')

plt.title('Train and test distributions')
plt.xlabel('x coordinate')
plt.ylabel('y coordinate')
plt.legend()
plt.show()
assert False

maxK = 20
bicscore = np.zeros(maxK)
for i in range(0,maxK):
    print(i)
    clfi = mixture.GaussianMixture(n_components=i+1, covariance_type='full')
    clfi.fit(X)
    bicscore[i] = clfi.bic(X)
print(bicscore)
bicgradient = np.zeros(maxK-1)
#plt.plot(range(maxK),bicscore)
for i in range(maxK -1):
    bicgradient[i] = bicscore[i+1] - bicscore[i]
plt.plot(range(maxK-1),bicgradient)
plt.title("Gradient of BIC scores for UPS TW Central area data")
plt.xlabel("Number of components")
plt.ylabel("Value of gradient")
plt.show()

plt.plot(range(maxK-1),bicscore[:(maxK-1)])
plt.title("BIC scores for UPS TW Central area data")
plt.xlabel("Number of components")
plt.ylabel("Value of BIC")
plt.show()
