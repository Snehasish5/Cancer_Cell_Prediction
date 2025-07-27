import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()

x=cancer.data[:,:2]
y=cancer.target

mylbl=cancer.target_names
print(mylbl)

xdf=pd.DataFrame(x)

sns.heatmap(xdf.isnull())

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2)

from sklearn.svm import SVC
svm_model=SVC(kernel="rbf",gamma=.5,C=1)
svm_model.fit(x_train,y_train)

from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.colors
colmap=matplotlib.colors.ListedColormap(['green','red'])
DecisionBoundaryDisplay.from_estimator(
    svm_model,
    x,
    response_method="predict",
    cmap=colmap,
    xlabel="Radius",
    ylabel="Textture"
)

plt.scatter(x[:,0],x[:,1],c=y,s=20)

inp=[[17,20]]
yp=svm_model.predict(inp)
print(yp)

from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.colors
colmap=matplotlib.colors.ListedColormap(['red','green'])
DecisionBoundaryDisplay.from_estimator(
    svm_model,
    x,
    response_method="predict",
    cmap=colmap,
    xlabel="Radius",
    ylabel="Textture"
)

plt.scatter(x[:,0],x[:,1],c=y,s=20)
plt.scatter(17,20,c="pink",s=40)