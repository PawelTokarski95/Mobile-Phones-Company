import pandas as pd
import seaborn as sns
import numpy as np
import statistics
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures


MPhones = pd.read_csv('C:/Users/pawel/OneDrive/Pulpit/Projekty Python/Mobile-phones.csv', sep=';')

print(MPhones.head())
MPhones.info()

print(np.mean(MPhones['m_dep']))
print(np.nanmedian(MPhones['m_dep']))
print(statistics.mode(MPhones['m_dep']))

plt.hist(MPhones['m_dep'])
plt.show()
MPhones['m_dep'] = MPhones['m_dep'].fillna(np.mean(MPhones['m_dep']))
print(MPhones['m_dep'].describe())

print(np.mean(MPhones['px_height']))
print(np.nanmedian(MPhones['px_height']))
print(statistics.mode(MPhones['px_height']))
plt.hist(MPhones['px_height'])
plt.show()

MPhones = MPhones.fillna(0)
MPhones = MPhones.astype(int)
imputer = KNNImputer()
imputer.fit_transform(MPhones)
print(len(MPhones))
MPhones.info()


sns.heatmap(MPhones.corr())
plt.show()
##Covariance matrix



X_train, X_test, y_train, y_test = train_test_split(MPhones.iloc[:, 0:19], MPhones['price_range'], random_state=42,
                                                    test_size=0.33)

LR = LinearRegression()
LR.fit(X_train, y_train)
Prediction_LR = LR.predict(X_test)
print(r2_score(y_test, Prediction_LR))

Poly = PolynomialFeatures()
Poly_final = Poly.fit_transform(X_train)

LR2 = LinearRegression()
LR2.fit(Poly_final, y_train)


Poly_predict = Poly.fit_transform(X_test)
Poly_predict_Final = LR2.predict(Poly_predict)
print(r2_score(y_test, Poly_predict_Final))





