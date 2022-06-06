import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv("Book.csv")

plt.scatter(data.videos, data.views, color = 'red')# methanin thamai prasthare adinne,
plt.xlabel("number of videos")# x akshaye nama
plt.ylabel("number of viwes")# y akshaye nama
plt.show()

x = np.array(data.videos.values)# model ekaka train katanna kalin data tika array ekakata daganna ona
y = np.array(data.views.values)

model = LinearRegression() # class ekata adalawa object ekak hadagaththa
model.fit(x.reshape(-1, 1), y)# model eka traing karanne methanadi

#************** x akshaye values api model eka train karanna denna ona 2 dimentional array ekaka vidiyata

# traing karapu model kakata x agaya dunnama eta adala y agaya genima
new_x = np.array(45).reshape(-1,1)# api dena x value eka dennath ona 2D array ekaka vidiyata
pred = model.predict(new_x)# aip hadagaththa model ekata dena eka
print(pred)

# best fit Line eka hoyaganna vidiya
plt.scatter(data.videos, data.views, color = 'red')
m, c = np.polyfit(x, y, 1)# anucramanaya saha anthak kandaya soyana vidiya
bestFitLine = plt.plot(x, m*x+c)
plt.show()

#multipul Vereable Lenear regretion

datas = pd.read_csv("Book2.csv")
models = LinearRegression()
models.fit(datas[['videos','days', 'subscribers']], datas.views)
print(models)

# model ekata value dila y agaya genima
preds = models.predict([[45, 180, 3100]])

# anucramana genima
coef = models.coef_
inter = models.intercept_
print(coef)
print(inter)













































