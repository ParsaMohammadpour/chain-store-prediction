#!/usr/bin/env python
# coding: utf-8

# # Importing libraries we are going to use

# In[246]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import datetime


# # Loading dataframes 

# In[247]:


print('original dataframe:')
df = pd.read_csv('sales_train.csv')
print(df.head())
print('shops dataframe:')
shops = pd.read_csv('shops.csv')
print(shops.head())
print('item categories dataframe:')
categories = pd.read_csv('item_categories.csv')
print(categories.head())
print('items dataframe:')
items = pd.read_csv('items.csv')
print(items.head().to_string())


# # Merging all dataframes into one

# In[248]:


df = pd.merge(df, shops, on='shop_id')
df = pd.merge(df, items, on='item_id')
df = pd.merge(df, categories, on='item_category_id')
print(df.head())


# # Correcting columns with negative value and making month column

# In[249]:


df['item_cnt_day'] = abs(df['item_cnt_day'])
df['item_price'] = abs(df['item_price'])

df['month'] = df['date'].str[3:5]
print(df.head())


# # Making one row for each item of shop that was sold in every month

# In[250]:


df = df.groupby(['item_id', 'item_category_id', 'shop_id', 'date_block_num', 'month', 'item_price'])['item_cnt_day'].sum().reset_index(name='total-sell')
print(df.head())

# This Part Was Only Added For Test
# a = df.groupby(['item_id', 'item_category_id', 'shop_id', 'date_block_num', 'month', 'item_price'])['item_cnt_day'].sum().reset_index(name='total-sell')
# b = df.loc[df['item_id'] == 22145]
# b = b.loc[b['shop_id'] == 25]
# b = b.loc[b['date_block_num'] == 14]
# print(len(b))
# print(b)
# print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
# b = a.loc[a['item_id'] == 22145]
# b = b.loc[b['shop_id'] == 25]
# b = b.loc[b['date_block_num'] == 14]
# print(len(b))
# print(b)


# # One-Hot-Encode for month & shop

# In[251]:


df = pd.get_dummies(df, prefix=['month', 'shop'], columns = ['month', 'shop_id'])
print(df.head())


# In[258]:


print(df.columns)
columns = ['item_id', 'item_category_id', 'item_price', 'date_block_num']
[columns.append('month_0' + str(i)) if i < 10 else columns .append('month_' + str(i)) for i in range(1, 13)]
[columns.append('shop_' + str(i)) for i in range(0, 60)]
x = df.loc[:, columns]
y = df.loc[:, 'total-sell']
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.25)
print('train_x:\n', len(train_x))
print('train_y:\n', len(train_y))
print('test_x:\n', len(test_x))
print('test_y:\n', len(test_y))


# # Linear regression

# In[260]:


regr = LinearRegression()

regr.fit(train_x, train_y)
print(regr.score(test_x, test_y))

y_predict = regr.predict(test_x)
y_predict = np.around(y_predict).astype(int)

# The coefficients
print("Coefficients: \n", regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(test_y, y_predict))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(test_y, y_predict))

plt.plot([i for i in range(len(y_predict[:100]))], y_predict[:100], color='red')
plt.plot([i for i in range(len(y_predict[:100]))], test_y[:100], color='blue')
plt.show()

