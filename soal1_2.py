import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load Data
df_1 = pd.read_excel('indo_12_1.xls')
df_1 = df_1.drop([0,1,2])
# Set Columns
df = df_1.iloc[0:34]
List_Column = ['Provinsi', 1971, 1980, 1990, 1995, 2000, 2010]

df = df.set_index(np.arange(0, 34))

# print(df.iloc[0].values)
df.columns = List_Column
# print(df.columns)	# print(type(df.columns))

# Cleaning Data
df = df.replace(' ', '')
df = df.replace('-', np.NaN)
# print(df.info())	# Cleaned

# =============================================
# Processing
# =============================================

list_jumlah = []
for x in df.columns:
	if type(x) == int:
		list_jumlah.append(df[x].iloc[0:33].sum())
 
max_2010 = df.loc[df[2010] == df[2010].iloc[0:33].max()].values
max_2010_val = np.array(max_2010[:,1:]).reshape(-1,1)

min_1971 = df.loc[df[1971] == df[1971].iloc[0:33].min()].values
min_1971_val = np.array(min_1971[:,1:]).reshape(-1,1)

list_tahun_col = [1971, 1980, 1990, 1995, 2000, 2010]

Nama_Prov = [
	df.Provinsi.iloc[11],	
	df.Provinsi.iloc[6],
	df.Provinsi.iloc[33],
	'Best Fit Line'
]

# =============================================
# Machine Learning
# =============================================
from sklearn import linear_model
model_Jabar = linear_model.LinearRegression()
model_Bengk = linear_model.LinearRegression()
model_jumlah = linear_model.LinearRegression()

# X=[[1971, 1980, 1990, 1995, 2000, 2010]]
X = np.array([list_tahun_col]).reshape(-1,1)
y_Jabar = np.array(max_2010_val)
y_Bengk = np.array(min_1971_val)
y_Indo = np.array(list_jumlah)
# print(type(X)) ; print(type(y))
# print(X.shape)	;	print(y.shape)

model_Jabar.fit(X, y_Jabar)
model_Bengk.fit(X, y_Bengk)
model_jumlah.fit(X, y_Indo)

print('Prediksi jumlah penduduk {} di tahun 2050:'.format(Nama_Prov[0]), int(model_Jabar.predict([[2050]]).round()))
print('Prediksi jumlah penduduk {} di tahun 2050:'.format(Nama_Prov[1]), int(model_Bengk.predict([[2050]]).round()))
print('Prediksi jumlah penduduk {} di tahun 2050:'.format(Nama_Prov[2]), int(model_jumlah.predict([[2050]]).round()))

y_pred_jabar = model_Jabar.predict(X)
y_pred_bengk = model_Bengk.predict(X)
y_pred_indo = model_jumlah.predict(X)


# =============================================
# Plotting
# =============================================
plt.figure('soal1_2')
plt.plot(list_tahun_col, max_2010_val, 'g', marker='o')
plt.plot(list_tahun_col, min_1971_val, 'b', marker='o')
plt.plot(list_tahun_col, list_jumlah, 'r', marker='o')

plt.plot(list_tahun_col, y_pred_jabar, 'y')
plt.plot(list_tahun_col, y_pred_bengk, 'y')
plt.plot(list_tahun_col, y_pred_indo, 'y')

plt.title('Jumlah Penduduk INDONESIA (1971-2010)')
plt.xlabel('Tahun')
plt.ylabel('Jumlah penduduk (ratus juta jiwa)')
plt.legend(Nama_Prov)

plt.grid(True)
plt.show() 