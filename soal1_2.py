import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# =============================================
# Establish Dataset, Pre-processing
# =============================================
# Load Data
df_1 = pd.read_excel('indo_12_1.xls')
df_1 = df_1.drop([0,1,2])
# Set Columns
df = df_1.iloc[0:34]
List_Column = ['Provinsi', 1971, 1980, 1990, 1995, 2000, 2010]

df = df.set_index(np.arange(0, 34))
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

Observation_Name = [
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

X = np.array([list_tahun_col]).reshape(-1,1)
y_Jabar = np.array(max_2010_val)
y_Bengk = np.array(min_1971_val)
y_Indo = np.array(list_jumlah)
# print(type(X))	; print(type(y))
# print(X.shape)	; print(y.shape)

model_Jabar.fit(X, y_Jabar)
model_Bengk.fit(X, y_Bengk)
model_jumlah.fit(X, y_Indo)

print('Prediksi jumlah penduduk {} di tahun 2050:'.format(Observation_Name[0]), int(model_Jabar.predict([[2050]]).round()))
print('Prediksi jumlah penduduk {} di tahun 2050:'.format(Observation_Name[1]), int(model_Bengk.predict([[2050]]).round()))
print('Prediksi jumlah penduduk {} di tahun 2050:'.format(Observation_Name[2]), int(model_jumlah.predict([[2050]]).round()))

y_pred_jabar = model_Jabar.predict(X)
y_pred_bengk = model_Bengk.predict(X)
y_pred_indo = model_jumlah.predict(X)

# =============================================
# Plotting
# =============================================
X 		= list_tahun_col
list_y	= [max_2010_val, min_1971_val, list_jumlah]
list_color = ['g', 'b', 'r']
list_y_pred = [y_pred_jabar, y_pred_bengk, y_pred_indo]

plt.figure('soal1_2')
plt.style.use('ggplot')

for e, rgb in zip(list_y, list_color):
	plt.plot(X, e, color=rgb, linestyle='-')
	plt.scatter(X, e, color=rgb)
for f in list_y_pred:
	plt.plot(X, f, color='y')

plt.title('Jumlah Penduduk {} (1971-2010)'.format(Observation_Name[2]))
plt.xlabel('Tahun')
plt.ylabel('Jumlah penduduk (ratus juta jiwa)')
plt.legend(Observation_Name)

plt.grid(True)
plt.show()
plt.clf()

"""
Terminal:
Prediksi jumlah penduduk {Obs0} di tahun 2050: 65443585
Prediksi jumlah penduduk {Obs1} di tahun 2050: 3139135
Prediksi jumlah penduduk {Obs2} di tahun 2050: 359127110
"""