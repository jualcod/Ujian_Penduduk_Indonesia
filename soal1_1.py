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
]
# =============================================
# Plotting
# =============================================
X 		= list_tahun_col
list_y	= [max_2010_val, min_1971_val, list_jumlah]
list_color = ['g', 'b', 'r']

plt.figure('soal1_1')
plt.style.use('ggplot')

for e, rgb in zip(list_y, list_color):
	plt.plot(X, e, color=rgb, linestyle='-')
	plt.scatter(X, e, color=rgb)

plt.title('Jumlah Penduduk {} (1971-2010)'.format(Observation_Name[2]))
plt.xlabel('Tahun')
plt.ylabel('Jumlah penduduk (ratus juta jiwa)')
plt.legend(Observation_Name)

plt.grid(True)
plt.show() 
plt.clf()