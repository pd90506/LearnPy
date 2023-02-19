#%%
%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
fruits = pd.read_table("fruit_data_with_colors.txt")
fruits.head()
# %%
names = fruits["fruit_name"].unique()
print(names)
print(fruits.shape)
# %%
print(fruits.groupby("fruit_name").size())
# %%
import seaborn as sns
sns.countplot(fruits["fruit_name"], label="Count")
plt.show
# %%
fruits.drop('fruit_label', axis=1).plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False, figsize=(9,9), 
                                       title='Box Plot for each input variable')
plt.savefig('fruits_box')
plt.show()
# %%
fruits.drop("fruit_label", axis=1).hist(bins=30, figsize=(9,9))
plt.suptitle("Histogram for each numeric input variable")
plt.savefig('fruits_hist')
plt.show()

# %%
from pandas.plotting import scatter_matrix
from matplotlib import cm
feature_names = ['mass', 'width', 'height', 'color_score']
X = fruits[feature_names]
y = fruits['fruit_label']
cmap = cm.get_cmap('gnuplot')
scatter = scatter_matrix(X, c = y, marker = 'o', 
    s=40, hist_kwds={'bins':15}, figsize=(9,9), cmap=cmap)
plt.suptitle("Scatter-matrix for each input variable")
plt.savefig('fruits_scatter_matrix')
# %%
