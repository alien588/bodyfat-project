import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import PredictionErrorDisplay

def load_data(dt):
    df = pd.read_csv(dt+".csv")

    return df


data = load_data('bodyfat')

# Check for na values
missing_v = data.isnull().sum()
print("The sum of missing values in our dataset is: " + str(missing_v.sum()))

for i in data:
    median = data[i].median()
    avg = data[i].mean()
    data[i].plot(kind= 'hist')
    plt.axvline(median,color= 'k', linewidth= 1)
    plt.axvline(avg, color= 'r', linewidth= 1)
    #plt.show()


# Create new features from existing ones (Density and Abdomen-Hip ratio)
data['density [g/cm^3]'] = np.pi * data['hip [cm]'] * data['hip [cm]'] * data['height [cm]'] / 1000
# Normalize density
data['density [g/cm^3]'] = data['density [g/cm^3]'] / data['density [g/cm^3]'].max()

data['abdomen-hip ratio [_]'] = (data['abdomen [cm]'] / data['hip [cm]'])
# Normalize the ratio
data['abdomen-hip ratio [_]'] = data['abdomen-hip ratio [_]']/data['abdomen-hip ratio [_]'].abs().max()

new_features = ['density [g/cm^3]', 'abdomen-hip ratio [_]']

# Normalize the entire data set
for i in data:
   data[i] = data[i]/data[i].abs().max()

# Get the columns with high correlation with other columns
data.corr().applymap(lambda x: x if abs(x)>.80 else "")
corr = data.corr().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
corr = corr[corr['level_0'] != corr['level_1']]
corr = corr[corr[0] > 0.80]
corr = corr.drop_duplicates(subset=['level_0', 'level_1'])

# Get columns with high kendall correlation with other columns
corr_kendall = data.corr('kendall').unstack().sort_values(kind= "quicksort", ascending= False).reset_index()
corr_kendall = corr_kendall[corr_kendall['level_0'] != corr_kendall['level_1']]
corr_kendall = corr_kendall.drop_duplicates(subset=['level_0', 'level_1'])

# Print the highly correlated features
corr = corr['level_0'].unique()
print(corr)


# Separate the target variable from the other ones
X_all = data.loc[:, data.columns != 'body fat [%]']
Y_all = data['body fat [%]']


X_train, X_test, Y_train, Y_test = train_test_split(X_all,Y_all,test_size= 0.3, random_state= 50)

knn_res = []

for i in range(175):
    regressor = KNeighborsRegressor(n_neighbors= i+1)
    regressor.fit(X_train,Y_train)
    knn_score = regressor.score(X_test,Y_test)
    knn_res.append(knn_score)
knn_best_num = knn_res.index(max(knn_res))
print('Knn best k: '+str(max(knn_res))+' '+str(knn_best_num))


X_most_corr = data[corr]
X_most_corr = X_most_corr.drop('body fat [%]', axis= 1)
#print(X_most_corr)

X_train, X_test, Y_train, Y_test = train_test_split(X_most_corr,Y_all,test_size= 0.3, random_state= 50)

knn_corr = []
for i in range(175):
    regressor2 = KNeighborsRegressor(n_neighbors= i+1)
    regressor2.fit(X_train,Y_train)
    knn_score = regressor2.score(X_test,Y_test)
    knn_corr.append(knn_score)
    #print(knn_score)
knn_corr_best_num = knn_corr.index(max(knn_corr))
print('Knn correlated best k: '+str(max(knn_corr))+' '+str(knn_corr_best_num))


X_new = data[new_features]
X_train, X_test, Y_train, Y_test = train_test_split(X_new,Y_all,test_size= 0.3, random_state= 50)

knn_new = []
for i in range(175):
    regressor3 = KNeighborsRegressor(n_neighbors= i+1)
    regressor3.fit(X_train,Y_train)
    knn_score = regressor3.score(X_test,Y_test)
    knn_new.append(knn_score)
    #print(knn_score)
knn_best_new = knn_new.index(max(knn_new))
print('Knn new features best k: '+str(max(knn_new))+' '+str(knn_best_new))

# Visualizing the best regressor results
reg_final_1 = KNeighborsRegressor(n_neighbors= knn_best_num)
reg_final_1.fit(X_train,Y_train)

y_p = reg_final_1.predict(X_test)

display = PredictionErrorDisplay.from_predictions(y_true= Y_test,y_pred= y_p,kind= "actual_vs_predicted")
plt.show()

