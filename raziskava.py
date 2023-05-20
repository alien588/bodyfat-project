import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,cross_val_predict, cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer, r2_score, PredictionErrorDisplay


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
#plt.savefig('porazdelitve_podatkov.jpg', format= 'jpeg')

# Create new features from existing ones (Density and Abdomen-Hip ratio)
data['density [g/cm^3]'] = (data['weight [kg]'] * (4 * np.pi * 1000) )/ data['hip [cm]'] * data['hip [cm]'] * data['height [cm]'] 
# Normalize density
data['density [g/cm^3]'] = data['density [g/cm^3]'] / data['density [g/cm^3]'].max()

data['abdomen-hip ratio [_]'] = (data['abdomen [cm]'] / data['hip [cm]'])
# Normalize the ratio
#data['abdomen-hip ratio [_]'] = data['abdomen-hip ratio [_]']/data['abdomen-hip ratio [_]'].abs().max()

new_features = ['density [g/cm^3]', 'abdomen-hip ratio [_]']

X_all = data.loc[:, data.columns != 'body fat [%]']
Y_all = data['body fat [%]']

# Normalize the entire data set
for i in X_all:
   X_all[i] = X_all[i]/X_all[i].abs().max()

# Get the columns with high correlation with other columns
X_all.corr().applymap(lambda x: x if abs(x)>.80 else "")
corr = X_all.corr().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
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
#X_most_corr = X_most_corr.drop('body fat [%]', axis= 1)
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
plt.savefig('regresija_rezultati_1.jpg', format= 'jpeg')
plt.show()

# Plotting a scatter plot in 2d where 2 of the most correlated features represent the axis and
# the hue represent the value of the target variable

subsets = [data['weight [kg]'], data['hip [cm]']]

plt.scatter(subsets[0], subsets[1], c= data['body fat [%]'], cmap= 'viridis')
plt.xlabel("Weight [kg]")
plt.ylabel("Hip [cm]")
plt.colorbar()
plt.savefig('barvi_graf_now_with_labels.jpg', format= 'jpeg')
plt.show()


# Implement cross validation and add additional result measuring metrics

reg_final_2 = KNeighborsRegressor(n_neighbors= knn_corr_best_num)
reg_final_3 = KNeighborsRegressor(n_neighbors= knn_best_new)

kfold = KFold(n_splits= 10, shuffle= True, random_state= 50)

scoring = {'RMSE': make_scorer(mean_squared_error, squared=False),
           'R-squared': make_scorer(r2_score)}

results_all = cross_val_score(reg_final_1, X_all, Y_all, cv= kfold, scoring='neg_mean_squared_error')
results_corr = cross_val_score(reg_final_2, X_most_corr, Y_all, cv= kfold, scoring= 'neg_mean_squared_error')
results_new = cross_val_score(reg_final_3, X_most_corr, Y_all, cv= kfold, scoring= 'neg_mean_squared_error')

results_all = np.sqrt((-1)*results_all)
results_corr = np.sqrt((-1)*results_corr)
results_new = np.sqrt((-1)*results_new)
#rmse_scores = results_all['test_RMSE']
print("All features RMSE scores:", results_all.mean())
print("Corralated features RMSE scores:", results_corr.mean())
print("New features RMSE scores:", results_new.mean())


results_all = cross_val_score(reg_final_1, X_all, Y_all, cv= kfold, scoring='r2')
results_corr = cross_val_score(reg_final_2, X_most_corr, Y_all, cv= kfold, scoring= 'r2')
results_new = cross_val_score(reg_final_3, X_most_corr, Y_all, cv= kfold, scoring= 'r2')

print("All features R^2 scores:", results_all.mean())
print("Corralated features R^2 scores:", results_corr.mean())
print("New features R^2 scores:", results_new.mean())
