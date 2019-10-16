import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))

#Function to perform target encoding
def target_encode(trn_series,tst_series,target):
    min_samples_leaf=1 
    smoothing=1,
    noise_level=0
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean 
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply aver
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean 
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index 
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)

df = pd.read_csv('C:/Semester 1/Machine Learning/K/tcd ml 2019-20 income prediction training (with labels).csv', na_values = {
    'Year of Record': ["#NA"],
    'Gender': ["#NA","0","unknown"],
    'Age': ["#NA"],
    'Profession':["#NA"],
    'University Degree' : ["0","#NA"],
    'Hair Color': ["#NA","0","Unknown"]
 } )
dftest = pd.read_csv('C:/Semester 1/Machine Learning/K/tcd ml 2019-20 income prediction test (without labels).csv', na_values = {
    'Year of Record': ["#NA"],
    'Gender': ["#NA","0","unknown"],
    'Age': ["#NA"],
    'Profession':["#NA"],
    'University Degree' : ["0","#NA"],
    'Hair Color': ["#NA","0","Unknown"]
 } )

df['Year of Record'].fillna(df['Year of Record'].interpolate(method='slinear'), inplace=True)
df['Gender'].fillna('other', inplace=True)
df['Age'].fillna(df['Age'].interpolate(method='slinear'), inplace=True)
df['Gender'].fillna(method="ffill", inplace=True)
df['Profession'].fillna(method="ffill", inplace=True)
df['University Degree'].fillna(method="ffill", inplace=True)
df['Hair Color'].fillna(method="ffill", inplace=True)
dftest['Year of Record'].fillna(dftest['Year of Record'].interpolate(method='slinear'), inplace=True)
dftest['Gender'].fillna('other', inplace=True)
dftest['Age'].fillna(dftest['Age'].interpolate(method='slinear'), inplace=True)
dftest['Gender'].fillna(method="ffill", inplace=True)
dftest['Profession'].fillna(method="ffill", inplace=True)
dftest['University Degree'].fillna(method="ffill", inplace=True)
dftest['Hair Color'].fillna(method="ffill", inplace=True)

scale_age = pp.StandardScaler()
df['Age'] = scale_age.fit_transform(df['Age'].values.reshape(-1, 1))
dftest['Age'] = scale_age.transform(dftest['Age'].values.reshape(-1, 1))
scale_year = pp.StandardScaler()
df['Year of Record'] = scale_year.fit_transform(df['Year of Record'].values.reshape(-1, 1))
dftest['Year of Record'] = scale_year.transform(dftest['Year of Record'].values.reshape(-1, 1))

Var = df['Income in EUR']
Var = Var.abs()
df = df.drop(columns=['Income in EUR'])

professionList = df['Profession'].unique()
professionReplaced = df.groupby('Profession').count()
professionReplaced = professionReplaced[professionReplaced['Age'] < 3].index
df['Profession'].replace(professionReplaced, 'other profession', inplace=True)
countryList = df['Country'].unique()
countryReplaced = df.groupby('Country').count()
countryReplaced = countryReplaced[countryReplaced['Age'] < 3].index
df['Country'].replace(countryReplaced, 'other', inplace=True)

Var1 = dftest['Income']
dftest = dftest.drop(columns=['Income'])

testProfessionList = dftest['Profession'].unique()
encodedProfession = list(set(professionList) - set(professionReplaced))
testProfessionReplace = list(set(testProfessionList) - set(encodedProfession))
dftest['Profession'] = dftest['Profession'].replace(testProfessionReplace, 'other profession')
testCountryList = dftest['Country'].unique()
encodedCountries = list(set(countryList) - set(countryReplaced))
testCountryReplace = list(set(testCountryList) - set(encodedCountries))
dftest['Country'] = dftest['Country'].replace(testCountryReplace, 'other')
df['Gender'],dftest['Gender']=target_encode(df['Gender'],dftest['Gender'],Var)
df['University Degree'],dftest['University Degree']=target_encode(df['University Degree'],dftest['University Degree'],Var)
df['Hair Color'],dftest['Hair Color']=target_encode(df['Hair Color'],dftest['Hair Color'],Var)
df['Profession'],dftest['Profession']=target_encode(df['Profession'],dftest['Profession'],Var)
df['Country'],dftest['Country']=target_encode(df['Country'],dftest['Country'],Var)

X_train, X_test, y_train, y_test = train_test_split(df, Var, test_size=0.2, random_state=0)
regressor = RandomForestRegressor(n_estimators = 100, random_state = 42)  
regressor.fit(X_train, y_train) #training the algorithm
y_pred = regressor.predict(X_test)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
Instance = dftest['Instance']
Instance = pd.DataFrame(Instance, columns=['Instance'])
y_pred1 = regressor.predict(dftest)
Income = pd.DataFrame(y_pred1,columns=['Income'])
file = Instance.join(Income)
file.to_csv('results.csv',index=False)

