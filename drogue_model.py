from pandas import pd
from sklearn.ensemble import RandomForestRegressor
import pickle

df = pd.read_csv('/content/drug_consumption.data', header=None)
df.columns=(['ID','Age','Gender','Education','Country','Ethnicity','Nscore','Escore','Oscore','Ascore','Cscore','Impulsive','SS','Alcohol','Amphet','Amyl','Benzos','Caff','Cannabis','Choc','Coke','Crack','Ecstasy','Heroin','Ketamine','Legalh','LSD','Meth','Mushrooms','Nicotine','Semer','VSA'])
df=df.replace({'CL': ''}, regex=True)
for column in ['Alcohol','Amphet','Amyl','Benzos','Caff','Cannabis','Choc','Coke','Crack','Ecstasy','Heroin','Ketamine','Legalh','LSD','Meth','Mushrooms','Nicotine','Semer','VSA']:
    df[column]=df[column].astype('int32')
df.dtypes

y_train=df['Cannabis']
X_train=df.drop('Alcohol','Amphet','Amyl','Benzos','Caff','Cannabis','Choc','Coke','Crack','Ecstasy','Heroin','Ketamine','Legalh','LSD','Meth','Mushrooms','Nicotine','Semer','VSA',axis=1)
model = RandomForestRegressor(n_estimators=857, min_samples_split=5,
                                  min_samples_leaf=1, max_features="sqrt", max_depth=110,
                                  bootstrap=False)
okk=model.fit(X_train, y_train)
pickle.dump(okk, open("drug.pkl","wb"))