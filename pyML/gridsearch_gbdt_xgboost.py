import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
import xgboost


from sklearn import datasets
iris = datasets.load_iris()
params = {'n_estimators': [1000, 500, 100], 'subsample': [0.5,0.8],
          'learning_rate': [0.01,0.05], 'min_samples_leaf': [2,10]}
gbdt0 = GradientBoostingClassifier()
clf = GridSearchCV(gbdt0, params, cv=5)
clf.fit(iris.data, iris.target)

#clf.best_params_
#clf.best_score_


## gbdt best model
gbdt = GradientBoostingClassifier(learning_rate=0.01, min_samples_leaf=2, n_estimators=100, subsample=0.8)
gbdt.fit(iris.data, iris.target)
gbdt.predict(iris.data)
gbdt.predict_proba(iris.data)


# ## xgboost  test
def scorebyself(self, X, y):
       from sklearn.metrics import roc_auc_score
       probas = self.predict_proba(X)
       auc = roc_auc_score(y, probas)
       return auc

from xgboost import XGBClassifier as xgb

params = {'n_estimators': [1000, 500, 100], 'subsample': [0.5,0.8], 'learning_rate': [0.01,0.05]}
gsmodel = xgb()
xgbmodel0 = GridSearchCV(gsmodel, params, cv=5, n_jobs=5)
xgbmodel0.fit(iris.data, iris.target)
xgbest = xgb(learning_rate=0.01, n_estimators=1000, subsample=0.5, max_depth=3)
xgbest.fit(iris.data, iris.target)
xgbest.predict(iris.data)
xgbest.predict_proba(iris.data)




