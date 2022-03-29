import pickle
import sys
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import writeprintsStatic.writeprintsStatic as ws
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import VarianceThreshold
import numpy as np
from EnsembleConstructor import getEnsemble1, getEnsemble2, getEnsemble3
from sklearn.pipeline import make_pipeline
from mlxtend.feature_selection import ColumnSelector
import util

# Parameter Setting
try:
    datasetName = str(sys.argv[1])
    authorsRequired = int(sys.argv[2])
except:
    datasetName, authorsRequired = 'amt', 5

def main():
    X_train, X_test, y_train, y_test = util.loadData(datasetName, authorsRequired)

    cs = ColumnSelector(cols=util.featuresToKeep())
    vt = VarianceThreshold()
    total_features = len(vt.fit_transform(cs.fit_transform(X_train))[0])

    clf = getEnsemble1(total_features)
    clf = make_pipeline(cs, vt, clf)

    print("Starting Training...")
    clf.fit(X_train, y_train)
    print("Starting Saving...")

    if not os.path.exists("trainedModels/" + datasetName + "-" + str(authorsRequired) + "/" ):
        os.makedirs("trainedModels/" + datasetName + "-" + str(authorsRequired) + "/")

    filename = "trainedModels/" + datasetName + "-" + str(authorsRequired) + "/" + f'trained_model.sav'
    pickle.dump(clf, open(filename, 'wb'))

    # load the model from disk
    loaded_model = pickle.load(open(filename, 'rb'))

    print(accuracy_score(y_test, loaded_model.predict(X_test)))

if __name__ == '__main__':
    main()
