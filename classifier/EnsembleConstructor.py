from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from mlxtend.classifier import EnsembleVoteClassifier, StackingClassifier
from mlxtend.feature_selection import ColumnSelector
import util
import random


TOTAL_HIDDEN_CLFS = 10

def getRFC():
    rfc = RandomForestClassifier(random_state=10, n_estimators=50)
    return rfc

def makePipeline(total_features):
    pipe = make_pipeline(
        ColumnSelector(cols=random.sample(range(total_features), 30)),
        SVC(kernel='linear', probability=True, random_state=0)
        )
    return pipe

def getHiddenClfs(total_features):
    clfs = []
    for _ in range(TOTAL_HIDDEN_CLFS):
        clfs.append(makePipeline(total_features))
    return clfs

def getMetaClf(total_features):
    # return majority voting for HiddenClfs
    hidden_clfs = getHiddenClfs(total_features)

    meta_pipeline = make_pipeline(
        StandardScaler(),
        StackingClassifier(hidden_clfs, meta_classifier=LogisticRegression(multi_class='auto', solver='lbfgs'), use_probas=True, use_clones=False))

    return meta_pipeline

def getEnsemble1(total_features):
    hidden_clfs = getHiddenClfs(total_features)
    
    return make_pipeline(
        StandardScaler(),
        VarianceThreshold(),
        EnsembleVoteClassifier(hidden_clfs, voting='soft', use_clones=False)
    )


def getEnsemble2(total_features):
    hidden_clfs = getHiddenClfs(total_features)
    
    return make_pipeline(
        StandardScaler(),
        VarianceThreshold(),
        StackingClassifier(hidden_clfs, meta_classifier=LogisticRegression(multi_class='auto', solver='lbfgs'), use_probas=True, use_clones=False)
    )

def getEnsemble3(total_features):
    return EnsembleVoteClassifier([getEnsemble2(total_features), getRFC()], voting='soft')
