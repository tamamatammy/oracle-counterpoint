import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import entropy
from sklearn.model_selection import KFold


def cleanDataFrame(df: pd.DataFrame):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(axis=0, how="any", inplace=True)
    return df


def computeMutualInformation(df: pd.DataFrame, targetColumn: str):
    """targetColumn is a column of df"""
    target = df[targetColumn]
    features = df.drop([targetColumn], axis=1)

    mi = mutual_info_regression(features, target)
    targetEntropy = entropy(target)

    normalizedMI = mi / targetEntropy
    return normalizedMI


def RMSDif(vec: np.array, anchor: float):
    assert len(vec.shape) == 1 or vec.shape[1] == 1
    squaredDif = np.multiply(vec - anchor, vec - anchor)
    rmsd = np.sqrt(np.sum(squaredDif) / len(squaredDif))
    return rmsd


def computeMIStability(df: pd.DataFrame, targetColumn: str, numFolds: int):
    """compare stability across given number of folds of the dataset
    Output = dict of tuples (MI, RMSD / MI) where RMSD comes from k-fold calculation on the dataset
    """
    df = cleanDataFrame(df)
    features = df.drop([targetColumn], axis=1).columns

    mi = computeMutualInformation(df, targetColumn)
    MIDict = {}
    if numFolds<2:
        for i in range(len(features)):
            MIDict[features[i]] = mi[i]
        return MIDict
    
    elif numFolds>=2 :
        mi_kfold = np.zeros((numFolds, len(features)))
        kf = KFold(n_splits=numFolds)
        i = 0
        for train, test in kf.split(df):
            mi_kfold[i, :] = computeMutualInformation(df.iloc[train], targetColumn)
            i += 1
            
        MIStability = {}
        for i in range(len(features)):
            rmsd = RMSDif(mi_kfold[:, i], mi[i])
            MIStability[features[i]] = rmsd / mi[i]
            MIDict[features[i]] = mi[i]

        return MIDict, MIStability
