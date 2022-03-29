from tqdm.auto import tqdm
import subprocess
import time
from random import randint
import numpy as np
from math import ceil
import os
import pickle
import writeprintsStatic.writeprintsStatic as ws
from matplotlib import pyplot as plt
import matplotlib

labels = open('writeprintsStatic/writeprintresources/feature_names.txt').read().split('\n')
global_space = ['totalWords', 'averageWordLength', 'noOfShortWords', 'charactersCount', 'digitsPercentage', 'upperCaseCharactersPercentage']

def clean_obfuscated_file(f):
    return '\n'.join(f.split('\n')[:-1]).strip()

def amt5Authors():
    return ['h', 'm', 'pp', 'qq', 'y']

def simulateRun(M, t, tm, n):
    for i in tqdm(range(1, 300, 10)):
        for j in range(i, i + n):
            subprocess.Popen(f"python3 Obfuscator.py --indexNumber {j} -M {M} -tmindex {tm} > logs/log{j}.txt 2> temp", shell=True)
        time.sleep(t * 60) # sleep for 60minutes

def subspaceLabels(features, labels, subspace):
    return [value for value, label in zip(features, labels) if label in subspace or label in global_space]

def getSubspaceColumns(subspace):
    return [ind for ind, label in enumerate(labels) if label == subspace or label in global_space]

def featuresToKeep():
    return [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 14, 15, 20, 21, 24, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 129, 139, 140, 145, 168, 170, 174, 188, 193, 196, 224, 228, 231, 232, 233, 237, 241, 247, 251, 257, 261, 262, 269, 282, 284, 288, 290, 295, 298, 300, 304, 308, 314, 324, 325, 339, 347, 350, 352, 360, 367, 369, 375, 392, 397, 401, 417, 438, 439, 440, 444, 452, 457, 459, 465, 471, 473, 477, 480, 485, 492, 493, 496, 497, 498, 503, 510, 526, 527, 528, 529, 532, 533, 534, 535, 536, 537, 538, 539, 540, 542, 543, 544, 545]

def featureSubspaces():
    return ['frequencyOfSpecialCharacters', 'frequencyOfLetters', 'frequencyOfDigits', 'mostCommonLetterBigrams', 'mostCommonLetterTrigrams', 'legomena', 'functionWordsPercentage', 'posTagFrequency', 'frequencyOfPunctuationCharacters']

def l(zj, y):
    return np.count_nonzero(zj == y)

def entropy(y_true, y_preds):
    L, N = y_preds.shape
    
    coef = 1 / (L - ceil(L / 2))
    
    add = 0
    for j in range(N):
        lzj = l(y_preds[:,j], y_true[j])
        add += coef * min(lzj, L - lzj)
    
    return (1 / N) * add

def prediction_matrix(x, individual):
        return np.matrix([i.predict(x) for i in individual])

def loadData(datasetName, authorsRequired, basePath = '../datasets/'):

    def getData():
        picklesPath = basePath + datasetName + "-" + str(authorsRequired) + "/"
        with open(picklesPath+'X_train.pickle', 'rb') as handle:
            X_train = pickle.load(handle)

        with open(picklesPath+'X_test.pickle', 'rb') as handle:
            X_test = pickle.load(handle)

        return (X_train, X_test)

    def getAllData():
        (X_train_all, X_test_all) = getData()
        X_train = []
        y_train = []

        X_test = []
        y_test = []
        print("Getting Training Data")
        for (filePath, filename, authorId, author, inputText) in X_train_all:
            features = getFeatures(inputText)
            X_train.append(features)
            y_train.append(authorId)
        print("Getting Testing Data")
        for (filePath, filename, authorId, author, inputText) in X_test_all:
            features = getFeatures(inputText)
            X_test.append(features)
            y_test.append(authorId)

        return np.matrix(X_train), np.matrix(X_test), y_train, y_test

    def getFeatures(inputText):
        return ws.calculateFeatures(inputText)

    if os.path.exists(f'{datasetName}-{authorsRequired}.data.pkl'):
        return pickle.load(open(f'{datasetName}-{authorsRequired}.data.pkl', 'rb'))
    
    return getAllData()
    
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.
    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """
    if not ax:
        ax = plt.gca()
    # Plot the heatmap
    im = ax.imshow(data, **kwargs)
    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")
    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)
    return im


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.
    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()
    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.
    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)
    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)
    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)
    return texts
