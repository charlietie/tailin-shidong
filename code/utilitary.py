from keras import backend as K

import numpy as np
import os
import sys


#-------------------------------------------------------------------------------------------------------
# Utilitary variables
#-------------------------------------------------------------------------------------------------------
# Gives the association between the original labels and the labels used for the training of the network
labelsTable = {
    '0': 0,
    '406516': 1,
    '406517': 2,
    '404516': 3,
    '404517': 4,
    '406520': 5,
    '404520': 6,
    '406505': 7,
    '404505': 8,
    '406519': 9,
    '404519': 10,
    '406511': 11,
    '404511': 12,
    '406508': 13,
    '404508': 14,
    '408512': 15,
    '407521': 16,
    '405506': 17
}

nbClasses = len(labelsTable)

# Input structure for model params
class modelParam():

    def __init__(self,name,params):
        self.name = name # str
        self.params = params # dict

    def getModelName(self):
        print('Model selected: %s' % (self.name))


#-------------------------------------------------------------------------------------------------------
# Function shuffleInUnisson : apply the same random permutation to 2 different arrays/vectors
#-------------------------------------------------------------------------------------------------------
def shuffleInUnisson(a,b): # a and b must be vectors or arrays with the same number of lines
    assert len(a) == len(b)
    randomPermutation = np.random.permutation(len(a))
    return a[randomPermutation], b[randomPermutation], randomPermutation

#-------------------------------------------------------------------------------------------------------
# Function invertPermutation : apply the inverse permutation to a permuted vector or array
#-------------------------------------------------------------------------------------------------------
def invertPermutation(a,perm): # a and b must be have the same number of lines
    assert len(a) == len(b)
    # Compute the inverse permutation
    inversePerm = np.zeros((len(perm)),dtype=int)
    for idx, e in enumerate(perm):
        inversePerm[e] = idx
    # Return the array with the inverse permutation
    return a[inversePerm]


#-------------------------------------------------------------------------------------------------------
# Function precision, recall, fbeta_score and fmeasure
#-------------------------------------------------------------------------------------------------------
def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta=1):
    """Computes the F score.

    The F score is the weighted harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.

    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.

    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    """
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def fmeasure(y_true, y_pred):
    """Computes the f-measure, the harmonic mean of precision and recall.

    Here it is only computed as a batch-wise average, not globally.
    """
    return fbeta_score(y_true, y_pred, beta=2)



#-------------------------------------------------------------------------------------------------------
# Function loadCleanOpportunityDataFromFolder : load the training, testing and validation data purged of NaN values
# from the OPPORTUNITY set.
# NOTE: the data and labels designated by dataFolder and labelsFolder are assumed to be under the .npy format
# NOTE 2: the data and labels from the .npy files are not shuffled
# NOTE 3: the labels have to projected in the range [0,nbClasses-1]
# NOTE 4: the dataFolder and labelsFolder input arguments can also take a list of .npy files as inputs
#-------------------------------------------------------------------------------------------------------
def loadCleanOpportunityDataFromFolder(dataFolder,labelsFolder,labelsTable=labelsTable,permute=True):
    
    ### Check if the input arguments are string or list of strings
    assert type(dataFolder) is type(labelsFolder)
    # If the input arguments are string of characters
    if isinstance(dataFolder,str):
        # List in order the .npy files in the folder and sort between data and labels
        sortedDataList = sorted(os.listdir(dataFolder))
        sortedLabelsList = sorted(os.listdir(labelsFolder))
        nbFiles = len(sortedDataList)
        assert len(sortedLabelsList) == nbFiles
        ### Allocate data and labels arrays
        # Compute the total number of examples
        nbExamplesInFiles = np.zeros((nbFiles),dtype=int)
        for idx in range(nbFiles):
            labels = np.load(labelsFolder+'/'+sortedLabelsList[idx])
            nbExamplesInFiles[idx] = len(labels)
        totalNbExamples = np.sum(nbExamplesInFiles)
        # Compute the dimension of examples
        firstDataExample = np.load(dataFolder+'/'+sortedDataList[0])
        height, width = firstDataExample[0].shape
        # Allocate the result arrays
        data = np.empty((totalNbExamples,height,width),dtype=np.float32)
        labels = np.empty((totalNbExamples),dtype=int)

        ### Concatenate the data and label files
        storageIdx = 0
        for idx in range(nbFiles):
            dataTmp = np.load(dataFolder+'/'+sortedDataList[idx])
            labelsTmp = np.load(labelsFolder+'/'+sortedLabelsList[idx])
            data[storageIdx:storageIdx+nbExamplesInFiles[idx]] = dataTmp
            labels[storageIdx:storageIdx+nbExamplesInFiles[idx]] = labelsTmp
            storageIdx += nbExamplesInFiles[idx]
  
    # Otherwise, they are list of string of characters
    else:
        sortedDataList = sorted(dataFolder)
        sortedLabelsList = sorted(labelsFolder)
        nbFiles = len(sortedDataList)
        assert len(sortedLabelsList) == nbFiles
        ### Allocate data and labels arrays
        # Compute the total number of examples
        nbExamplesInFiles = np.zeros((nbFiles),dtype=int)
        for idx in range(nbFiles):
            labels = np.load(sortedLabelsList[idx])
            nbExamplesInFiles[idx] = len(labels)
        totalNbExamples = np.sum(nbExamplesInFiles)
        # Compute the dimension of examples
        firstDataExample = np.load(sortedDataList[0])
        height, width = firstDataExample[0].shape
        # Allocate the result arrays
        data = np.empty((totalNbExamples,height,width),dtype=np.float32)
        labels = np.empty((totalNbExamples),dtype=int)

        ### Concatenate the data and label files
        storageIdx = 0
        for idx in range(nbFiles):
            dataTmp = np.load(sortedDataList[idx])
            labelsTmp = np.load(sortedLabelsList[idx])
            data[storageIdx:storageIdx+nbExamplesInFiles[idx]] = dataTmp
            labels[storageIdx:storageIdx+nbExamplesInFiles[idx]] = labelsTmp
            storageIdx += nbExamplesInFiles[idx]

    
    ### Shuffle in unisson labels and data
    if permute:
        data, labels, permutation = shuffleInUnisson(data,labels)
    else:
        permutation = list(range(len(labels)))

    ### Project labels in the range [0,nbClasses-1]
    for idx in range(totalNbExamples):
        labels[idx] = labelsTable[str(labels[idx])]

    ### Return the data and labels
    # Also returns information on the shape of the data
    return data, labels, data.shape, permutation



#-------------------------------------------------------------------------------------------------------
# Function loadCleanOpportunityData : load the training, testing and validation data purged of NaN values
# from the OPPORTUNITY set
# NOTE: the data and labels designated by dataFolder and labelsFolder are assumed to be under the .npy format
#-------------------------------------------------------------------------------------------------------
def loadCleanOpportunityData(fileFolder,labelsTable=labelsTable,permute=True):

    ### List the files in file folder, and then split them between training and testing sets | TODO: change to switch back to majoritary labels
    fileList = os.listdir(fileFolder)
    trainingDataList = [s for s in fileList if '_data' in s and ('ADL1' in s or 'ADL2' in s or 'ADL3' in s or 'Drill' in s)]
    trainingLabelsList = [s for s in fileList if '_labels' in s and not 'last' in s and ('ADL1' in s or 'ADL2' in s or 'ADL3' in s or 'Drill' in s)]
    testingDataList = [s for s in fileList if '_data' in s and ('ADL4' in s or 'ADL5' in s)]
    testingLabelsList = [s for s in fileList if '_labels' in s and not 'last' in s and ('ADL4' in s or 'ADL5' in s)]

    trainingDataList = [fileFolder+'/'+s for s in trainingDataList]
    trainingLabelsList = [fileFolder+'/'+s for s in trainingLabelsList]
    testingDataList = [fileFolder+'/'+s for s in testingDataList]
    testingLabelsList = [fileFolder+'/'+s for s in testingLabelsList]

    ### Extract the data and label files
    trainingData, trainingLabels, trainingShape, trainPermutation = loadCleanOpportunityDataFromFolder(trainingDataList,trainingLabelsList,permute=permute)
    testingData, testinglabels, testingShape, testPermutation = loadCleanOpportunityDataFromFolder(testingDataList,testingLabelsList,permute=permute)

    ### Return the data and labels
    # Also returns information on the shape of the data
    return trainingData, trainingLabels, trainingShape, testingData, testinglabels, testingShape, trainPermutation, testPermutation
