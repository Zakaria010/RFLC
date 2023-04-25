import os
import readline
import rlcompleter
import sys
import math
import random
import time
import csv

import numpy as np
import scipy as scp
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from os.path import join,getsize


readline.parse_and_bind('tab:complete')


################################################################################
# collecting trees from a scikit RF
################################################################################
def collectTreesRF(ensemble):
    n_nodes, children_left, children_right, feature, threshold, node_depth, is_leaves, stack, nodeValues = [], [], [], [], [], [], [], [], []
    for i in range(ensemble.n_estimators):
        nodeValues.append([])
        n_nodes.append(ensemble.estimators_[i].tree_.node_count)
        children_left.append(ensemble.estimators_[i].tree_.children_left)
        children_right.append(ensemble.estimators_[i].tree_.children_right)
        feature.append(ensemble.estimators_[i].tree_.feature)
        threshold.append(ensemble.estimators_[i].tree_.threshold)
    
        node_depth.append(np.zeros(shape=n_nodes[i], dtype=np.int64))
        is_leaves.append(np.zeros(shape=n_nodes[i], dtype=bool))
        for j in range(n_nodes[i]):
            nodeValues[i].append([])
        stack.append([(0, -1)])  # (root node id, parent depth)
        while len(stack[i]) > 0:
            node_id, parent_depth = stack[i].pop()
            node_depth[i][node_id] = parent_depth + 1
    
            if (children_left[i][node_id] != children_right[i][node_id]): # we have an internal node
                stack[i].append((children_left[i][node_id], parent_depth + 1))
                stack[i].append((children_right[i][node_id], parent_depth + 1))
            else: # we have a leave
                is_leaves[i][node_id] = True
                nodeValues[i][node_id] = ensemble.estimators_[i].tree_.value[node_id].tolist()[0]
    return n_nodes, children_left, children_right, feature, threshold, node_depth, is_leaves, nodeValues

#################################################################################
# exporting a tree collection
#################################################################################
def exportTreeCollection(datasetName, ensemble, runcount, numFeatures, numClasses, n_nodes, children_left, children_right, feature, threshold, node_depth, is_leave, nodeValues):
    maxTreeDepth = []
    for tree in range(len(n_nodes)):
        maxTreeDepth.append(max(node_depth[tree]))
    with open(datasetName + ".{}{}.txt".format(ensemble,runcount),"w+") as f:
        f.write("DATASET_NAME: " + datasetName.split('/')[-1] + ".train{}.csv\n".format(runcount))
        f.write("ENSEMBLE: " + ensemble +"\n")
        f.write("NB_TREES: %s\n" %(len(n_nodes)))
        f.write("NB_FEATURES: %s\n" %numFeatures)
        f.write("NB_CLASSES: %s\n" %numClasses)
        f.write("MAX_TREE_DEPTH: %s\n" %(max(maxTreeDepth)))
        f.write("Format: node / node type (LN - leave node, IN - internal node) left child / right child / feature / threshold / node_depth / majority class (starts with index 0)\n")
        f.write("\n")
        for tree in range(len(n_nodes)):
            f.write("[TREE %s]\n"%tree)
            f.write("NB_NODES: %s\n" %n_nodes[tree])
            for node in range(n_nodes[tree]):
                if is_leave[tree][node]:
                    tmpValues = nodeValues[tree][node]
                    f.write("%s LN -1 -1 -1 -1 %s %s" %(node, node_depth[tree][node], np.argmax(nodeValues[tree][node])))
                else:
                    f.write("%s IN %s %s %s %s %s -1" %(node, children_left[tree][node], children_right[tree][node],  feature[tree][node], threshold[tree][node], node_depth[tree][node]))
                f.write("\n")

            f.write("\n")
        f.close()

################################################################################
# prepare data
################################################################################
def prepareData(benchmarkIdentifier):
    targetData, featureData = [], []
    
    print ("Reading dataset " + benchmarkIdentifier)
    with open(benchmarkIdentifier, 'r', encoding='cp1252') as csvFile:
        csvReader = csv.reader(csvFile, delimiter = ',')
        linecount = 0
        for row in csvReader:
            try:
                if not linecount <= 1:
                    featureData.append([float(x) for x in row[:-1]])
                    targetData.append(int(row[-1]))
                if linecount == 0:
                    header = row
                if linecount == 1:
                    continuousFeatureList = row
            except:
                print('Invalid line', linecount, ':', row)
            linecount = linecount + 1
        numFeatures = len(continuousFeatureList)

    
    target_map = {c: i for i, c in enumerate(np.unique(np.array(targetData)))}
    targetData = [target_map[t] for t in targetData]
    numFeatures = len(continuousFeatureList)-1
    continuousFeaturesLabels = []
    for index in range(numFeatures):
        if continuousFeatureList[index] == 'F':
            continuousFeaturesLabels.append(index)
    
    return targetData, featureData, continuousFeaturesLabels, header, continuousFeatureList

################################################################################
# binning for continuous features
################################################################################
def binning(featureData, continuousFeatureList, numOfBins):
    numSamples, numFeatures = len(featureData), len(featureData[0])
    temporaryBucket, temporaryBucketBinned  = [], []
    enc = KBinsDiscretizer(n_bins = numOfBins, encode = 'ordinal')
    numContinuousFeatures = len(continuousFeatureList)
    #
    for index in range(numContinuousFeatures):
        temporaryBucket.append([])
        for sample in range(numSamples):
            temporaryBucket[index].append(featureData[sample][continuousFeatureList[index]])
        tmpNPArray = np.array(temporaryBucket[index])
        tmpFit = enc.fit_transform(tmpNPArray.reshape(-1,1))
        temporaryBucketBinned.append(tmpFit.tolist())
    for sample in range(numSamples):
        count = 0
        for feature in continuousFeatureList:
            featureData[sample][feature] = temporaryBucketBinned[count][sample][0]
            count = count + 1

    return featureData

################################################################################
# main
################################################################################
def process_dataset(dataset, numOfTrees, treeDepth, numOfRuns):
    targetData, featureData, continuousFeatureList, header, featureList = prepareData(dataset)
    #
    if len(continuousFeatureList) > 0:
        featureData = binning(featureData, continuousFeatureList, 10)
    
    targetDataNP, featureDataNP = np.array(targetData), np.array(featureData)
    # generate the relevant data sets
    numSamples = len(targetData) 
    filename = dataset[:-4]
    # save the full data set
    with open(filename + ".featurelist.csv","w") as f:
        writer = csv.writer(f)
        writer.writerow(featureList)
    with open(filename + ".full.csv","w") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for index in range(numSamples):
            tmpRow = featureData[index].copy()
            tmpRow.append(targetData[index])
            writer.writerow(tmpRow)
    # split the data
    kf = KFold(n_splits=int(numOfRuns), shuffle=True, random_state=42)
    kf.get_n_splits(featureData)
    #
    count = 1
    for train_index, test_index in kf.split(featureData):
        trainFeature, testFeature = featureDataNP[train_index].tolist(), featureDataNP[test_index].tolist()
        trainTarget, testTarget = targetDataNP[train_index].tolist(), targetDataNP[test_index].tolist()
    #
        with open(filename + ".test%s.csv"%(count),"w") as fTest, open(filename + ".train%s.csv"%(count),"w") as fTrain:
            testWriter = csv.writer(fTest)
            trainWriter = csv.writer(fTrain)
            testWriter.writerow(header)
            trainWriter.writerow(header)
            for index in range(len(testTarget)):
                tmpRow = testFeature[index].copy()
                tmpRow.append(testTarget[index])
                testWriter.writerow(tmpRow)
            for index in range(len(trainTarget)):
                tmpRow = trainFeature[index].copy()
                tmpRow.append(trainTarget[index])
                trainWriter.writerow(tmpRow)
        # generate RF
        rfModel = RandomForestClassifier(n_estimators = int(numOfTrees), max_depth = int(treeDepth), max_features = 0.5, random_state=42)
        # train RF
        rfModel.fit(trainFeature, trainTarget)
        #yp = rfModel.predict(testFeature)
        #print(np.count_nonzero(testTarget==yp)/len(yp))

        # collect Trees
        n_nodesRF, children_leftRF, children_rightRF, featureRF, thresholdRF, node_depthRF, is_leavesRF, nodeValuesRF = collectTreesRF(rfModel)
        # export the trees
#        exportTreeCollection(filename + ".depth%s.result%s"%(treeDepth,count), "RF" , len(featureData[0]), len(rfModel.classes_), n_nodesRF, children_leftRF, children_rightRF, featureRF, thresholdRF, node_depthRF, is_leavesRF, nodeValuesRF)
        exportTreeCollection(filename, "RF",  count, len(featureData[0]), len(rfModel.classes_),
                             n_nodesRF, children_leftRF, children_rightRF, featureRF, thresholdRF,
                             node_depthRF, is_leavesRF, nodeValuesRF)
  
        count = count + 1

################################################################################
#main(sys.argv[1],sys.argv[2],sys.argv[3], sys.argv[4])
