
#导入模块
import pandas as pd
import numpy as np
from collections import Counter
from math import log2


#导入并处理数据
def getData(filePath):
    data = pd.read_excel(filePath)
    return data
def dataDeal(data):
    dataList = np.array(data).tolist()
    dataSet = [element[1:] for element in dataList]
    return dataSet
# getData()通过pandas模块中的read_excel()函数读取样本数据。
# dataDeal()函数将dataframe转换为list，并且去掉了编号列。因为编号列不是西瓜的属性。但如果把它当属性的话，可获得最大信息增益。

#获取属性名称
def getLabels(data):
    labels = list(data.columns)[1:-1]
    return labels


#获取类别标记
def targetClass(dataSet):
    classification = set([element[-1] for element in dataSet])
    return classification 
#获取一个样本是否为好瓜的标记


#将分支结点标记为叶结点，选择样本数最多的类作为类标记
def majorityRule(dataSet):
    mostKind = Counter([element[-1] for element in dataSet]).most_common(1)
    majorityKind = mostKind[0][0]
    return majorityKind


#计算信息熵
def infoEntropy(dataSet):
    classColumnCnt = Counter([element[-1] for element in dataSet])
    Ent = 0
    for symbol in classColumnCnt:
        p_k = classColumnCnt[symbol]/len(dataSet)
        Ent = Ent-p_k*log2(p_k)
    return Ent


#子数据集构建
def makeAttributeData(dataSet,value,iColumn):
    attributeData = []
    for element in dataSet:
        if element[iColumn]==value:
            row = element[:iColumn]
            row.extend(element[iColumn+1:])
            attributeData.append(row)
    return attributeData


#计算信息增益
def infoGain(dataSet,iColumn):
    Ent = infoEntropy(dataSet)
    tempGain = 0.0
    attribute = set([element[iColumn] for element in dataSet])
    for value in attribute:
        attributeData = makeAttributeData(dataSet,value,iColumn)
        tempGain = tempGain+len(attributeData)/len(dataSet)*infoEntropy(attributeData)
        Gain = Ent-tempGain
    return Gain


#选择最优属性                
def selectOptimalAttribute(dataSet,labels):
    bestGain = 0
    sequence = 0
    for iColumn in range(0,len(labels)):#不计最后的类别列
        Gain = infoGain(dataSet,iColumn)
        if Gain>bestGain:
            bestGain = Gain
            sequence = iColumn
        print(labels[iColumn],Gain)
    return sequence


#建立决策树
def createTree(dataSet,labels):
    classification = targetClass(dataSet) #获取类别种类（集合去重）
    if len(classification) == 1:
        return list(classification)[0]
    if len(labels) == 1:
        return majorityRule(dataSet)#返回样本种类较多的类别
    sequence = selectOptimalAttribute(dataSet,labels)
    print(labels)
    optimalAttribute = labels[sequence]
    del(labels[sequence])
    myTree = {optimalAttribute:{}}
    attribute = set([element[sequence] for element in dataSet])
    for value in attribute:
        
        print(myTree)
        print(value)
        subLabels = labels[:]
        myTree[optimalAttribute][value] = \
                createTree(makeAttributeData(dataSet,value,sequence),subLabels)
    return myTree


#定义主函数
def main():
    filePath = 'D:\\machine_learning\\xiguadataset.xls'
    data = getData(filePath)
    dataSet = dataDeal(data)
    labels = getLabels(data)
    myTree = createTree(dataSet,labels)
    return myTree


#生成树
if __name__ == '__main__':
    myTree = main()
    
    
#结果显示   
    print(myTree)
