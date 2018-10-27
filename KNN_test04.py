# -*- coding: utf-8 -*-
import numpy as np
import operator
from os import listdir
from sklearn.neighbors import KNeighborsClassifier as KNN
"""
函数说明:将32x32的二进制图像转换为1x1024向量。
Parameters:
    filename - 文件名
Returns:
    returnVect - 返回的二进制图像的1x1024向量
"""
def img2vector(filename):
     #创建1x1024零向量
    returnVector=np.zeros((1,1024))
    fr=open(filename)
    #按行读取
    for i in range(32):
        lineStr=fr.readlines()
         #每一行的前32个元素依次添加到returnVect中
         for j in range(32):
             returnVector[0,32*i+j]=int(lineStr[j])
    return returnVector
def handwritingclassTest():
    hwLables=[]
    #os.listdir(path)返回指定路径下的文件和文件夹名字列表
    traingFilrList=listdir("trainingDigits")
    m=len(traingFilrList)
    traingMat=np.zeros((m,1024))
    #从文件名中解析出训练集的类别
    for i in range(m):
        #获得文件的名字
        fileNameStr=traingFilrList[i]
        #方法split()以空格为分隔符将字符串拆分成多个部分储存到一个列表中
        hwLables.append(fileNameStr.split('_')[0])
        #将每一个文件的1x1024数据存储到trainingMat矩阵中
        traingMat[i,:]=img2vector(fileNameStr)
        #http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        ##构建kNN分类器
    neigh=KNN(n_neighbors=3,algorithem='auto')
        #使用X作为训练数据并使用y作为目标值来拟合模型
    neigh.fit(traingMat,hwLables)
         #返回testDigits目录下的文件列表
    testFileList=listdir("testDigits")
    mTest=len(testFileList)
    errorCount=0.0
    #从文件中解析出测试集的类别并进行分类测试
    for i range(mTest):
        #获得文件的名字
        fileNameStr=testFileList[i]
        #获得分类的数字
        classNumber=int(fileNameStr.split('_')[0])
        #获得测试集的1x1024向量,用于测试
        vectorUnderTest=img2vector(fileNameStr)
        #获得预测结果
        # classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        classifierResult=neigh.predict(vectorUnderTest)
        print("分类结果为：%d\t真实结果为：%d"%d(classifierResult,classNumber))
        if classifierResult!=classNumber:
            errorCount+=1.0
    print("总共错了%d个数据\n错误率为%f%%"%(errorCount,errorCount/mTest*100)
            

             