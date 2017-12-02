
import pandas as pd
import random 
import itertools
import copy
from sklearn import svm
from sklearn.externals import joblib


class Svm:
    
    #Construtor.
    def __init__(self, file = 0, c = 1.0, algo = 0, testPercSize = 0.3, seed = 0, pickled = 0):
        
        if pickled != 0:
            self.supVecMach = joblib.load(pickled) 
        else:
            if algo == 0:
                self.supVecMach = svm.SVC(c)  #rbf
            elif algo == 1:
                self.supVecMach = svm.SVC(c, kernel = 'poly')
            elif algo == 2:
                self.supVecMach = svm.SVC(c, kernel = 'linear')
            elif algo == 3:
                if c >= 1.0 or c <= 0.0:
                    c = 0.5
                self.supVecMach = svm.NuSVC(c)  #rbf
            elif algo == 4:
                if c >= 1.0 or c <= 0.0:
                    c = 0.5
                self.supVecMach = svm.NuSVC(c, kernel = 'poly')
            elif algo == 5:
                if c >= 1.0 or c <= 0.0:
                    c = 0.5
                self.supVecMach = svm.NuSVC(c, kernel = 'linear')
            else:
                self.supVecMach = svm.LinearSVC(C = c)
                
            if file == 0:
                print("You need to specify a file with data.")
        
        #Prepare the data.
        if file != 0:
            self.trainAndTest = 1
            self.testPercSize = testPercSize
            self.dataFrame = pd.read_csv(file)
            self.dataFrame.replace('?',-999999, inplace = True)
            self.dataFrame.drop(self.dataFrame.columns[[0]], 1, inplace = True)
            self.dataComplete = self.dataFrame.astype(float).values.tolist()
            
            random.seed(seed)
            random.shuffle(self.dataComplete)
            self.trainData = self.dataComplete[:int((1 - self.testPercSize) * len(self.dataComplete))]
            self.testData = self.dataComplete[int((1 - self.testPercSize) * len(self.dataComplete)):]
            
            self.trainDataY = list(itertools.chain.from_iterable([sublist[slice(len(self.trainData[0]) - 1,len(self.trainData[0]))] for sublist in self.trainData]))
            self.trainDataX = [sublist[slice(0,len(self.trainData[0]) - 1)] for sublist in self.trainData]
            
            self.testDataY = list(itertools.chain.from_iterable([sublist[slice(len(self.testData[0]) - 1,len(self.testData[0]))] for sublist in self.testData]))
            self.testDataX = [sublist[slice(0,len(self.testData[0]) - 1)] for sublist in self.testData]
            
            self.yValues = list(set(self.testDataY))
        else:
            self.trainAndTest = 0

    #Training the model.
    def training(self):
        
        if self.trainAndTest == 1:
            return self.supVecMach.fit(self.trainDataX, self.trainDataY)
        else:
            print("You haven't specified data for training.")
            return self.supVecMach
    
    #Testing the data.
    def testing(self):
        
        if self.trainAndTest == 1:
            self.classifTest = self.supVecMach.predict(self.testDataX)
            self.testCheck = [None]*len(self.classifTest)
            self.corr = 0
            
            for i in range(len(self.classifTest)):
                if self.classifTest[i] == self.testDataY[i]:
                    self.testCheck[i] = 1
                    self.corr += 1
                else:
                    self.testCheck[i] = 0
            
            self.percCorr = self.corr / len(self.classifTest)
            
            return self.testCheck, self.percCorr
        else:
            print("You haven't specified data for testing.")
            return 0, 0
    
    #Classifying data.
    def classify(self, classiFile):
        
        self.dataFrame2 = pd.read_csv(classiFile)
        self.dataFrame2.replace('?',-99999, inplace = True)
        self.dataFrame3 = (copy.deepcopy(self.dataFrame2)).astype(float).values.tolist()
        self.dataFrame2.drop(self.dataFrame2.columns[[0]], 1, inplace = True)
        self.dataToClass = self.dataFrame2.astype(float).values.tolist()
        
        self.classification = self.supVecMach.predict(self.dataToClass)
        
        return self.classification

    #Exporting the model using pickle.
    def exportModel(self, nameFile = "svcTrained.pkl"):
        
        joblib.dump(self.supVecMach, nameFile) 
        

####################################################################################

a = Svm("breast-cancer-wisconsin.data.txt", algo = 0, seed = 0)  
#a = Svm(algo = 0, pickled = "svcTrained.pkl")  

b = a.__dict__

a.training()

testResults = a.testing()

print(testResults[0])
print("Correct percentage: ", testResults[1] * 100, "%", sep = "")

print(a.classify("ToClassProva.txt"))

a.exportModel()







