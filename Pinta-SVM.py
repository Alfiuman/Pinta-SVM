
import pandas as pd
import numpy as np
import random 
import itertools
import copy
from sklearn import svm
from sklearn.externals import joblib


class SVM:
    
    #Construtor.
    def __init__(self, file = 0, c = 1.0, algo = 0, test_perc_size = 0.3, seed = 0, pickled = 0):
        
        if pickled != 0:
            self.svm = joblib.load(pickled) 
        else:
            if algo == 0:
                self.svm = svm.SVC(c)  #rbf
            elif algo == 1:
                self.svm = svm.SVC(c, kernel = 'poly')
            elif algo == 2:
                self.svm = svm.SVC(c, kernel = 'linear')
            elif algo == 3:
                if c >= 1.0 or c <= 0.0:
                    c = 0.5
                self.svm = svm.NuSVC(c)  #rbf
            elif algo == 4:
                if c >= 1.0 or c <= 0.0:
                    c = 0.5
                self.svm = svm.NuSVC(c, kernel = 'poly')
            elif algo == 5:
                if c >= 1.0 or c <= 0.0:
                    c = 0.5
                self.svm = svm.NuSVC(c, kernel = 'linear')
            else:
                self.svm = svm.LinearSVC(C = c)
                
            if file == 0:
                print("You need to specify a file with data.")
        
        #Prepare the data.
        if file != 0:
            self.train_and_test = 1
            self.test_perc_size = test_perc_size
            self.dataframe = pd.read_csv(file)
            self.dataframe.replace('?',-999999, inplace = True)
            self.dataframe.drop(self.dataframe.columns[[0]], 1, inplace = True)
            self.data_complete = self.dataframe.astype(float).values.tolist()
            
            random.seed(seed)
            random.shuffle(self.data_complete)
            self.train_data = self.data_complete[:int((1 - self.test_perc_size) * len(self.data_complete))]
            self.test_data = self.data_complete[int((1 - self.test_perc_size) * len(self.data_complete)):]
            
            self.train_data_y = list(itertools.chain.from_iterable([sublist[slice(len(self.train_data[0]) - 1,len(self.train_data[0]))] for sublist in self.train_data]))
            self.train_data_x = [sublist[slice(0,len(self.train_data[0]) - 1)] for sublist in self.train_data]
            
            self.test_data_y = list(itertools.chain.from_iterable([sublist[slice(len(self.test_data[0]) - 1,len(self.test_data[0]))] for sublist in self.test_data]))
            self.test_data_x = [sublist[slice(0,len(self.test_data[0]) - 1)] for sublist in self.test_data]
            
            self.y_values = list(set(self.test_data_y))
        else:
            self.train_and_test = 0

    #Training the model.
    def training(self):
        
        if self.train_and_test == 1:
            return self.svm.fit(self.train_data_x, self.train_data_y)
        else:
            print("You haven't specified data for training.")
            return self.svm
    
    #Testing the data.
    def testing(self):
        
        if self.train_and_test == 1:
            classif_test = self.svm.predict(self.test_data_x)
            test_check = [None]*len(classif_test)
            corr = 0
            
            for i in range(len(classif_test)):
                if classif_test[i] == self.test_data_y[i]:
                    test_check[i] = 1
                    corr += 1
                else:
                    test_check[i] = 0
            
            perc_corr = corr / len(classif_test)
            
            return classif_test, test_check, perc_corr
        else:
            print("You haven't specified data for testing.")
            return np.ndarray([0]), [0], 0
    
    #Classifying data.
    def classify(self, classiFile):
        
        self.dataframe2 = pd.read_csv(classiFile)
        self.dataframe2.replace('?',-99999, inplace = True)
        self.dataframe3 = (copy.deepcopy(self.dataframe2)).astype(float).values.tolist()
        self.dataframe2.drop(self.dataframe2.columns[[0]], 1, inplace = True)
        self.data_to_class = self.dataframe2.astype(float).values.tolist()
        
        return self.svm.predict(self.data_to_class)

    #Exporting the model using pickle.
    def export_model(self, nameFile = "svcTrained.pkl"):
        
        joblib.dump(self.svm, nameFile) 
        

####################################################################################

a = SVM("breast-cancer-wisconsin.data.txt", algo = 0, seed = 0)  
#a = Svm(algo = 0, pickled = "svcTrained.pkl")  

b = a.__dict__

a.training()

test_results = a.testing()

print(test_results[0])
print(test_results[1])
print("Correct percentage: ", test_results[2] * 100, "%", sep = "")

print(a.classify("ToClassProva.txt"))

a.export_model()







