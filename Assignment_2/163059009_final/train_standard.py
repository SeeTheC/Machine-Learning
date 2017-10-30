
# coding: utf-8

# In[16]:

import pandas;
import numpy as np;
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

np.set_printoptions(suppress=True)

def readCSVFile(file):
    data=pandas.read_csv(file,",",header=0, na_values='?', skipinitialspace=True);
    return data;
    pass;
def readTrainData(dataset):    
    return dataset.ix[:,1:-1], dataset.ix[:,-1:];
    pass;

def readTestData(dataset):    
    return dataset.ix[:,1:],dataset.ix[:,0:1];
    pass;

def normalizePhi(unNormalizedPhi,last_col_bias=False):    
    #assuming last column as bias column
    no_of_column=len(unNormalizedPhi[0]);
    phi=np.array(unNormalizedPhi);
    std=phi.std(0);
    mean=phi.mean(0);    
    std[no_of_column-1]=1;
    mean[no_of_column-1]=0;
    #phi_normalize=(phi-mean)/std;    
    
    max_vector=phi.max(axis=0)
    phi_normalize=phi/max_vector;    
    
    return phi_normalize;
    pass;

def writeTestData(test_id,ystar,filenumber=0,filename=None):
    if(filename==None):
        fo = open("log/output/sampleSubmission-"+str(filenumber)+".csv", "w");               
    else:
        fo = open(filename+".csv", "w");        
    fo.write("id,salary\n");
    m=len(ystar);
    for i in range(m):
        fo.write(str(test_id[i][0])+","+str(ystar[i])+"\n");
    fo.close();
    pass;

def dropColumns(dataframe,colList):
    for c in colList:
        dataframe.drop([c], axis = 1, inplace = True);
    pass;

def addColByCategory(dataset):
    return pandas.get_dummies(dataset);
    pass;

def categoryToNumber(dataset,categoryList):
    for c in categoryList:
        if (c in dataset):            
            dataset[c]=pandas.get_dummies(dataset[c]).values.argmax(1);        
    return dataset;
    pass;
    

def handleCategoryData(dataset,categoryList=None,byNumber=False):
    if(byNumber):
        return categoryToNumber(dataset,categoryList)
    else:
        return addColByCategory(dataset);
def findMostFrequentCount(dataset):
    #arr=dataset;
    #axis = 0
    #u, indices = np.unique(arr, return_inverse=True)
    #print(u);
    #u[np.argmax(np.apply_along_axis(np.bincount, axis, indices.reshape(arr.shape),None, np.max(indices) + 1), axis=axis)]
    x = np.array([0, 1, 1, 3, 2, 1, 2, 3]);
    w=np.bincount(dataset[:,1].astype(int))#work-class: Private
    o=np.bincount(dataset[:,6].astype(int))#occupation: Married-civ-spouse
    c=np.bincount(dataset[:,13].astype(int))#country : US
    pass;

def fillNanValue(dataframe,col,value):
    if (col in dataframe):
        dataframe[col].fillna(value, inplace=True);
    pass;

def imputeUnknowValue(dataframe):
    #by most frequent value;
    fillNanValue(dataframe,"workclass","Private");
    fillNanValue(dataframe,"occupation","Craft-repair");
    fillNanValue(dataframe,"native-country","United-States");
    pass;

def addRemainingCol(colList,dataframe,rowCount):
    i=0;
    for c in colList:
        if( c not in dataframe):
            dataframe.insert(i, c, 0);
        i+=1;
    pass;


# In[ ]:




# In[21]:

#--settings--
pandas.set_option('display.max_columns', None);
#---init---
dir=""
trainFile=dir+"train.csv";
testFile=dir+"kaggle_test_data.csv";
categoryList=["workclass","education","marital-status","occupation","relationship","race","sex","native-country"];
drop_col=['native-country',"race","education"]
trained_dataset=readCSVFile(trainFile);
trained_data,trained_y=readTrainData(trained_dataset);

test_dataset=readCSVFile(testFile);
test_data,test_id=readTestData(test_dataset);

#droping unrelated-columns
dropColumns(trained_data,drop_col);
dropColumns(test_data,drop_col);

#impute:
imputeUnknowValue(trained_data);
imputeUnknowValue(test_data);

#converting categorical data to point wise data
byNumber=False;
dummy_trained_data=handleCategoryData(trained_data,categoryList,byNumber);
dummy_test_data=handleCategoryData(test_data,categoryList,byNumber);


#adding missing column
trained_columns_name=list(dummy_trained_data.columns.values);
addRemainingCol(trained_columns_name,dummy_test_data,len(trained_data))
test_columns_name=list(dummy_test_data.columns.values);

#converting panda data frame to numpy martix
mtx_dummy_tds=dummy_trained_data.as_matrix(columns=None)
mtx_dummy_testds=dummy_test_data.as_matrix(columns=None)
mtx_trained_y=trained_y.as_matrix(columns=None);
mtx_test_id=test_id.as_matrix(columns=None);

#adding bias column
mtx_dummy_tds=np.column_stack((mtx_dummy_tds,np.ones((len(mtx_dummy_tds),1))))
mtx_dummy_testds=np.column_stack((mtx_dummy_testds,np.ones((len(mtx_dummy_testds),1))))

#normalization
mtx_dummy_tds_norm=normalizePhi(mtx_dummy_tds)
mtx_dummy_testds_norm=normalizePhi(mtx_dummy_testds)


#print(mtx_dummy_tds)
#print(mtx_dummy_tds_norm)
#pandas.get_dummies(trained_data.ix[:,1:2])
#print("train",np.shape(mtx_dummy_tds_norm),"test",np.shape(mtx_dummy_testds_norm))


# In[25]:

td=dummy_trained_data;
X = td;
y = np.ravel(trained_y)

index = ["Logistic Regression","Gaussian Navie Baise","K Neighbors"]

columns = ["Mean error","Misclassified Points", "Accuracy"]
mc = pandas.DataFrame(index=index, columns=columns)

lgr = 0, LogisticRegression(penalty='l2',max_iter=100)
gnb = 1, GaussianNB()
knn = 2, KNeighborsClassifier()

classfier=lgr;
trained_system=classfier[1].fit(X, y);
y_pred = trained_system.predict(X)
misclassifedPoints = (y_pred != y).sum()
accuracy = (len(td.index) - misclassifedPoints) / len(td.index)
mean=np.mean(np.abs(y_pred - y));
mc.ix[classfier[0]] = [mean, misclassifedPoints, accuracy];
test_pre=trained_system.predict(dummy_test_data);
writeTestData(mtx_test_id,test_pre,filename="predictions_1"); #0.60958

classfier=gnb;
trained_system=classfier[1].fit(X, y);
y_pred = trained_system.predict(X)
misclassifedPoints = (y_pred != y).sum()
accuracy = (len(td.index) - misclassifedPoints) / len(td.index)
mean=np.mean(np.abs(y_pred - y));
mc.ix[classfier[0]] = [mean, misclassifedPoints, accuracy];
test_pre=trained_system.predict(dummy_test_data);
writeTestData(mtx_test_id,test_pre,filename="predictions_2"); #0.63380


classfier=knn;
trained_system=classfier[1].fit(X, y);
y_pred = trained_system.predict(X);
misclassifedPoints = (y_pred != y).sum()
accuracy = (len(td.index) - misclassifedPoints) / len(td.index)
mean=np.mean(np.abs(y_pred - y));
mc.ix[classfier[0]] = [mean, misclassifedPoints, accuracy];
test_pre=trained_system.predict(dummy_test_data);
writeTestData(mtx_test_id,test_pre,filename="predictions_3"); #0.0.62932


# In[ ]:



