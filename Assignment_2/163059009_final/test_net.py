
# coding: utf-8

# In[7]:

import pandas;
import numpy as np;
import os.path;
import os;
np.set_printoptions(suppress=True)


# In[ ]:




# In[8]:

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
        fo.write(str(test_id[i][0])+","+str(ystar[i][0])+"\n");
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




# In[9]:

class NN:
    logging_enabled=True;
    trained_ds=None;
    trained_output=None;
    no_of_hiddenlayer=0;
    neurons_per_hiddenlayer=0;    
    weights=list();#list of matrix of weights
    layer_output=list();#vector of layers
    layer_delta=list(); 
    layer_error=list(); 
    no_of_iteration=1;
    mean_print_rate=10000;
    percent_print_rate=1000;
    learning_rate=1;
    approching_to_zero=1e-15;
    approching_to_one=1-1e-15;
    lamda=0.001;
    log_dir="log";
    log_folder=None;
    enable_bias_per_hidden=False;
    def __init__(self):
        np.random.seed(1);
        pass;
    
    def reInit(self):        
        self.layer_output=[0]*(self.no_of_hiddenlayer+2);
        self.layer_delta=[0]*(self.no_of_hiddenlayer+2);
        self.layer_error=[0]*(self.no_of_hiddenlayer+2);      
        self.weights=list();
        pass;
        
    def createNN(self):
        self.no_of_features=len(self.trained_ds[0]);
        self.no_of_datapoint=len(self.trained_ds); 
        if(self.enable_bias_per_hidden and self.neurons_per_hiddenlayer>1):
            self.neurons_per_hiddenlayer+=1;#adding bias neuron whose o/p will always be 1;
            
        self.reInit();
        self.initWeightMatrix();  
        self.log("Neural Network Created...","-");        
        pass;
    
    def initWeightMatrix(self):        
        for i in range(self.no_of_hiddenlayer):
            if(i==0):
                m=self.no_of_features;
            else:
                m=self.neurons_per_hiddenlayer;
            w_matrix = 2*np.random.random((m,self.neurons_per_hiddenlayer)) - 1;            
            
            if(self.enable_bias_per_hidden):
                #making weights of bias neuron as zero. So that its o/p is always one.
                w_matrix[:,0]=0;
                
            self.weights.append(w_matrix);
            #print(w_matrix);
            
        #last layer weight: For single output
        if(self.no_of_hiddenlayer==0):
            n=self.no_of_features;
        else:
            n=self.neurons_per_hiddenlayer;
        w_vector=2*np.random.random((n,1)) - 1;
        #print(w_vector);
        self.weights.append(w_vector);
        self.weights=np.array(self.weights);
        pass;
        
    def getTimestamp(self):
        import os.path;
        import datetime;
        import time;
        ts = datetime.datetime.fromtimestamp(time.time()).strftime('%d-%m-%Y-%H:%M:%S')
        return ts;

    def createDir(self,directory):
        import os.path;
        if not os.path.exists(directory):
            os.makedirs(directory);
        pass;
    
    def writeWeights(self,filenumber=0):
        fname=self.log_folder+"/weights-"+str(filenumber)+"-";
        m=len(self.weights);
        for i in range(m):
            np.save(fname+""+str(i), nn.weights[i])        
        pass;
    
    def writeWeights(self,filenumber=0):
        fname=self.log_folder+"/weights-"+str(filenumber)+"-";
        m=len(self.weights);
        for i in range(m):
            np.save(fname+""+str(i), self.weights[i])        
        pass;
    import os;


    def writeFinalWeights(self):
        filename="weights";        
        np.save(filename, self.weights);
        os.rename(filename+".npy", filename+".txt");
        pass;

    
    def loadWeights(self,folder=None,filename_offset="weights-0-",filename=None):        
        w=list();
        i=0;
        if(filename==None):
            file_path=folder+"/"+filename_offset+str(i)+".npy";        
            while (os.path.isfile(file_path)):
                w.append(np.load(file_path));
                i+=1;
                file_path=folder+"/"+filename_offset+str(i)+".npy";
            self.weights=w;
        else:
            self.weights=np.load(filename);
        return w;
    pass

    def activationFunction(self,x,deriv=False):
        return self.actFunSigmoid(x,deriv);
    
    def actFunSigmoid(self,x,deriv=False):
        if(deriv==True):
            return x*(1-x)
        return 1/(1+np.exp(-x))
    
    def actFunReLu(self,x,deriv=False):
        p = x > 0;            
        if(deriv==True):
            return p.astype(int);
        return x * p;
    
    def forwardPropogation(self,datapoints):        
        #intial layer i.e l0 zeroth layer
        self.layer_output[0]=datapoints;
        for i in range(1,self.no_of_hiddenlayer+2):
            prev_layer=self.layer_output[i-1];
            w=self.weights[i-1]
            sum_l=prev_layer.dot(w);           
            l_i=self.activationFunction(sum_l);
            self.layer_output[i]=l_i;   
            if(self.enable_bias_per_hidden and i<=self.no_of_hiddenlayer):
                #making weights of bias neuron as zero. So that its o/p is always one.
                l_i[:,0]=1;
            #print("l"+str(i)+":",l_i);
        return self.layer_output[self.no_of_hiddenlayer+1];
        pass;
    
    def findError(self):
        self.backPropogation(sel.layer_output);
        return self.getMeanError();
        
    def backPropogation(self,l_output):
        last_layer=len(l_output)-1;            
        
        #(target-output)        
        li=l_output[last_layer];
        error_diff=self.trained_output-li;
        delta=error_diff*self.activationFunction(li,deriv=True)    
        self.layer_delta[last_layer]=delta;
        self.layer_error[last_layer]=error_diff;
        
        for i in range(last_layer-1,0,-1):            
            #i-1 th layer calculation of delta
            error_diff=self.layer_delta[i+1].dot(self.weights[i].T);
            #print("l"+str(i)+"_error:",delta);
            delta=error_diff*self.activationFunction(l_output[i],deriv=True)
            self.layer_delta[i]=delta;
            self.layer_error[i]=error_diff;  
            
        #print("l"+str(i)+"_delta:",delta);        
        pass;
    
    def updateWeights(self,l_delta):
        #print(self.learning_rate);
        for i in range(self.no_of_hiddenlayer,-1,-1): # loop upto 0
            self.weights[i]+=self.learning_rate*self.layer_output[i].T.dot(l_delta[i+1]);
            if(self.lamda!=0):
                self.weights[i]-=self.learning_rate*-((self.lamda)*self.weights[i]);
            #print("w"+str(i)+":",self.layer_output[i].T.dot(l_delta[i+1]));
    
    
    def getMeanError(self):   
        return np.mean(np.abs(self.layer_error[self.no_of_hiddenlayer+1]));
    
    def getTrainedStatus(self,predicted_y):
        index = ["Neural-Network"];
        columns = ["Mean","Misclassified", "Accuracy"];
        status = pandas.DataFrame(index=index, columns=columns);       
        t_y=self.getThresholdValue(predicted_y);
        error_diff=self.trained_output-predicted_y;
        mean=np.mean(np.abs(error_diff));
        misclassifed= (self.trained_output != t_y).sum();
        accuracy = (len(t_y) - misclassifed) / len(t_y);
        status.ix["Neural-Network"]=[mean,misclassifed,accuracy];
        return status;
    
    def gradientDescent(self):
        for i in range(self.no_of_iteration):
            result=self.forwardPropogation(self.trained_ds);
            self.backPropogation(self.layer_output);            
            
            if ((i%self.mean_print_rate==0 or i >=self.no_of_iteration-6)):
                self.log(str(i)+"Error:",self.getMeanError());
                self.log(self.getTrainedStatus(result));
                
            if (i%self.percent_print_rate==0 or i >=self.no_of_iteration-6):
                self.log(str(i)+"Percent completed",(i*100)/self.no_of_iteration)
                #self.writeWeights();        
            if(i!=self.no_of_iteration-1):#donot update for last iteration            
                self.updateWeights(self.layer_delta);
        print("Error:",self.getMeanError());
        self.writeWeights();
        pass;
    
    def getThresholdValue(self,result,threshold=0.5):
        return(result>threshold).astype(int);

    def train(self):
         self.gradientDescent();
            
    def predict(self,datapoints):#forward propogation
        loutput=[0]*(self.no_of_hiddenlayer+2);
        #intial layer i.e l0 zeroth layer
        loutput[0]=datapoints;
        for i in range(1,self.no_of_hiddenlayer+2):
            prev_layer=loutput[i-1];
            w=self.weights[i-1]
            sum_l=prev_layer.dot(w);           
            l_i=self.activationFunction(sum_l);
            loutput[i]=l_i;   
            if(self.enable_bias_per_hidden and i<=self.no_of_hiddenlayer):
                #making weights of bias neuron as zero. So that its o/p is always one.
                l_i[:,0]=1;
        return loutput[self.no_of_hiddenlayer+1];
        pass;
    
    def log(self,text,data=None):
        if self.logging_enabled:
            if(data!=None):
                print(text,data);
            else:
                print(text);
        pass;
pass;


# In[ ]:




# In[10]:

#--settings--
pandas.set_option('display.max_columns', None);
#---init---
dir=""
trainFile=dir+"train.csv";
testFile=dir+"kaggle_test_data.csv";
categoryList=["workclass","education","marital-status","occupation","relationship","race","sex","native-country"];
drop_col=["native-country","race","education"]
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

#print("train",np.shape(mtx_dummy_tds_norm),"test",np.shape(mtx_dummy_testds_norm));


# In[ ]:




# In[13]:

#neural network
#dataset: census income
nn= NN();
x=mtx_dummy_tds_norm;
y=mtx_trained_y;
nn.no_of_hiddenlayer=2;
nn.neurons_per_hiddenlayer=20;
nn.no_of_iteration=101;
nn.learning_rate=0.0001;
nn.lamda=0;
nn.mean_print_rate=20;
nn.percent_print_rate=20;
nn.enable_bias_per_hidden=True;

#Common setting
nn.logging_enabled=False;
nn.trained_ds=x;
nn.trained_output=y;
nn.createNN();
nn.loadWeights(filename="weights.txt");
result=nn.predict(mtx_dummy_testds_norm);
writeTestData(mtx_test_id,nn.getThresholdValue(result),filename="predictions");


# In[ ]:



