#Suleiman A.T Suleiman
#2589984
#CNG 514 Data Mining HW2


#Imported Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


#Function for checking for and removing empty fields in dataset      
def checkEmptyCells(data):

    #first we replace empty cells with np.nan objects
    data = data.replace('', np.nan)

    #next we check  of there are any np.nan objects
    numOfEmpty = np.where(pd.isnull(data))
    if(len(numOfEmpty[0]) == 0):
        print("The dataset has no empty cells\n")
    #We can expand this function handle the empty cells differently depending
    #The number of empty cells(Either remove the row or normalize data)
    else:
        print("This column contains " +str(len(numOfEmpty))+" empty cells\n")
        print("...Removing Rows with Empty Cells...")
        data = data.dropna()
    return data  
#-----------------------------------------------------------------------------#
#Removing Outliers
def removeOutliers(data):
    #remove upper outliers first
    q_high  = data.quantile(0.99)
    q_low = data.quantile(0.01)

    q = data[( data< q_high ) & ( data > q_low)]
    return q

#-----------------------------------------------------------------------------#

#This function checks for outliers in the dataset
def checkOutliers(data):
    print(data.describe())
    print("To visually check for outliers we will use boxplots\n")
    
    plt.figure(1)
    data['Age'].plot.box(title="Boxplot of Age", xticks=[])

    plt.figure(2)
    data['Gender'].plot.box(title="Boxplot of Gender", xticks=[])

    plt.figure(3)
    data['Air Pollution'].plot.box(title="Boxplot of Air Pollution", xticks=[])

    plt.figure(4)
    data['Alcohol use'].plot.box(title="Boxplot of Alcohol use", xticks=[])

    plt.figure(5)
    data['chronic Lung Disease'].plot.box(title="Boxplot of chronic Lung Disease", xticks=[])

    plt.figure(6)
    data['Obesity'].plot.box(title="Boxplot of Obesity", xticks=[])

    plt.figure(7)
    data['Smoking'].plot.box(title="Boxplot of Smoking", xticks=[])

    plt.figure(8)
    data['Chest Pain'].plot.box(title="Boxplot of Chest Pain", xticks=[])

    plt.figure(9)
    data['Coughing of Blood'].plot.box(title="Boxplot of Coughing", xticks=[])

    plt.figure(10)
    data['Swallowing Difficulty'].plot.box(title="Boxplot of Swallowing Difficulty", xticks=[])

    plt.figure(11)
    data['Frequent Cold'].plot.box(title="Boxplot of Frequent Cold", xticks=[])

    plt.figure(12)
    data['Snoring'].plot.box(title="Boxplot of Snoring", xticks=[])

    plt.show()

#Function for training and testing
def trainingAndTesting(X,Y):
    print("Spliting Dataset into training and Testing\n")
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.33, random_state=1, stratify=Y)

    #Choosing best K
    # We choose the best K for this classification based on the error rate of the
    # respective models perfomance with different K values.
    error_rate=[]#list that will store the average error rate value of k
    for i in range (1,10):  #Took the range of k from 1 to 10
        clf=KNeighborsClassifier(n_neighbors=i)
        clf.fit(X_train,Y_train)
        predict_i=clf.predict(X_test)
        error_rate.append(np.mean(predict_i!=Y_test))
    error_rate
    #plotting the error rate vs k graph
    plt.figure(figsize=(12,6))
    plt.plot(range(1,10),error_rate,marker="o",markerfacecolor="green",
         linestyle="dashed",color="red",markersize=15)
    plt.title("Error rate vs k value",fontsize=20)
    plt.xlabel("k- values",fontsize=20)
    plt.ylabel("error rate",fontsize=20)
    plt.xticks(range(1,10))
    plt.show()
    #Building Model

    # K from 1-> 8 have the same error rate I just chose 3 as I am most use to this number
    #create KNN Classifier
    knn = KNeighborsClassifier(n_neighbors = 3)

    print("...Training with Training Data...\n")
    #Fit the classifier to the dataset
    knn.fit(X_train,Y_train)

    #Testing the Model
    print("...Testing out Model...\n")
    predictions = knn.predict(X_test)

    #Checking predictions
    #print(predictions)

    #Checking Accuracy
    print("...Checking accuracy of predictions...\n")
    accuracy = knn.score(X_test, Y_test)
    print("The accuracy of our model is " +str(accuracy) + "\n")

    #Plotting Confusion Matrix
    print("...Calculating Confusion Matrix...\n")

    #Generate the confusion matrix
    cf_matrix = confusion_matrix(Y_test,predictions)
    print(cf_matrix)

    print("...Plotting Confusion Matrix...\n")
    ConfusionMatrixDisplay.from_predictions(Y_test,predictions)
    plt.show()    


#Main function

def main():
    print("----------------------------------\n")
    print("Lung Cancer Classification Program\n")
    print("...Reading CSV file...\n")

    data = pd.read_csv("cng514-cancer-patient-data-assignment-2.csv")
    print("...Loading CSV File...\n")
    #print(data)

    #Preprocessing
    print("...Checking For Empty cells in dataset...\n")
    checkEmptyCells(data)
    #print(data)

    print("...Looking for outliers...\n")
    checkOutliers(data)

    #Training and Testing
    print("Spliting up dataset to inputs and targets\n")
    print("We will create a dataset without the patient id, gender and level columns\n"
          "Because the level column is our labels and will there for be used for evaluation\n"
          "And the patient id and gender columns aren't relevant to the classification")
    X = data.drop(columns=['Patient Id','Gender','Level'])
    
    #Debugging
    print(X.head())

    #This will be our target values
    Z = data['Level'].values

    '''We need to convert our Level Labels from High, Medium and Low to binary values
    I chose to convert both High and Medium to 1 and Low to 0
     I chose this because it is better toclassify someone without cancer with it, than to classify
    someone with cancer as not having it.'''
    #Y = pd.Series(np.where( Z == 'YES', 1, 0),Z)
    Y = pd.Series(np.where((Z== 'High') | (Z== 'Medium') ,1,0),Z)
    Y.index = list(range(0,len(Y)))
    #Debugging
    print(Y)

    #Conduct Training and Testing and Calculate Accuracy and Confusion Matrix
    trainingAndTesting(X,Y)
    

#Calling main
if __name__=="__main__":
    main()
