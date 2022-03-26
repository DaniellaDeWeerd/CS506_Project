import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt

def load_data(filename):
    '''Takes file name and outputs the data in a format that is easy to use for ML.'''
    f = open(filename, 'r')
    data = f.read().splitlines()
    new_data = []
    for line in data:
        line = line.split(',')
        new_data.append(line)
    f.close() # close f
    column_names = "name,MDVP:Fo(Hz),MDVP:Fhi(Hz),MDVP:Flo(Hz),MDVP:Jitter(%),MDVP:Jitter(Abs),MDVP:RAP,MDVP:PPQ,Jitter:DDP,MDVP:Shimmer,MDVP:Shimmer(dB),Shimmer:APQ3,Shimmer:APQ5,MDVP:APQ,Shimmer:DDA,NHR,HNR,status,RPDE,DFA,spread1,spread2,D2,PPE"
    column_names = column_names.split(",")
    df = pd.DataFrame(new_data, columns = column_names)
    names = df[['name']].copy()
    data = df.drop('name',1)
    data = data.apply(pd.to_numeric) #since loaded in as an object (str)

    #normalize the data
    for column in data.columns:
        data[column] = (data[column] - data[column].min()) / (data[column].max() - data[column].min())
    
    return names,data

def split (the_data):
    '''Takes in the datat and split the data into training and testing sets.'''
    #Split into training and testing:
    diagnosis = the_data[['status']].copy()
    the_data = the_data.drop('status',1)
    X_train, X_test, y_train, y_test = train_test_split(the_data, diagnosis, test_size=0.3)

    return X_train, X_test, y_train, y_test

def random_forest (the_data):
    '''Takes in data, runs random forest and tests it to find and return accuracy.'''
    #split data:
    X_train, X_test, y_train, y_test = split (the_data)

    #Run algorithm:
    rfc=RandomForestClassifier(n_estimators=100)
    rfc.fit(X_train,y_train)
    y_pred_rf=rfc.predict(X_test)

    #Test Algorithm:
    accuracy = metrics.accuracy_score(y_test, y_pred_rf)
    return accuracy

def logistic_regression (the_data):
    '''Takes in data, runs logistic regression and tests it to find and return accuracy.'''
    #split data:
    X_train, X_test, y_train, y_test = split (the_data)

    #Run algorithm:
    lrc = LogisticRegression()
    lrc.fit(X_train, y_train)
    y_pred_lr=lrc.predict(X_test)

    #Test Algorithm:
    accuracy = metrics.accuracy_score(y_test, y_pred_lr)
    return accuracy

def run_algos (the_data):
    run_num = 5
    rf_accs = []
    for num in range(0,run_num):
        acc = random_forest(the_data)
        rf_accs.append(acc)
    rf_average = sum(rf_accs)/len(rf_accs)

    lr_accs = []
    for num in range(0,run_num):
        acc = logistic_regression(the_data)
        lr_accs.append(acc)
    lr_average = sum(lr_accs)/len(lr_accs)

    print("The average accuracy for rf: ", rf_average)
    print("The average accuracy for lr: ", lr_average)

def display_statistics (filename,data):
    f = open(filename, 'r')
    meta_data = f.read().splitlines()
    new_meta_data = []
    for line in meta_data:
        line = line.split(',')
        new_meta_data.append(line)
    f.close() # close f
    column_names = "subject#,age,sex,test_time,motor_UPDRS,total_UPDRS,Jitter(%),Jitter(Abs),Jitter:RAP,Jitter:PPQ5,Jitter:DDP,Shimmer,Shimmer(dB),Shimmer:APQ3,Shimmer:APQ5,Shimmer:APQ11,Shimmer:DDA,NHR,HNR,RPDE,DFA,PPE"
    column_names = column_names.split(",")
    meta_data = pd.DataFrame(new_meta_data, columns = column_names)
    meta_data = meta_data.apply(pd.to_numeric) #since loaded in as an object (str)

    age_PD = meta_data[data['status'] == 1]
    age_PD = age_PD['age']
    age_HC = meta_data[data['status'] == 0]
    age_HC = age_HC['age']
    plt.hist(age_PD)
    plt.title('Age with PD')
    plt.show() 
    plt.clf()
    plt.hist(age_HC)
    plt.title('Age without PD')
    plt.show() 
#########################################################
#Run program:
#########################################################
names,the_data = load_data("parkinsons.data")
display_statistics ('parkinsons_updrs.data',the_data)
run_algos(the_data)










