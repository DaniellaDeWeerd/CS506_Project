import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

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
    '''Takes in the data and split the data into training and testing sets.'''
    #Split into training and testing:
    diagnosis = the_data[['status']].copy()
    the_data = the_data.drop('status',1)
    X_train, X_test, y_train, y_test = train_test_split(the_data, diagnosis, test_size=0.3)
    

    return X_train, X_test, y_train, y_test

def random_forest(the_data):
    '''Takes in data, runs random forest and tests it to find and return accuracy.'''
    #split data:
    X_train, X_test, y_train, y_test = split (the_data)

    #Run algorithm:
    rfc=RandomForestClassifier(criterion = 'entropy', max_depth = 8, max_features = 'sqrt', n_estimators = 100)
    rfc.fit(X_train,y_train)
    y_pred_rf=rfc.predict(X_test)

    #Test Algorithm:
    accuracy = metrics.accuracy_score(y_test, y_pred_rf)
    return accuracy

def run_algos (the_data):
    run_num = 10
    rf_accs = []
    for num in range(0,run_num):
        acc = random_forest(the_data)
        rf_accs.append(acc)
    rf_average = sum(rf_accs)/len(rf_accs)

    print("The average accuracy for rf: ", rf_average)

#########################################################
#Run program:
#########################################################
names,the_data = load_data("parkinsons.data")
run_algos(the_data)
