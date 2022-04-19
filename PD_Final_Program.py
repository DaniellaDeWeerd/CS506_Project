import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

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

def random_forest (the_data):
    '''Takes in data, runs random forest and tests it to find and return accuracy.'''
    #split data:
    X_train, X_test, y_train, y_test = split (the_data)

    rfc = RandomForestClassifier()
    forest_params = [{'n_estimators': [100,200,500],'max_depth': list(range(4, 15, 2)), 'max_features': ['auto', 'sqrt', 'log2'], 'criterion': ['gini', 'entropy']}]
    clf = GridSearchCV(rfc, forest_params, cv = 3, scoring='accuracy')
    clf.fit(X_train, y_train)
    print(clf.best_params_)
    print(clf.best_score_)
    #Run algorithm:
    y_pred_rf=clf.predict(X_test)

    #Test Algorithm:
    accuracy = metrics.accuracy_score(y_test, y_pred_rf)
    return accuracy

def K_Nearest (the_data):
    '''Takes in data, runs logistic regression and tests it to find and return accuracy.'''
    #split data:
    X_train, X_test, y_train, y_test = split (the_data)

    estimator_KNN = KNeighborsClassifier(algorithm='auto')
    parameters_KNN = {
        'n_neighbors': (1,10, 1),
        'leaf_size': (20,40,1),
        'p': (1,2),
        'weights': ('uniform', 'distance'),
        'metric': ('minkowski', 'chebyshev')}
                    
    # with GridSearch
    grid_search_KNN = GridSearchCV(
        estimator=estimator_KNN,
        param_grid=parameters_KNN,
        scoring = 'accuracy',
        n_jobs = -1,
        cv = 5
    )
    
    neigh = grid_search_KNN.fit(X_train, y_train)
    y_pred_knn=neigh.predict(X_test)

    print(grid_search_KNN.best_params_ ) 
    print('Best Score - KNN:', grid_search_KNN.best_score_ )
    #Test Algorithm:
    accuracy = metrics.accuracy_score(y_test, y_pred_knn)
    return accuracy

def random_forest_Final (the_data):
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


def K_Nearest_Final (the_data):
    '''Takes in data, runs logistic regression and tests it to find and return accuracy.'''
    #split data:
    X_train, X_test, y_train, y_test = split (the_data)

    #Run algorithm:
    neigh = KNeighborsClassifier(leaf_size = 20, metric = 'minkowski', n_neighbors = 1, p =  2, weights = 'uniform')
    neigh.fit(X_train, y_train)
    y_pred_knn=neigh.predict(X_test)

    #Test Algorithm:
    accuracy = metrics.accuracy_score(y_test, y_pred_knn)
    return accuracy

def run_algos (the_data):
    run_num = 10
    rf_accs = []
    for num in range(0,run_num):
        acc = random_forest_Final(the_data)
        rf_accs.append(acc)
    rf_average = sum(rf_accs)/len(rf_accs)
    knn_accs = []
    for num in range(0,run_num):
        acc = K_Nearest_Final(the_data)
        knn_accs.append(acc)
    knn_average = sum(knn_accs)/len(knn_accs)

    print("The average accuracy for rf: ", rf_average)
    print("The average accuracy for knn: ", knn_average)

#########################################################
#Run program:
#########################################################
names,the_data = load_data("parkinsons.data")
run_algos(the_data)
