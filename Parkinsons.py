import pandas as pd
from sklearn import neural_network
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
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
    
    # plt.hist([78,79,67,70,73,53,48,64,68,50,60,60,76,81,70,73,85,72,61,62,72,74,63])
    # plt.title('Age with PD')
    # plt.show() 
    # plt.clf()
    # plt.hist([46,48,61,62,64,66,66,69])
    # plt.title('Age without PD')
    # plt.show()
    plt.hist([0,1,0,0,0,1,1,0,0,0,0,0,0,1,0,0,1,1,0,0,0,0,1])
    plt.title('sex with PD')
    plt.savefig('Sex_PD.PNG')
    # plt.clf()
    # plt.hist([1,1,0,0,1,1,1,0])
    # plt.title('sex without PD')
    # plt.show()

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

def SVM (the_data):
    '''Takes in data, runs logistic regression and tests it to find and return accuracy.'''
    #split data:
    X_train, X_test, y_train, y_test = split (the_data)

    #Run algorithm:
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    y_pred_SVM=clf.predict(X_test)

    #Test Algorithm:
    accuracy = metrics.accuracy_score(y_test, y_pred_SVM)
    return accuracy
    
def Naive_Bayes (the_data):
    '''Takes in data, runs logistic regression and tests it to find and return accuracy.'''
    #split data:
    X_train, X_test, y_train, y_test = split (the_data)

    #Run algorithm:
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred_GNB=gnb.predict(X_test)

    #Test Algorithm:
    accuracy = metrics.accuracy_score(y_test, y_pred_GNB)
    return accuracy

def Neural_network (the_data):
    '''Takes in data, runs logistic regression and tests it to find and return accuracy.'''
    #split data:
    X_train, X_test, y_train, y_test = split (the_data)

    #Run algorithm:
    nnmlp = MLPClassifier()
    nnmlp.fit(X_train, y_train)
    y_pred_nn=nnmlp.predict(X_test)

    #Test Algorithm:
    accuracy = metrics.accuracy_score(y_test, y_pred_nn)
    return accuracy

def K_Nearest (the_data):
    '''Takes in data, runs logistic regression and tests it to find and return accuracy.'''
    #split data:
    X_train, X_test, y_train, y_test = split (the_data)

    #Run algorithm:
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train, y_train)
    y_pred_knn=neigh.predict(X_test)

    #Test Algorithm:
    accuracy = metrics.accuracy_score(y_test, y_pred_knn)
    return accuracy

def PCA_random_forest (the_data):
    '''Takes in data, runs logistic regression and tests it to find and return accuracy.'''
    #split data:
    X_train, X_test, y_train, y_test = split (the_data)

    #PCA
    pca = PCA(n_components=6)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    #Run algorithm:
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train, y_train)
    y_pred_pca_knn=neigh.predict(X_test)

    #Test Algorithm:
    accuracy = metrics.accuracy_score(y_test, y_pred_pca_knn)
    return accuracy

def ADABoost (the_data):
    '''Takes in data, runs logistic regression and tests it to find and return accuracy.'''
    #split data:
    X_train, X_test, y_train, y_test = split (the_data)

    #Run algorithm:
    ada = AdaBoostClassifier()
    ada.fit(X_train, y_train)
    y_pred_ada=ada.predict(X_test)

    #Test Algorithm:
    accuracy = metrics.accuracy_score(y_test, y_pred_ada)
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

    svm_accs = []
    for num in range(0,run_num):
        acc = SVM(the_data)
        svm_accs.append(acc)
    svm_average = sum(svm_accs)/len(svm_accs)

    nb_accs = []
    for num in range(0,run_num):
        acc = Naive_Bayes(the_data)
        nb_accs.append(acc)
    nb_average = sum(nb_accs)/len(nb_accs)

    nn_accs = []
    for num in range(0,run_num):
        acc = Neural_network(the_data)
        nn_accs.append(acc)
    nn_average = sum(nn_accs)/len(nn_accs)

    knn_accs = []
    for num in range(0,run_num):
        acc = K_Nearest(the_data)
        knn_accs.append(acc)
    knn_average = sum(knn_accs)/len(knn_accs)

    pca_accs = []
    for num in range(0,run_num):
        acc = PCA_random_forest(the_data)
        pca_accs.append(acc)
    pca_average = sum(pca_accs)/len(pca_accs)

    ada_accs = []
    for num in range(0,run_num):
        acc = ADABoost(the_data)
        ada_accs.append(acc)
    ada_average = sum(ada_accs)/len(ada_accs)

    print("The average accuracy for rf: ", rf_average)
    print("The average accuracy for lr: ", lr_average)
    print("The average accuracy for svm: ", svm_average)
    print("The average accuracy for nb: ", nb_average)
    print("The average accuracy for nn: ", nn_average)
    print("The average accuracy for knn: ", knn_average)
    print("The average accuracy for pca: ", pca_average)
    print("The average accuracy for ada: ", ada_average)


#########################################################
#Run program:
#########################################################
names,the_data = load_data("parkinsons.data")
run_algos(the_data)










