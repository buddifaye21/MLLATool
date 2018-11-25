from sklearn import feature_selection as fs
from sklearn import preprocessing as prep
from sklearn import svm as svm
from sklearn import metrics as mets
from sklearn.model_selection import GridSearchCV as gs
import tensorflow as tf
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

import pickle


def main():
    choice = '3'
    print("+================================================================+")
    print("| Please select an option:                                       |")
    print("| 1. SVM                                                         |")
    print("| 2. Neural Network                                              |")
    print("| 3. Quit                                                        |")
    print("+================================================================+")
    choice = input("Please make a selection: ")
    while choice != '3':
        if choice == '1':
            svm_menu()
        elif choice == '2':
            neural_network()
        elif choice == '3':
            print("Quitting, thanks for using the tool! :)")
            break
        else:
            print("Invalid Option!")
        print("+================================================================+")
        print("| Please select an option:                                       |")
        print("| 1. SVM                                                         |")
        print("| 2. Neural Network                                              |")
        print("| 3. Quit                                                        |")
        print("+================================================================+")
        choice = input("Please make a selection: ")


def svm_menu():

    print("|----------------------------------------------|")
    print("| Please select an Option:                     |")
    print("|----------------------------------------------|")
    print("|1. Load Data and Train Model                  |")
    print("|2. Begin Testing                              |")
    print("|3. Previous Menu                              |")
    print("|----------------------------------------------|")
    choice = '3'
    choice = input("Please make a selection: ")
    while choice != '3':
        if choice == '1':
            train_model()
        elif choice == '2':
            begin_testing()
        elif choice == '3':
            print("Going back to previous menu...")
            break
        else:
            print("Invalid Choice!")
        print("|----------------------------------------------|")
        print("| Please select an Option:                     |")
        print("|----------------------------------------------|")
        print("|1. Load Data and Train Model                  |")
        print("|2. Begin Testing                              |")
        print("|3. Previous Menu                              |")
        print("|----------------------------------------------|")
        choice = input("Please make a selection: ")


def train_model():
    print("Loading Data...")
    pd.set_option('display.max_columns', 20)
    f = open("NSL_KDD-master\KDDTrain+.csv")
    cnfile = open("NSL_KDD-master\Field Names.csv")
    column_names = pd.read_csv(cnfile, header=None)
    column_names_list = column_names[0].tolist()
    col_list = list(range(0, 42))
    column_names_list.append("lables")
    # print(column_names_list)
    data = pd.read_csv(f, header=None, names=column_names_list, usecols=col_list)
    data = data[data.service != "harvest"]
    data = data[data.service != "urh_i"]
    data = data[data.service != "red_i"]
    data = data[data.service != "ftp_u"]
    data = data[data.service != "tftp_u"]
    data = data[data.service != "aol"]
    data = data[data.service != "http_8001"]
    data = data[data.service != "http_2784"]
    data = data[data.service != "pm_dump"]
    data = data[data.lables != "spy"]
    data = data[data.lables != "warezclient"]

    #print(data.service)
    #plt.plot(data.index, data.iloc['labels'].values)

    le = prep.LabelEncoder()
    # Categorical boolean mask
    categorical_feature_mask = data.dtypes == object
    # filter categorical columns using mask and turn it into a list
    categorical_cols = data.columns[categorical_feature_mask].tolist()

    #print(data[categorical_cols[0:3]].head(20))
    data[categorical_cols[0]] = le.fit_transform(data[categorical_cols[0]])
    protocols_map = dict(zip(le.classes_, le.transform(le.classes_)))
    data[categorical_cols[1]] = le.fit_transform(data[categorical_cols[1]])
    services_map = dict(zip(le.classes_, le.transform(le.classes_)))
    data[categorical_cols[2]] = le.fit_transform(data[categorical_cols[2]])
    flags_map = dict(zip(le.classes_, le.transform(le.classes_)))
    data[categorical_cols[3]] = le.fit_transform(data[categorical_cols[3]])
    labels_map = dict(zip(le.classes_, le.transform(le.classes_)))
    training_label_map = pickle.dump(labels_map, open("training_label_map.sav", 'wb'))
    #print(services_map)
    #print(data[categorical_cols].head(20))
    #print(labels_map) # Dictionary of training labels
    all_col_names = list(protocols_map.keys()) + list(services_map.keys()) + \
                    list(flags_map.keys()) + column_names_list[4:41]
    all_col_names.insert(0, column_names_list[0])
    all_col_names.append("labels")
    #print(all_col_names)
    categorical_feature_mask[41] = False
    protocols = list(protocols_map.keys())
    #print(len(protocols))
    services = list(services_map.keys())
    #print(len(services))
    flags = list(flags_map.keys())
    #print(len(flags))
    categories = []
    categories.append(protocols)
    categories.append(services)
    categories.append(flags)
    print("training categories: ", categories)

    #print(categorical_feature_mask)
    enc = prep.OneHotEncoder(categorical_features=categorical_feature_mask, sparse=False)
    #print(enc.categories)
    data_ohe = enc.fit_transform(data)
    print(data_ohe)
    #print(enc.get_feature_names(input_features=None))
    ohe_df = pd.DataFrame.from_records(data=data_ohe)
    #print(pd.get_dummies(ohe_df))
    print(ohe_df.head(100))
    ohe_df.columns = all_col_names

    array = ohe_df.values
    X = array[:, 0:115]
    Y = array[:, -1]


    #print(X)
    #print("X shape: ", X.shape)
    #print(Y)

    features = fs.SelectKBest(score_func=fs.f_classif, k=23).fit_transform(X, Y)
    selected_feature_mask = fs.SelectKBest(score_func=fs.chi2, k=23).fit(X, Y).get_support()
    #print("Selected feature mask: ", selected_feature_mask)
    new_cols = []
    count = 0
    for i in all_col_names:
        if selected_feature_mask[count]:
            new_cols.append(all_col_names[count])
        count = count + 1
        if count == 115:
            break

    #print("new_cols Length: ", len(new_cols))
    #print("Features Shape: ", features.shape)

    #train_new_feats = features[0:113375]
    #train_new_feats_Y = features[0:113375, -1]

    #test_new_feats = features[113376:125054]
    #test_new_feats_Y = features[113376:125054, -1]

    new_feats = pd.DataFrame(features, columns=new_cols)

    #print("New feats: ", new_feats)

    #gamma_range = [.00001, .000001, .0000001]
    #c_range = [1, 5, 10, 50]
    #tuned_parameters = dict(kernel=['rbf'], gamma=gamma_range, C=c_range, shrinking=[False])
    #grid = gs(svm.SVC(), tuned_parameters, cv=3, scoring='accuracy', n_jobs=2, verbose=10)
    #grid.fit(new_feats, Y)
    #print(grid.best_params_)
    #print(grid.best_score_)


    clf = svm.SVC(gamma=.00001, verbose=1, shrinking=False, C=61, kernel='rbf', max_iter=100000)
    clf.fit(new_feats, Y)
    train_pred = clf.predict(new_feats)
    print("Y: ", Y)
    print(train_pred)
    #print(clf.score(train_pred, Y))
    print(mets.accuracy_score(Y, train_pred))
    cm = mets.confusion_matrix(Y, train_pred)
    #print(cm)
    plt.imshow(cm, interpolation='nearest')
    plt.show()
    save_file = 'SVM_trained_model.sav'
    pickle.dump(clf, open(save_file, 'wb'))
    feat_save_file = 'feat_mask.sav'
    pickle.dump(selected_feature_mask, open(feat_save_file, 'wb'))


def begin_testing():
    print("Loading Data...")
    pd.set_option('display.max_columns', 20)
    f = open("NSL_KDD-master\KDDTest+.csv")
    cnfile = open("NSL_KDD-master\Field Names.csv")
    column_names = pd.read_csv(cnfile, header=None)
    column_names_list = column_names[0].tolist()
    col_list = list(range(0, 42))
    column_names_list.append("labels")
    #print(column_names_list)
    data = pd.read_csv(f, header=None, names=column_names_list, usecols=col_list)

    print("Data Loaded... Starting Preprocessing.")
    #print("Before Filter Shape: ", data.shape)
    data = data[data.labels != 'apache2']
    data = data[data.labels != 'httptunnel']
    data = data[data.labels != 'mailbomb']
    data = data[data.labels != 'mscan']
    data = data[data.labels != 'named']
    data = data[data.labels != 'processtable']
    data = data[data.labels != 'ps']
    data = data[data.labels != 'saint']
    data = data[data.labels != 'sendmail']
    data = data[data.labels != 'snmpgetattack']
    data = data[data.labels != 'snmpguess']
    data = data[data.labels != 'sqlattack']
    data = data[data.labels != 'udpstorm']
    data = data[data.labels != 'worm']
    data = data[data.labels != 'xlock']
    data = data[data.labels != 'xsnoop']
    data = data[data.labels != 'xterm']
    #print("After Filter Shape: ", data.shape)
    #print(data)

    le = prep.LabelEncoder()
    # Categorical boolean mask
    categorical_feature_mask = data.dtypes == object
    # filter categorical columns using mask and turn it into a list
    categorical_cols = data.columns[categorical_feature_mask].tolist()
    #print(categorical_cols)
    #print(data[categorical_cols[0:3]].head(20))
    data[categorical_cols[0]] = le.fit_transform(data[categorical_cols[0]])
    protocols_map = dict(zip(le.classes_, le.transform(le.classes_)))
    data[categorical_cols[1]] = le.fit_transform(data[categorical_cols[1]])
    services_map = dict(zip(le.classes_, le.transform(le.classes_)))
    data[categorical_cols[2]] = le.fit_transform(data[categorical_cols[2]])
    flags_map = dict(zip(le.classes_, le.transform(le.classes_)))
    data[categorical_cols[3]] = le.fit_transform(data[categorical_cols[3]])
    labels_map = dict(zip(le.classes_, le.transform(le.classes_)))
    #print(data[categorical_cols].head(20))
    #print(labels_map) # Dictionary of testing labels
    #print(services_map)

    #unknown_label_map = {}
    #for label in labels_map:
    #    if label not in training_label_map:
    #        unknown_label_map[label] = 1"""

    #print("Training labels map: ", training_label_map)
    #print("unknown label map: ", unknown_label_map)
    all_col_names = list(protocols_map.keys()) + list(services_map.keys()) + \
                    list(flags_map.keys()) + column_names_list[4:41]
    all_col_names.insert(0, column_names_list[0])
    all_col_names.append("labels")
    #print(all_col_names)
    categorical_feature_mask[41] = False
    #print(categorical_feature_mask)

    protocols = list(protocols_map.keys())
    #print(len(protocols))
    services = list(services_map.keys())
    #print(len(services))
    flags = list(flags_map.keys())
    categories = []
    categories.append(protocols)
    categories.append(services)
    categories.append(flags)
    print("test categories: ", categories)

    # print(categorical_feature_mask)
    #enc = prep.OneHotEncoder(categories=categories, sparse=False)
    #print(enc.categories)
    enc = prep.OneHotEncoder(categorical_features=categorical_feature_mask, sparse=False)
    data_ohe = enc.fit_transform(data)
    ohe_df = pd.DataFrame.from_records(data=data_ohe)
    #print(pd.get_dummies(ohe_df))
    print(ohe_df.head(100))
    ohe_df.columns = all_col_names

    array = ohe_df.values
    X = array[:, 0:115]
    Y = array[:, -1]
    #print(X)
    #print(X.shape)
    #print(Y)
    selected_feature_mask = pickle.load(open('feat_mask.sav', 'rb'))
    #print("Selected feature mask length: ", len(selected_feature_mask))
    test_data = pd.DataFrame.from_records(X[:, selected_feature_mask])
    #print("test data shape: ", test_data.shape)
    trained_clf = pickle.load(open('SVM_trained_model.sav', 'rb'))
    prediction = trained_clf.predict(test_data)
    score = trained_clf.score(test_data, Y)
    print(Y)
    print("prediction: ", prediction)
    print("Score: ", score)
    print("Mets score: ", mets.accuracy_score(Y, prediction))
    cm = mets.confusion_matrix(Y, prediction)
    print(cm)
    plt.imshow(cm, interpolation='nearest')
    plt.show()


def neural_network():
    print("Loading testing Data...")
    pd.set_option('display.max_columns', 6)
    f = open("NSL_KDD-master\KDDTest+.csv")
    cnfile = open("NSL_KDD-master\Field Names.csv")
    column_names = pd.read_csv(cnfile, header=None)
    column_names_list = column_names[0].tolist()
    col_list = list(range(0, 42))
    column_names_list.append("labels")
    #print(column_names_list)
    test_data = pd.read_csv(f, header=None, names=column_names_list, usecols=col_list)

    print("Data Loaded... Starting Preprocessing.")
    print("Before Filter Shape: ", test_data.shape)
    test_data = test_data[test_data.labels != 'apache2']
    test_data = test_data[test_data.labels != 'httptunnel']
    test_data = test_data[test_data.labels != 'mailbomb']
    test_data = test_data[test_data.labels != 'mscan']
    test_data = test_data[test_data.labels != 'named']
    test_data = test_data[test_data.labels != 'processtable']
    test_data = test_data[test_data.labels != 'ps']
    test_data = test_data[test_data.labels != 'saint']
    test_data = test_data[test_data.labels != 'sendmail']
    test_data = test_data[test_data.labels != 'snmpgetattack']
    test_data = test_data[test_data.labels != 'snmpguess']
    test_data = test_data[test_data.labels != 'sqlattack']
    test_data = test_data[test_data.labels != 'udpstorm']
    test_data = test_data[test_data.labels != 'worm']
    test_data = test_data[test_data.labels != 'xlock']
    test_data = test_data[test_data.labels != 'xsnoop']
    test_data = test_data[test_data.labels != 'xterm']
    print("After Filter Shape: ", test_data.shape)
    print(test_data)

    le = prep.LabelEncoder()
    # Categorical boolean mask
    categorical_feature_mask = test_data.dtypes == object
    # filter categorical columns using mask and turn it into a list
    categorical_cols = test_data.columns[categorical_feature_mask].tolist()
    #print(categorical_cols)
    #print(data[categorical_cols[0:3]].head(20))
    test_data[categorical_cols[0]] = le.fit_transform(test_data[categorical_cols[0]])
    protocols_map = dict(zip(le.classes_, le.transform(le.classes_)))
    test_data[categorical_cols[1]] = le.fit_transform(test_data[categorical_cols[1]])
    services_map = dict(zip(le.classes_, le.transform(le.classes_)))
    test_data[categorical_cols[2]] = le.fit_transform(test_data[categorical_cols[2]])
    flags_map = dict(zip(le.classes_, le.transform(le.classes_)))
    test_data[categorical_cols[3]] = le.fit_transform(test_data[categorical_cols[3]])
    labels_map = dict(zip(le.classes_, le.transform(le.classes_)))
    #print(data[categorical_cols].head(20))
    print(labels_map) # Dictionary of testing labels

    all_col_names = list(protocols_map.keys()) + list(services_map.keys()) + \
                    list(flags_map.keys()) + column_names_list[4:41]
    all_col_names.insert(0, column_names_list[0])
    all_col_names.append("labels")
    categorical_feature_mask[41] = False
    #print(categorical_feature_mask)
    enc = prep.OneHotEncoder(categorical_features=categorical_feature_mask, sparse=False)
    test_data_ohe = enc.fit_transform(test_data)
    test_ohe_df = pd.DataFrame.from_records(data=test_data_ohe)
    test_ohe_df.columns = all_col_names
    array = test_ohe_df.values
    test_X = array[:, 0:115]
    test_Y = array[:, -1]
    #print(test_X)
    print(test_X.shape)
    #print(test_Y)

    print("Loading Training Data...")
    pd.set_option('display.max_columns', 6)
    f = open("NSL_KDD-master\KDDTrain+.csv")
    cnfile = open("NSL_KDD-master\Field Names.csv")
    column_names = pd.read_csv(cnfile, header=None)
    column_names_list = column_names[0].tolist()
    col_list = list(range(0, 42))
    column_names_list.append("lables")
    # print(column_names_list)
    train_data = pd.read_csv(f, header=None, names=column_names_list, usecols=col_list)
    train_data = train_data[train_data.service != "harvest"]
    train_data = train_data[train_data.service != "urh_i"]
    train_data = train_data[train_data.service != "red_i"]
    train_data = train_data[train_data.service != "ftp_u"]
    train_data = train_data[train_data.service != "tftp_u"]
    train_data = train_data[train_data.service != "aol"]
    train_data = train_data[train_data.service != "http_8001"]
    train_data = train_data[train_data.service != "http_2784"]
    train_data = train_data[train_data.lables != "spy"]
    train_data = train_data[train_data.lables != "warezclient"]

    # print(data)
    # plt.plot(data.index, data.iloc['labels'].values)

    le = prep.LabelEncoder()
    # Categorical boolean mask
    categorical_feature_mask = train_data.dtypes == object
    # filter categorical columns using mask and turn it into a list
    categorical_cols = train_data.columns[categorical_feature_mask].tolist()

    # print(data[categorical_cols[0:3]].head(20))
    train_data[categorical_cols[0]] = le.fit_transform(train_data[categorical_cols[0]])
    protocols_map = dict(zip(le.classes_, le.transform(le.classes_)))
    train_data[categorical_cols[1]] = le.fit_transform(train_data[categorical_cols[1]])
    services_map = dict(zip(le.classes_, le.transform(le.classes_)))
    train_data[categorical_cols[2]] = le.fit_transform(train_data[categorical_cols[2]])
    flags_map = dict(zip(le.classes_, le.transform(le.classes_)))
    train_data[categorical_cols[3]] = le.fit_transform(train_data[categorical_cols[3]])
    labels_map = dict(zip(le.classes_, le.transform(le.classes_)))
    training_label_map = pickle.dump(labels_map, open("training_label_map.sav", 'wb'))
    all_col_names = list(protocols_map.keys()) + list(services_map.keys()) + \
                    list(flags_map.keys()) + column_names_list[4:41]
    all_col_names.insert(0, column_names_list[0])
    all_col_names.append("labels")
    categorical_feature_mask[41] = False
    # print(categorical_feature_mask)
    enc = prep.OneHotEncoder(categorical_features=categorical_feature_mask, sparse=False)
    train_data_ohe = enc.fit_transform(train_data)
    train_ohe_df = pd.DataFrame.from_records(data=train_data_ohe)
    train_ohe_df.columns = all_col_names
    array = train_ohe_df.values
    train_X = array[:, 0:115]
    train_Y = array[:, 115]
    # print(train_X)
    print("train_X shape: ", train_X.shape)
    # print(train_Y)

    model = keras.Sequential([
        keras.layers.Dense(115, activation=tf.nn.relu),
        keras.layers.Dense(21, activation=tf.nn.softmax)
    ])
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_X, train_Y, epochs=15)
    test_loss, test_acc = model.evaluate(test_X, test_Y)
    print('Test accuracy:', test_acc)


main()


