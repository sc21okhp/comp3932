import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Preprocess
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
#KNN
from sklearn.neighbors import KNeighborsClassifier
#LogReg
from sklearn.linear_model import LogisticRegression
#Evaluation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

global labelDeathDuration
labelDeathDuration = 360

def loadData():
    #SUBJECT_ID, GENDER, DOB, DOD, EXPIRE_FLAG
    patients = pd.read_csv('MIMIC-3/dataset/PATIENTS/PATIENTS_sorted.csv')
    patients = patients[['SUBJECT_ID', 'GENDER', 'DOB', 'DOD', 'EXPIRE_FLAG']].copy()
    #SUBJECT_ID, ADMITTIME, ADMISSION_TYPE, INSURANCE, LANGUAGE, RELIGION, ETHNICITY 
    admissions = pd.read_csv('MIMIC-3/dataset/ADMISSIONS/ADMISSIONS_sorted.csv')
    admissions = admissions[['SUBJECT_ID', 'ADMITTIME', 'ADMISSION_TYPE', 'ADMISSION_LOCATION', 'INSURANCE', 'LANGUAGE', 'RELIGION', 'MARITAL_STATUS', 'ETHNICITY']].copy()
    #SUBJECT_ID, HADM_ID, SEQ_NUM, ICD9_CODE
    diagnoses = pd.read_csv('MIMIC-3/dataset/DIAGNOSES_ICD/DIAGNOSES_ICD_sorted.csv')
    diagnoses = diagnoses[['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM', 'ICD9_CODE']].copy()

    return patients, admissions, diagnoses

def labelDeath(row):
    global labelDeathDuration
    if pd.isnull(row['DOD']) or pd.isnull(row['ADMITTIME']):
        return 0

    duration = (row['DOD'] - row['ADMITTIME']).days
    return 1 if duration >= 0 and duration <= labelDeathDuration else 0

def mergeLabel(patients, admissions, diagnoses):
    patients['DOD'] = pd.to_datetime(patients['DOD'], errors='coerce')
    admissions['ADMITTIME'] = pd.to_datetime(admissions['ADMITTIME'], errors='coerce')

    admissions_patients = admissions.merge(patients, on='SUBJECT_ID', how='left')
    admissions_patients_diagnoses = admissions_patients.merge(diagnoses, on='SUBJECT_ID', how='left')

    admissions_patients_diagnoses['DEAD_AFTER_DURATION'] = admissions_patients_diagnoses.apply(labelDeath, axis=1)

    return admissions_patients_diagnoses

def preprocess(df):
    numerical = ['HADM_ID', 'SEQ_NUM']
    catagorical = ['GENDER', 'ADMISSION_TYPE', 'ADMISSION_LOCATION', 'INSURANCE', 'LANGUAGE', 'RELIGION', 'MARITAL_STATUS', 'ETHNICITY', 'ICD9_CODE']

    for col in catagorical:
        df[col] = df[col].fillna('UNKNOWN')
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    #?REMOVE THIS FOR FULL DATA
    df = df.iloc[:10000]

    return df

def featuresTarget(df):
    features = ['GENDER', 'ADMISSION_TYPE', 'ADMISSION_LOCATION', 'INSURANCE', 'LANGUAGE', 'RELIGION', 'MARITAL_STATUS', 'ETHNICITY', 'HADM_ID', 'SEQ_NUM', 'ICD9_CODE']

    X = df[features].copy()
    Y = df['DEAD_AFTER_DURATION'].copy()
    return X, Y

def trainTest(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    imputer = SimpleImputer(strategy='most_frequent')
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X.columns)

    return X_train, X_test, Y_train, Y_test

def tune_model(model, params, X_train, Y_train, cv=5, scoring='accuracy'):
    grid_search = GridSearchCV(model, params, cv=cv, scoring=scoring, n_jobs=-1)
    grid_search.fit(X_train, Y_train)
    return grid_search.best_estimator_

def evaluate(model, X_test, Y_test):
    Y_pred = model.predict(X_test)
    acc = accuracy_score(Y_test, Y_pred)
    report = classification_report(Y_test, Y_pred, output_dict=True)
    weighted_f1 = report['weighted avg']['f1-score']
    print('Accuracy:', acc)
    print('Classification Report:')
    #Inefficient but oh well idk how to do dict conversions
    print(classification_report(Y_test, Y_pred))
    print('Confusion Matrix:')
    print(confusion_matrix(Y_test, Y_pred))

    return acc, weighted_f1

def zeroR(Y_train, Y_test):
    majority_class = Y_train.mode()[0]
    Y_pred = np.full_like(Y_test, majority_class)
    acc = accuracy_score(Y_test, Y_pred)
    report = classification_report(Y_test, Y_pred, output_dict=True)
    weighted_f1 = report['weighted avg']['f1-score']
    print('\n----- ZeroR Classifier -----')
    print('Accuracy:', acc)
    print('Classification Report:')
    print(classification_report(Y_test, Y_pred))
    print('Confusion Matrix:')
    print(confusion_matrix(Y_test, Y_pred))

    return acc, weighted_f1

def main():
    acc_list = []
    weighted_f1_list = []

    patients, admissions, diagnoses = loadData()

    df = mergeLabel(patients, admissions, diagnoses)

    df = preprocess(df)

    X, Y = featuresTarget(df)

    X_train, X_test, Y_train, Y_test = trainTest(X, Y) 

    #ZeroR
    test_acc, test_f1 = zeroR(Y_train, Y_test)

    acc_list.append(test_acc)
    weighted_f1_list.append(test_f1)

    classifiers = {
        'KNN': (KNeighborsClassifier(), {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance']
        }),
        'Logistic Regression': (LogisticRegression(), {
            'penalty': ['l2'],
            'C': [0.1, 1.0, 10.0],
            'class_weight': ['balanced'],
            'solver': ['lbfgs'],
            'max_iter': [10000]
        })
    }

    for name, (model, params) in classifiers.items():
        print(f'\n==================\n  {name} Classifier \n==================')
        trained_model = tune_model(model, 
                                   params, 
                                   X_train, 
                                   Y_train, 
                                   cv=5, 
                                   scoring='accuracy')
        
        test_acc, test_f1 = evaluate(trained_model, X_test, Y_test)

        acc_list.append(test_acc)
        weighted_f1_list.append(test_f1)

    #Plotting hehehehe
    classifiers = ['ZeroR', 'KNN', 'Logistic Regression']

    temp = []
    for classifier in range(len(classifiers)):
        temp.append(classifiers[classifier] + ' (' + str(round(acc_list[classifier], 2)) + ')')

    plt.figure(figsize=(12, 5))
    
    plt.subplot(2, 1, 1)
    plt.bar(temp, acc_list)
    plt.title('Accuracy')
    plt.ylabel('Accuracy Probability')
    plt.ylim(0, 1)

    temp = []
    for classifier in range(len(classifiers)):
        temp.append(classifiers[classifier] + ' (' + str(round(weighted_f1_list[classifier], 2)) + ')')

    plt.subplot(2, 1, 2)
    plt.bar(temp, weighted_f1_list)
    plt.title('Weighted F1')
    plt.ylabel('Weighted F1 Probability')
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.show()

main()