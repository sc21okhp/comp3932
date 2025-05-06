import pandas as pd
import numpy as np
#Preprocess
from sklearn.model_selection import train_test_split
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

def evaluate(model, X_test, Y_test):
    Y_pred = model.predict(X_test)
    acc = accuracy_score(Y_test, Y_pred)
    print('Accuracy:', acc)
    print('Classification Report:')
    #Inefficient but oh well idk how to do dict conversions
    print(classification_report(Y_test, Y_pred))
    print('Confusion Matrix:')
    print(confusion_matrix(Y_test, Y_pred))

def zeroR(Y_train, Y_test):
    majority_class = Y_train.mode()[0]
    Y_pred = np.full_like(Y_test, majority_class)
    acc = accuracy_score(Y_test, Y_pred)
    print('\n----- ZeroR Classifier -----')
    print('Accuracy:', acc)
    print('Classification Report:')
    print(classification_report(Y_test, Y_pred))
    print('Confusion Matrix:')
    print(confusion_matrix(Y_test, Y_pred))

def main():
    patients, admissions, diagnoses = loadData()

    df = mergeLabel(patients, admissions, diagnoses)

    df = preprocess(df)

    X, Y = featuresTarget(df)

    X_train, X_test, Y_train, Y_test = trainTest(X, Y) 

    #ZeroR
    zeroR(Y_train, Y_test)

    #KNN
    print('\n----- KNN Classifier -----')
    knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')
    knn.fit(X_train, Y_train)
    evaluate(knn, X_test, Y_test)

    #Logistic Regression
    print('\n----- Logistic Regression Classifier -----')
    lr = LogisticRegression(penalty='l2', C=1.0, class_weight='balanced', max_iter=1000)
    lr.fit(X_train, Y_train)
    evaluate(lr, X_test, Y_test)

main()