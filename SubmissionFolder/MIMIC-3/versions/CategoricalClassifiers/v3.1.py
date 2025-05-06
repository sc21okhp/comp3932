import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
#Preprocess
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
#KNN
from sklearn.neighbors import KNeighborsClassifier
#LogReg
from sklearn.linear_model import LogisticRegression
#Bagging & Boosting
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
#Random Forest
from sklearn.ensemble import RandomForestClassifier
#Evaluation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, brier_score_loss

global LABELDEATHDURATION
LABELDEATHDURATION = 360
death_threshold_year = LABELDEATHDURATION / 365.25

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
    global LABELDEATHDURATION
    if pd.isnull(row['DOD']) or pd.isnull(row['ADMITTIME']):
        return 0

    duration = (row['DOD'] - row['ADMITTIME']).days
    return 1 if duration >= 0 and duration <= LABELDEATHDURATION else 0

def processPatients(patients, admissions, diagnoses):
    #Create empty features list to fill and return
    features = []

    #!RIGHT CENSORED
    alive_patients = patients[patients['EXPIRE_FLAG'] == 0].copy()
    alive_patient_ids = set(alive_patients['SUBJECT_ID'])

    #Get all admissions and diagnoses for alive patients
    alive_admissions = admissions[admissions['SUBJECT_ID'].isin(alive_patient_ids)].copy()
    alive_diagnoses = diagnoses[diagnoses['SUBJECT_ID'].isin(alive_patient_ids)].copy()

    tempDF = alive_admissions.merge(alive_patients, on='SUBJECT_ID', how='left')
    df = tempDF.merge(alive_diagnoses, on='SUBJECT_ID', how='left')

    #Update the values to Pandas time datatype
    alive_patients['DOB'] = pd.to_datetime(alive_patients['DOB'], errors='coerce')
    alive_admissions['ADMITTIME'] = pd.to_datetime(alive_admissions['ADMITTIME'], errors='coerce')

    catagorical = ['GENDER', 'ADMISSION_TYPE', 'ADMISSION_LOCATION', 'INSURANCE', 'LANGUAGE', 'RELIGION', 'MARITAL_STATUS', 'ETHNICITY', 'ICD9_CODE']

    #Apply encoding to catagorical variables
    for col in catagorical:
        df[col] = df[col].fillna('UNKNOWN')
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    #Deal with numerical NAN values
    df['SEQ_NUM'] = df['SEQ_NUM'].fillna(0)

    #Apply label for point prediction
    df['DEAD_AFTER_DURATION'] = df.apply(labelDeath, axis=1)

    for subject in alive_patients['SUBJECT_ID'].unique():
        #Get the patient in questions row
        patient_row = alive_patients[alive_patients['SUBJECT_ID'] == subject].iloc[0]
        #Get the patient admissions 
        patient_admits = alive_admissions[alive_admissions['SUBJECT_ID'] == subject]

        #If patient is dead but has no admissions or DOB ignore
        if patient_admits.empty or pd.isnull(patient_row['DOB']):
            print(f'\n\nSUBJECT_ID: "{subject}" Has either no recorded admissions or doesnt have a dob\n\n ')
            continue

        #First, last and total admissions
        first_admit = patient_admits['ADMITTIME'].min().to_pydatetime()
        last_admit = patient_admits['ADMITTIME'].max().to_pydatetime()
    
        num_admissions = len(patient_admits)
        #Get the number of diagnoses
        num_diagnoses = len(alive_diagnoses[alive_diagnoses['SUBJECT_ID'] == subject])

        #Store dob as python date objects (Previously Pandas)
        dob = patient_row['DOB'].to_pydatetime()

        #Try to record age at admission
        try:
            age_at_admission = (first_admit - dob).days / 365.25

        except OverflowError:
            print(f'\n\nSkipping SUBJECT_ID: "{subject}" due to date issue\n\n')
            continue

        #Categorical Values
        gender              = df[df['SUBJECT_ID'] == subject].iloc[0]['GENDER']
        admission_type      = df[df['SUBJECT_ID'] == subject].iloc[0]['ADMISSION_TYPE']
        admission_location  = df[df['SUBJECT_ID'] == subject].iloc[0]['ADMISSION_LOCATION']
        insurance           = df[df['SUBJECT_ID'] == subject].iloc[0]['INSURANCE']
        language            = df[df['SUBJECT_ID'] == subject].iloc[0]['LANGUAGE']
        religion            = df[df['SUBJECT_ID'] == subject].iloc[0]['RELIGION']
        marital_status      = df[df['SUBJECT_ID'] == subject].iloc[0]['MARITAL_STATUS']
        ethnicity           = df[df['SUBJECT_ID'] == subject].iloc[0]['ETHNICITY']
        icd9_code           = df[df['SUBJECT_ID'] == subject].iloc[0]['ICD9_CODE']

        #Numerical Values
        hadm_id             = df[df['SUBJECT_ID'] == subject].iloc[0]['HADM_ID']
        seq_num             = df[df['SUBJECT_ID'] == subject].iloc[0]['SEQ_NUM']

        #Dead = 1, Survive until threshold = 0
        #Not to do with censorship
        event = 0

        #We don't know much about duration for right censored data
        #This is the best we can do
        duration = (last_admit - first_admit).days / 365.25

        if duration < 0:
            print(f'\n\nSkipping SUBJECT_ID: "{subject}" due to date issue\n\n')
            continue

        if duration <= 0:
            duration == 1

        #Append subjects features
        features.append({
            'SUBJECT_ID': subject,
            'censored': False, #ScikitSurv expects Bools
            'DEAD_AFTER_DURATION': event,
            'duration': duration,
            'age_at_admission': age_at_admission,
            #'age_at_death': age_at_death, Can't use because model shouldn't have this data
            'num_admissions': num_admissions,
            'num_diagnoses': num_diagnoses,
            'gender': gender,
            'admission_type': admission_type,
            'admission_location': admission_location,
            'insurance': insurance,
            'language': language,
            'religion': religion,
            'marital_status': marital_status,
            'ethnicity': ethnicity,
            'icd9_code': icd9_code,
            'hadm_id': hadm_id,
            'seq_num': seq_num
        })


    #!DEAD PATIENTS
    #Select all patients that are dead
    patients = patients[patients['EXPIRE_FLAG'] == 1].copy()
    #Get dead patient ID
    dead_patients_ids = set(patients['SUBJECT_ID'])

    #Get all admissions and diagnoses for dead patients
    admissions = admissions[admissions['SUBJECT_ID'].isin(dead_patients_ids)].copy()
    diagnoses = diagnoses[diagnoses['SUBJECT_ID'].isin(dead_patients_ids)].copy()

    #Update the values to Pandas time datatype
    patients['DOB'] = pd.to_datetime(patients['DOB'], errors='coerce')
    patients['DOD'] = pd.to_datetime(patients['DOD'], errors='coerce')
    admissions['ADMITTIME'] = pd.to_datetime(admissions['ADMITTIME'], errors='coerce')
    
    #Move all patients, admissions and diagnoses into one df
    tempDF = admissions.merge(patients, on='SUBJECT_ID', how='left')
    df = tempDF.merge(diagnoses, on='SUBJECT_ID', how='left')

    #Apply label for point prediction
    df['DEAD_AFTER_DURATION'] = df.apply(labelDeath, axis=1)

    #All the numerical and catagorical columns
    numerical = ['HADM_ID', 'SEQ_NUM']
    catagorical = ['GENDER', 'ADMISSION_TYPE', 'ADMISSION_LOCATION', 'INSURANCE', 'LANGUAGE', 'RELIGION', 'MARITAL_STATUS', 'ETHNICITY', 'ICD9_CODE']

    #Apply encoding to catagorical variables
    for col in catagorical:
        df[col] = df[col].fillna('UNKNOWN')
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    #Deal with numerical NAN values
    df['SEQ_NUM'] = df['SEQ_NUM'].fillna(0)

    for subject in patients['SUBJECT_ID'].unique():
        #Get the patient in questions row
        patient_row = patients[patients['SUBJECT_ID'] == subject].iloc[0]

        #Get the patient admissions 
        patient_admits = admissions[admissions['SUBJECT_ID'] == subject]

        #If patient is dead but has no admissions or DOB ignore
        if patient_admits.empty or pd.isnull(patient_row['DOB']):
            print(f'\n\nSUBJECT_ID: "{subject}" Has either no recorded admissions or doesnt have a dob\n\n')
            continue

        #If patient is dead but no date attributed ignore
        if pd.isnull(patient_row['DOD']):
            print(f'\n\nSUBJECT_ID: "{subject}" Has either been labelled dead incorrectly or death wasnt recorded\n\n')
            continue

        #First and total admissions
        first_admit = patient_admits['ADMITTIME'].min().to_pydatetime()
        num_admissions = len(patient_admits)

        #Get the number of diagnoses
        num_diagnoses = len(diagnoses[diagnoses['SUBJECT_ID'] == subject])

        #Store dob and dod as python date objects (Previously Pandas)
        dob = patient_row['DOB'].to_pydatetime()
        dod = patient_row['DOD'].to_pydatetime()

        #Can't think of a better word to describe how long people survive after admission
        #Dod is always 00:00 on the day so can have negative duration
        duration = (dod + timedelta(days=1) - first_admit).days / 365.25

        if duration < 0:
            print(f'\n\nSkipping SUBJECT_ID: "{subject}" due to date issue\n\n')
            continue

        #Try to record age at admission and death
        try:
            age_at_admission = (first_admit - dob).days / 365.25

            age_at_death = (dod - dob).days / 365.25
        except OverflowError:
            print(f'\n\nSkipping SUBJECT_ID: "{subject}" due to date issue\n\n')
            continue
        
        #Categorical Values
        gender = df[df['SUBJECT_ID'] == subject].iloc[0]['GENDER']
        admission_type = df[df['SUBJECT_ID'] == subject].iloc[0]['ADMISSION_TYPE']
        admission_location = df[df['SUBJECT_ID'] == subject].iloc[0]['ADMISSION_LOCATION']
        insurance = df[df['SUBJECT_ID'] == subject].iloc[0]['INSURANCE']
        language = df[df['SUBJECT_ID'] == subject].iloc[0]['LANGUAGE']
        religion = df[df['SUBJECT_ID'] == subject].iloc[0]['RELIGION']
        marital_status = df[df['SUBJECT_ID'] == subject].iloc[0]['MARITAL_STATUS']
        ethnicity = df[df['SUBJECT_ID'] == subject].iloc[0]['ETHNICITY']
        icd9_code = df[df['SUBJECT_ID'] == subject].iloc[0]['ICD9_CODE']

        #Numerical Values
        hadm_id = df[df['SUBJECT_ID'] == subject].iloc[0]['HADM_ID']
        seq_num = df[df['SUBJECT_ID'] == subject].iloc[0]['SEQ_NUM']

        #Dead = 1, Survive until threshold = 0
        event = 0
        if duration <= death_threshold_year:
            event = 1

        #Append subjects features
        features.append({
            'SUBJECT_ID': subject,
            'censored': True,
            'DEAD_AFTER_DURATION': event,
            'duration': duration,
            'age_at_admission': age_at_admission,
            #'age_at_death': age_at_death, Can't use because model shouldn't have this data
            'num_admissions': num_admissions,
            'num_diagnoses': num_diagnoses,
            'gender': gender,
            'admission_type': admission_type,
            'admission_location': admission_location,
            'insurance': insurance,
            'language': language,
            'religion': religion,
            'marital_status': marital_status,
            'ethnicity': ethnicity,
            'icd9_code': icd9_code,
            'hadm_id': hadm_id,
            'seq_num': seq_num
        })
    
    return pd.DataFrame(features)

def featuresTarget(df):
    features = ['age_at_admission', 'gender', 'admission_type', 'admission_location', 'insurance', 'language', 'religion', 'marital_status', 'ethnicity', 'hadm_id', 'seq_num', 'icd9_code']

    X = df[features].copy()
    Y = df['DEAD_AFTER_DURATION'].copy()
    return X, Y

def splitScale(X, Y):
    X_train_eval, X_test, Y_train_eval, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    X_train, X_eval, Y_train, Y_eval = train_test_split(X_train_eval, Y_train_eval, test_size=0.5, random_state=42)

    imputer = SimpleImputer(strategy='most_frequent')
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X.columns)
    X_eval = pd.DataFrame(imputer.transform(X_eval), columns=X.columns)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_eval = scaler.transform(X_eval)

    return X_train, X_test, X_eval, Y_train, Y_test, Y_eval

def tune_model(model, params, X_train, Y_train, cv=5, scoring='roc_auc'):
    grid_search = GridSearchCV(model, params, cv=cv, scoring=scoring, n_jobs=-1)
    grid_search.fit(X_train, Y_train)
    return grid_search.best_estimator_

def rocAuc(model, X, Y):
    #Not all scikit functions have predict_proba :(((
    if hasattr(model, 'predict_proba'):
        Y_pred = model.predict_proba(X)[:, 1]
    elif hasattr(model, 'decision_function'):
        Y_pred = model.decision_function(X)
    else:
        return None
    return roc_auc_score(Y, Y_pred)

def brierScore(model, X, Y):
    #Not all scikit functions have predict_proba :(((
    if hasattr(model, 'predict_proba'):
        Y_pred = model.predict_proba(X)[:, 1]
    elif hasattr(model, 'decision_function'):
        Y_pred = model.decision_function(X)
    else:
        return None
    return brier_score_loss(Y, Y_pred)

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

    auc = rocAuc(model, X_test, Y_test)
    print('ROC-AUC:', auc)

    brier = brierScore(model, X_test, Y_test)
    print('Brier Score:', brier)

    return acc, weighted_f1, auc, brier

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
    auc_list = []
    brier_list = []

    patients, admissions, diagnoses = loadData()

    df = processPatients(patients, admissions, diagnoses)

    X, Y = featuresTarget(df)

    X_train, X_test, X_val, Y_train, Y_test, Y_val = splitScale(X, Y) 

    smote = SMOTE(random_state=42)
    X_train, Y_train = smote.fit_resample(X_train, Y_train)

    #ZeroR
    test_acc, test_f1 = zeroR(Y_train, Y_test)

    acc_list.append(test_acc)
    weighted_f1_list.append(test_f1)
    auc_list.append(0)
    brier_list.append(0)

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
        }),
        # Uses decision tree classifier as default
        'Bagging': (BaggingClassifier(), {
            'n_estimators': [5, 10, 20]
        }),
        # Same as above
        'Boosting': (AdaBoostClassifier(), {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.5, 1.0, 1.5]
        }),
        'Random Forest': (RandomForestClassifier(), {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5, 10],
            'class_weight': ['balanced']
        })
    }

    print("Starting Classification:")
    for name, (model, params) in classifiers.items():
        print(f'\n==================\n  {name} Classifier \n==================')
        #First batch of training without validation set
        first_tune_model = tune_model(model, 
                                      params, 
                                      X_train, 
                                      Y_train, 
                                      cv=5, 
                                      scoring='accuracy')

        #Now we can combine validation set with training set
        X_train_full = np.concatenate((X_train, X_val))
        Y_train_full = np.concatenate((Y_train, Y_val))

        #Second batch of training with validation set
        second_tune_model = tune_model(first_tune_model, 
                                       params, 
                                       X_train_full, 
                                       Y_train_full, 
                                       cv=5, 
                                       scoring='accuracy')
        test_acc, test_f1, auc, brier = evaluate(second_tune_model, X_test, Y_test)

        acc_list.append(test_acc)
        weighted_f1_list.append(test_f1)
        auc_list.append(auc)
        brier_list.append(brier)

    #Plotting hehehehe
    classifiers = ['ZeroR', 'KNN', 'Logistic Regression', 'Bagging', 'Boosting', 'Random Forest']
    plt.figure(figsize=(12, 5))

    #Accuracy
    temp = []
    for classifier in range(len(classifiers)):
        temp.append(classifiers[classifier] + ' (' + str(round(acc_list[classifier], 2)) + ')')
    
    plt.subplot(2, 1, 1)
    plt.bar(temp, acc_list)
    plt.title('Accuracy')
    plt.ylabel('Accuracy Probability')
    plt.ylim(0, 1)

    #Weighted F1
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

    classifiers = ['KNN', 'Logistic Regression', 'Bagging', 'Boosting', 'Random Forest']
    auc_list = auc_list[1:]
    brier_list = brier_list[1:]
    plt.figure(figsize=(12, 5))

    #ROC-AUC
    temp = []
    for classifier in range(len(classifiers)):
        temp.append(classifiers[classifier] + ' (' + str(round(auc_list[classifier], 2)) + ')')

    plt.subplot(2, 1, 1)
    plt.bar(temp, auc_list)
    plt.title('ROC-AUC')
    plt.ylabel('ROC Area Under Curve')
    plt.ylim(0, 1)

    #Brier-Score
    temp = []
    for classifier in range(len(classifiers)):
        temp.append(classifiers[classifier] + ' (' + str(round(brier_list[classifier], 2)) + ')')

    plt.subplot(2, 1, 2)
    plt.bar(temp, brier_list)
    plt.title('Brier-Score')
    plt.ylabel('Brier Score')
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.show()

main()