import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sksurv.util import Surv
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import integrated_brier_score, cumulative_dynamic_auc

#Python is such a hilarious language
#I don't even need the global identifier 
LABELDEATHDURATION = 360
death_threshold_year = LABELDEATHDURATION / 365.25
DYNAMICAUC = False

def loadData():
    #SUBJECT_ID, GENDER, DOB, DOD, EXPIRE_FLAG
    patients = pd.read_csv('MIMIC-3/dataset/PATIENTS/PATIENTS_sorted.csv')
    patients = patients[['SUBJECT_ID', 
                         'GENDER', 
                         'DOB', 
                         'DOD', 
                         'EXPIRE_FLAG'
                         ]].copy()
    #SUBJECT_ID, ADMITTIME, ADMISSION_TYPE, INSURANCE, LANGUAGE, RELIGION, ETHNICITY 
    admissions = pd.read_csv('MIMIC-3/dataset/ADMISSIONS/ADMISSIONS_sorted.csv')
    admissions = admissions[['SUBJECT_ID', 
                             'ADMITTIME', 
                             'ADMISSION_TYPE', 
                             'ADMISSION_LOCATION', 
                             'INSURANCE', 
                             'LANGUAGE', 
                             'RELIGION', 
                             'MARITAL_STATUS', 
                             'ETHNICITY'
                             ]].copy()
    #SUBJECT_ID, HADM_ID, SEQ_NUM, ICD9_CODE
    diagnoses = pd.read_csv('MIMIC-3/dataset/DIAGNOSES_ICD/DIAGNOSES_ICD_sorted.csv')
    diagnoses = diagnoses[['SUBJECT_ID', 
                           'HADM_ID', 
                           'SEQ_NUM', 
                           'ICD9_CODE'
                           ]].copy()

    return patients, admissions, diagnoses

def labelDeath(row):
    if pd.isnull(row['DOD']) or pd.isnull(row['ADMITTIME']):
        return 0

    duration = (row['DOD'] - row['ADMITTIME']).days
    return 1 if duration >= 0 and duration <= LABELDEATHDURATION else 0

def labelDurationEvent(row):
    #If they're dead
    if not row['censored']:
        duration = (row['DOD'] + timedelta(days=1) - row['first_admit']).days / 365.25
        event = 0
        if duration <= death_threshold_year:
            event = 1

    #Not dead
    else:
        duration = (row['last_admit'] - row['first_admit']).days / 365.25
        event  = 0

    if duration < 0:
        duration = 0

    return pd.Series({'duration': duration, 'event': event})

def merge(patients, admissions, diagnoses):
    patients['DOD'] = pd.to_datetime(patients['DOD'], errors='coerce')
    patients['DOB'] = pd.to_datetime(patients['DOB'], errors='coerce')
    admissions['ADMITTIME'] = pd.to_datetime(admissions['ADMITTIME'], errors='coerce')

    admissions_patients = admissions.merge(patients, on='SUBJECT_ID', how='left')
    admissions_patients_diagnoses = admissions_patients.merge(diagnoses, on='SUBJECT_ID', how='left')

    return admissions_patients_diagnoses

def processPatients(patients, admissions, diagnoses):
    df = merge(patients, admissions, diagnoses)

    catagorical = ['GENDER', 'ADMISSION_TYPE', 'ADMISSION_LOCATION', 'INSURANCE', 'LANGUAGE', 'RELIGION', 'MARITAL_STATUS', 'ETHNICITY', 'ICD9_CODE']

    for col in catagorical:
        df[col] = df[col].fillna('UNKNOWN')
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    df['SEQ_NUM'] = df['SEQ_NUM'].fillna(0)

    #Stupid calculation took me way too long
    #Have to do it as another table because 
    #you need every admission per SUBJECT_ID
    admit_bounds = (
        df.groupby('SUBJECT_ID')['ADMITTIME']
          .agg(first_admit='min', last_admit='max')
          .reset_index()
    )
    df = df.merge(admit_bounds, on='SUBJECT_ID', how='left')

    #Sets it as a bool
    df['censored'] = (df['EXPIRE_FLAG'] == 0)

    #Duration = How long they survive after first admit
    #Event = Whether they died before point threshold
    df[['duration', 'event']] = df.apply(labelDurationEvent, axis=1)

    return df

def trainTestSplit(df):
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    return train, test

def getXY(df):
    #Can't find a better way to format rip
    #As always
    #Y=Target
    #X=Features

    #Force Y to have typing I want 
    #Unsure why when they're declared they change or smth 
    Y = Surv.from_arrays(
        event=df['censored'].astype(bool), 
        time=df['duration'].astype(float)
    )

    #Only want the features to end up in X
    X = df.drop(
        columns=['SUBJECT_ID', 
                 'event', 
                 'censored', 
                 'duration',
                 'ADMITTIME',
                 'DOB',
                 'DOD',
                 'first_admit',
                 'last_admit',
                 'HADM_ID'
                 ]
    )

    imputer = SimpleImputer(strategy='most_frequent')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return Y, X

def trainCox(train):
    Y_train, X_train = getXY(train)

    cph = CoxPHSurvivalAnalysis()
    cph.fit(X_train, Y_train)
    return cph

def trainElasticCox(train, 
                    n_alphas=100, 
                    alpha_min_ratio='auto',
                    l1_ratio=0.5,
                    normalize=True):
    
    Y_train, X_train = getXY(train)

    enc = CoxnetSurvivalAnalysis(
        n_alphas=n_alphas,
        alpha_min_ratio=alpha_min_ratio,
        l1_ratio=l1_ratio,
        normalize=normalize
    )

    enc.fit(X_train, Y_train)
    return enc

def trainRSF(train, 
             n_estimators=100, 
             min_samples_split=15,
             min_samples_leaf=10, 
             max_features='sqrt', 
             random_state=42, 
             n_jobs=-1):
    
    Y_train, X_train = getXY(train)

    rsf = RandomSurvivalForest(
        n_estimators=n_estimators,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        n_jobs=n_jobs,
        random_state=random_state
    )

    rsf.fit(X_train, Y_train)
    return rsf

def generateAccReportMatrix(Y_test, Y_pred, classifier):
    acc = accuracy_score(Y_test, Y_pred)
    report = classification_report(Y_test, Y_pred, output_dict=True, zero_division=0)
    weighted_f1 = report['weighted avg']['f1-score']

    print(f'\n==================\n  {classifier} Classifier at {LABELDEATHDURATION} Days\n==================')
    print('Accuracy: ', acc)
    print('Classification Report:')
    print(classification_report(Y_test, Y_pred, zero_division=0))
    print('Confusion Matrix:')
    print(confusion_matrix(Y_test, Y_pred))
    print()

    return acc, weighted_f1

def calculateIBS(test, Y_train, Y_test, test_surv_funcs):
    #Get the longest duration
    max_time = test['duration'].max()
    #Get 1000 equally spaced steps between 0 and longest duration
    time_grid = np.linspace(0, max_time, 1000, endpoint=False)
    #Work out prediction for each time step
    ibs_test_surv_probs = np.asarray([[fn(t) for t in time_grid] for fn in test_surv_funcs])
    ibs = integrated_brier_score(
        Y_train, 
        Y_test, 
        ibs_test_surv_probs, 
        time_grid
    )   

    print(f'Integrated Brier Score: {ibs}')
    print()

def pointAUC(threshold_test_surv_probs, Y_point_test):
    #Couldn't figure out scikit's roc_auc_score function
    #So I just put two together haha
    #I think maybe scikit's base one doesn't work for timed stuff???
    threshold_test_death_probs = 1.0 - threshold_test_surv_probs

    false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_point_test, threshold_test_death_probs, pos_label=1)

    auc_result = auc(false_positive_rate, true_positive_rate)

    print(f'ROC AUC: {auc_result}')

def dynamicAUC(model, Y_test, Y_train, test, X_test, model_name):
    #Get the longest duration
    max_time = test['duration'].max()
    #Get 1000 equally spaced steps between 0 and longest duration
    time_grid = np.linspace(0, max_time, 1000, endpoint=False)

    risk_scores = model.predict(X_test)

    auc, mean_auc = cumulative_dynamic_auc(
        Y_train,
        Y_test,
        risk_scores,
        time_grid
    )

    plt.figure(figsize=(12, 5))
    plt.plot(time_grid, auc, marker='o', markersize='2', alpha=0.75, label=f'{model_name} (iAUC={mean_auc:.3f})')
    plt.xlabel('Time (Years)')
    plt.ylabel('AUC')
    plt.title('Cumulative Dynamic AUC over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f'Mean Dynamic Auc: {mean_auc}')

def evaluate_survival_model(model, model_name, train, test):
    Y_train, X_train = getXY(train)
    Y_test, X_test = getXY(test)

    #Get all survival functions for the test and training sets
    test_surv_funcs = model.predict_survival_function(X_test)
    train_surv_funcs = model.predict_survival_function(X_train)

    #Numpy to the rescue saving me a for loop
    threshold_test_surv_probs = np.array([fn(death_threshold_year) for fn in test_surv_funcs])

    #Get actual and predicted values at chosen threshold
    Y_point_test = test['event'].values
    Y_point_pred = (threshold_test_surv_probs < 0.5).astype(int)

    generateAccReportMatrix(Y_point_test, Y_point_pred, model_name)

    pointAUC(threshold_test_surv_probs, Y_point_test)

    calculateIBS(test, Y_train, Y_test, test_surv_funcs)

    if DYNAMICAUC:
        dynamicAUC(model, Y_test, Y_train, test, X_test, model_name)

def main():
    patients, admissions, diagnoses = loadData()

    features = processPatients(patients, admissions, diagnoses)

    train, test = trainTestSplit(features)

    #Cox's Proportional Hazard
    cph_model = trainCox(train)

    #Random Survival Forests
    rsf_model = trainRSF(train)

    #Elastic Net Cox
    enc_model = trainElasticCox(train)

    evaluate_survival_model(cph_model, 'CPH', train, test)

    evaluate_survival_model(rsf_model, 'RSF', train, test)

    evaluate_survival_model(enc_model, 'ENC', train, test)

main()