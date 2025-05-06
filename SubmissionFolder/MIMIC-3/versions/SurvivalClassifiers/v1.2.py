import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


global LABELDEATHDURATION
LABELDEATHDURATION = 360

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
    global LABELDEATHDURATION
    #Assign given threshold
    death_threshold_year = LABELDEATHDURATION / 365.25

    #Create empty features list to fill and return
    features = []

    #!RIGHT CENSORED
    alive_patients = patients[patients['EXPIRE_FLAG'] == 0].copy()

    alive_patient_ids = set(alive_patients['SUBJECT_ID'])

    #Get all admissions and diagnoses for dead patients
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

        #We don't know much about duration for right censored data
        #This is the best we can do
        duration = (last_admit - first_admit).days / 365.25

        #Append subjects features
        features.append({
            'SUBJECT_ID': subject,
            'censored': 0,
            'event': event,
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
        duration = (dod - first_admit).days / 365.25

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
            'censored': 1,
            'event': event,
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

def calcHazard(features, patient_id):
    global LABELDEATHDURATION

    print(features)

    #Get the features for the given patient (For testing purposes)
    patient_features = features[features['SUBJECT_ID'] == patient_id]

    #If you put a subject id that doesn't exist in that's on you
    if patient_features.empty:
        raise ValueError(f'Cannot find SUBJECT_ID: {patient_id}')

    #Split the data into train and testing
    train, test = train_test_split(features, test_size=0.2, random_state=42)

    #Create a CPH model
    #Newton-Raphson convergence error so penalizer added (Thank you Mrs Russell)
    cph = CoxPHFitter(penalizer=0.001)
    #Train the CPH model without subject ID 
    cph.fit(
        df=train.drop(columns=['SUBJECT_ID', 'event']),
        duration_col='duration',
        event_col='censored',
        robust=True
    )

    #Predict the risk associated with chosen patient
    risk_score = cph.predict_partial_hazard(patient_features.drop(columns=['SUBJECT_ID', 'event'])).values[0]
    #Work out survival probability for chosen patient after one year 
    survival_prob_one_year = cph.predict_survival_function(patient_features.drop(columns=['SUBJECT_ID', 'event']), times=[1]).iloc[0, 0]

    result = {
        'patient_id': patient_id,
        'risk_score': float(risk_score),
        'survival_probability_at_1_year': float(survival_prob_one_year),
        'features': patient_features.to_dict(orient='records')[0]
    }

    #Calculate the number of years the duration is
    time_threshold = LABELDEATHDURATION / 365.25
    #Get predictions for all test set subjects at the time threshold
    threshold_survival = cph.predict_survival_function(test.drop(columns=['SUBJECT_ID']), times=[time_threshold])

    #If subject is predicted to survive past threshold = 0, 
    #Predicted to die = 1
    Y_pred = (threshold_survival.iloc[0] < 0.5).astype(int).values
    Y_test = test['event'].values

    # Compute evaluation metrics using scikit-learn.
    acc = accuracy_score(Y_test, Y_pred)
    report = classification_report(Y_test, Y_pred, output_dict=True, zero_division=0)
    weighted_f1 = report['weighted avg']['f1-score']

    print(f'\n==================\n  CPH Classifier \n==================')
    print('Accuracy:', acc)
    print('Classification Report:')
    print(classification_report(Y_test, Y_pred, zero_division=0))
    print('Confusion Matrix:')
    print(confusion_matrix(Y_test, Y_pred))
    print()
    print()

    return cph, result

def plotModel(cph):
    plt.figure(figsize=(12, 5))

    baseline_hazard = cph.baseline_hazard_
    plt.plot(baseline_hazard.index, baseline_hazard.iloc[:, 0], label='Baseline Hazard')
    plt.xlabel('Time')
    plt.ylabel('Hazard')
    plt.title('Baseline Hazard Function')
    plt.legend()
    plt.show()
    
    coefs = cph.params_
    errors = cph.standard_errors_
    plt.figure(figsize=(12, 5))
    plt.bar(coefs.index, coefs.values, yerr=errors.values, capsize=5)
    plt.xlabel('Covariates')
    plt.ylabel('Coefficient')
    plt.title('Cox PH Model Coefficients')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

patient_id = 420

patients, admissions, diagnoses = loadData()

features = processPatients(patients, admissions, diagnoses)

cph, hazard_details = calcHazard(features, patient_id)

print(f'Hazard Details for Patient with SUBJECT_ID: {patient_id} \n{hazard_details}')

print(f'\n==================\nHazard Model:\n==================')
cph.print_summary()

cph.plot(hazard_ratios=True)

plotModel(cph)