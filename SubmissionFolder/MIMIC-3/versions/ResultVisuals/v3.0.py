import matplotlib.pyplot as plt
import pandas as pd

accuracy_df = pd.read_csv('MIMIC-3/results/accuracy.csv', index_col='Model')
brier_score_df = pd.read_csv('MIMIC-3/results/brier_score.csv', index_col='Model')
c_index_df = pd.read_csv('MIMIC-3/results/c_index.csv', index_col='Model')
ibs_df = pd.read_csv('MIMIC-3/results/ibs.csv', index_col='Model')
roc_auc_df = pd.read_csv('MIMIC-3/results/roc_auc.csv', index_col='Model')
weighted_f1_df = pd.read_csv('MIMIC-3/results/weighted_f1.csv', index_col='Model')

def line(df, metric_title, metric_name):
    markers = ['v', '^', '<', '>', 's', 'o']
    
    #This can't be one line???? Inline lovers in shambles
    df.columns = df.columns.astype(float)
    time_points = df.columns

    if 'ZeroR' in df.index:
        categorical = ['ZeroR', 'KNN', 'Log-Reg', 'Bagging', 'Boosting', 'RFC']
        survival = ['ZeroR', 'CPH', 'CoxNet', 'RSF']
    
    else:
        categorical = ['KNN', 'Log-Reg', 'Bagging', 'Boosting', 'RFC']
        survival = ['CPH', 'CoxNet', 'RSF']
        
    categorical_df = df.loc[categorical, time_points]
    survival_df = df.loc[survival, time_points]

    #!Categorical
    plt.figure(figsize=(12, 7))
    plt.subplot(2, 1, 1)
    m = 0
    for model in categorical_df.index:
        plt.plot(time_points, 
                 categorical_df.loc[model], 
                 marker=markers[m], 
                 markersize='4', 
                 alpha=0.6, 
                 label=model)
        m += 1 

    plt.xlabel('Time Threshold (Days)')
    plt.ylabel(str(metric_name) + ' Probability')
    plt.title(str(metric_title) + ' (Categorical Algorithms)')
    plt.legend(loc='best')
    plt.grid(True)

    #!Survival
    plt.subplot(2, 1, 2)
    m = 0
    for model in survival_df.index:
        plt.plot(time_points, 
                 survival_df.loc[model], 
                 marker=markers[m], 
                 markersize='4', 
                 alpha=0.6, 
                 label=model)
        m += 1

    plt.xlabel('Time Threshold (Days)')
    plt.ylabel(str(metric_name) + ' Probability')
    plt.title(str(metric_title) + ' (Survival Algorithms)')
    plt.legend(loc='best')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

line(accuracy_df, 'Point Accuracy Of Models', 'Accuracy')
line(weighted_f1_df, 'Weighted F1 Score Of Models', 'Weighted F1')
line(roc_auc_df, 'ROC-AUC Of Models', 'ROC-AUC')
line(brier_score_df, 'Brier Score Of Models', 'Brier Score')