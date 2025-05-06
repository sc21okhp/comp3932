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

def meanAndSdBar(df, metric_title, metric_name):
    means = df.mean(axis=1)
    std  = df.std(axis=1)
    
    #Sort the mean values and then apply this to std
    means = means.sort_values()
    std  = std[means.index]
    models = means.index

    plt.figure(figsize=(12, 7))
    bars = plt.bar(models, means, yerr=std, capsize=5)

    temp = 0
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width() / 2.0, 
                 0.5, 
                 'Mean:\n' + str(means[temp].round(3)),
                 ha='center')
        
        plt.text(bar.get_x() + bar.get_width() / 2.0, 
                 0.4, 
                 'Std:\n' + str(std[temp].round(3)),
                 ha='center')
        temp += 1

    plt.xlabel('Model')
    plt.ylabel(metric_name)
    plt.title(metric_title)
    plt.xticks(rotation=45, ha='right')
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    plt.tight_layout()
    plt.show()

def ibsAndCIndexBar(ibs, c_index):
    models = ibs.index
    ibs = ibs.iloc[:, 0]
    c_index = c_index.iloc[:, 0]

    plt.figure(figsize=(12, 7))
    plt.subplot(2, 1, 1)
    ibs_bars = plt.bar(models, ibs)
    temp = 0
    for bar in ibs_bars:
        plt.text(bar.get_x() + bar.get_width() / 2.0, 
                 bar.get_height() - 0.04, 
                 'IBS:\n' + str(ibs[temp].round(3)),
                 ha='center')
        temp += 1

    plt.ylabel('IBS Score')
    plt.title('Integrated Brier Score Of Models (Lower is Better)')
    plt.yticks([0.0, 0.05, 0.1, 0.15, 0.2])
    plt.xticks(rotation=45, ha='right')

    plt.subplot(2, 1, 2)
    c_index_bars = plt.bar(models, c_index)
    temp = 0
    for bar in c_index_bars:
        plt.text(bar.get_x() + bar.get_width() / 2.0, 
                 bar.get_height() - 0.2, 
                 'C-Index:\n' + str(c_index[temp].round(3)),
                 ha='center')
        temp += 1

    plt.ylabel('C-Index Score')
    plt.title('C-Index Score Of Models (Higher is Better)')
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.show()

# line(accuracy_df, 'Point Accuracy Of Models', 'Accuracy')
# line(weighted_f1_df, 'Weighted F1 Score Of Models', 'Weighted F1')
# line(roc_auc_df, 'ROC-AUC Of Models', 'ROC-AUC')
# line(brier_score_df, 'Brier Score Of Models', 'Brier Score')

# meanAndSdBar(accuracy_df, 'Mean And Standard Deviation Point Accuracy Of Models', 'Mean Accuracy Probability')
# meanAndSdBar(accuracy_df, 'Mean And Standard Deviation Weighted F1 Score Of Models', 'Mean Weighted F1 Probability')
# meanAndSdBar(accuracy_df, 'Mean And Standard Deviation ROC-AUC Of Models', 'Mean ROC-AUC Probability')
# meanAndSdBar(accuracy_df, 'Mean And Standard Deviation Brier Score Of Models', 'Mean Brier Score Probability')

ibsAndCIndexBar(ibs_df, c_index_df)
