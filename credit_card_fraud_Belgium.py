import os, matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from joblib import Parallel, delayed

from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit, learning_curve
from sklearn.metrics import accuracy_score, confusion_matrix, average_precision_score, precision_score, recall_score, plot_confusion_matrix

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from keras.layers import Input, Dense
from keras.models import Model, Sequential
from keras import regularizers


def plot_learning_curve(estimator, title, X, y, score_met='average_precision', ylim=None, cv=None, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 7)):

    fig, axes = plt.subplots(1, 1, dpi=dpi_setting)

    axes.set_title("Learning Curve for "+title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel(score_met)

    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, scoring=score_met, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, return_times=False)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot learning curve
    axes.grid()
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2, color="r")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2, color="g")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    axes.legend(loc="best")

    return plt


def data_preprocessing(data_loaded, features_drop_list=['Time'], ml_method='Supervised', fraction_sample=0.8, multiplier_factor_sample=1, unsupervised_train_onfraud=False, remove_minmax_outliers=False, do_over_sampling=False):

    data_csv = data_loaded
    data_csv = data_csv.drop(features_drop_list,axis=1)
    features_list = list(data_csv.columns)

    if remove_minmax_outliers in [True]:
        data_csv_fraud = data_csv[data_csv['Class'] == 1]
        min_fraud = data_csv_fraud[features_list].min(axis=0)
        max_fraud = data_csv_fraud[features_list].max(axis=0)
        for onefeature in features_list:
            if onefeature in ['Class']:
                continue
            else:
                data_csv = data_csv[data_csv[onefeature] >= min_fraud[onefeature]]
                data_csv = data_csv[data_csv[onefeature] <= max_fraud[onefeature]]

    fraud = data_csv[data_csv['Class'] == 1]
    normal = data_csv[data_csv['Class'] == 0]
    fraud_sampled = fraud.sample(frac=fraction_sample)
    normal_sampled = normal.sample(int(multiplier_factor_sample*fraud_sampled.shape[0]))
    data_csv = data_csv.drop(fraud_sampled.index)
    data_csv = data_csv.drop(normal_sampled.index)
    data_csv = data_csv.reset_index(drop=True)

    if do_over_sampling in [True]:
        smote_over_sample = BorderlineSMOTE(sampling_strategy='minority', random_state=42, k_neighbors=5, n_jobs=-1, m_neighbors=10, kind='borderline-1')
        train_concat = pd.concat([normal_sampled,fraud_sampled],axis=0)
        train_classes = train_concat['Class']
        train_concat = train_concat.drop(['Class'],axis=1)
        X_resampled, y_resampled = smote_over_sample.fit_resample(train_concat, train_classes)
        train_concat = pd.concat([X_resampled, y_resampled],axis=1)
        fraud_sampled = train_concat[train_concat['Class'] == 1]
        normal_sampled = train_concat[train_concat['Class'] == 0]

    if ml_method in ['Supervised']:

        X_train = pd.concat([normal_sampled,fraud_sampled],axis=0)
        y_train = X_train['Class']
        X_train = X_train.drop(['Class'],axis=1)
        y_valid = data_csv['Class']
        data_csv = data_csv.drop(['Class'],axis=1)
        X_valid = data_csv

    elif ml_method in ['Unsupervised']:

        if unsupervised_train_onfraud in [False]: #Training unsupervised model on normal transactions samples
            X_train = normal_sampled
            y_train = X_train['Class']
            X_train = X_train.drop(['Class'],axis=1)
            X_valid = pd.concat([data_csv, fraud_sampled], axis=0)
            y_valid = X_valid['Class']
            X_valid = X_valid.drop(['Class'],axis=1)

        elif unsupervised_train_onfraud in [True]: #Training unsupervised model on fraudulent transactions samples
            X_train = fraud_sampled
            y_train = X_train['Class']
            X_train = X_train.drop(['Class'],axis=1)
            X_valid = pd.concat([data_csv, normal_sampled], axis=0)
            y_valid = X_valid['Class']
            X_valid = X_valid.drop(['Class'],axis=1)

    elif ml_method in ['Semisupervised']:

        if unsupervised_train_onfraud in [False]: #Training unsupervised model on normal transactions samples
            X_train = normal #normal_sampled
            y_train = X_train['Class']
            X_train = X_train.drop(['Class'],axis=1)
            X_valid = fraud_sampled #pd.concat([data_csv, fraud_sampled], axis=0)
            y_valid = X_valid['Class']
            X_valid = X_valid.drop(['Class'],axis=1)

        elif unsupervised_train_onfraud in [True]: #Training unsupervised model on fraudulent transactions samples
            X_train = fraud_sampled
            y_train = X_train['Class']
            X_train = X_train.drop(['Class'],axis=1)
            X_valid = normal #normal_sampled #pd.concat([data_csv, normal_sampled], axis=0)
            y_valid = X_valid['Class']
            X_valid = X_valid.drop(['Class'],axis=1)


    # Standard scaling feature 'Amount'
    features_numerical = 'Amount'
    if features_numerical in features_list:
        stdscl = StandardScaler()
        one = np.asarray(X_train[features_numerical])
        two = np.asarray(X_valid[features_numerical])
        X_train_num_ss = stdscl.fit_transform(one.reshape(-1,1))
        X_valid_num_ss = stdscl.transform(two.reshape(-1,1))
        X_train = X_train.drop(features_numerical,axis=1)
        X_valid = X_valid.drop(features_numerical,axis=1)
        X_train[features_numerical] = X_train_num_ss
        X_valid[features_numerical] = X_valid_num_ss

    return X_train, X_valid, y_train, y_valid


def autoencoder_supervised_classifier(X_normal, X_fraud):

    ## input layer
    input_layer = Input(shape=(X_normal.shape[1],))

    ## encoding part
    encoded = Dense(200, activation='tanh', activity_regularizer=regularizers.l1(10e-5))(input_layer)
    encoded = Dense(50, activation='relu')(encoded)
    #encoded = Dense(25, activation='relu')(encoded)

    ## decoding part
    #decoded = Dense(25, activation='tanh')(encoded)
    decoded = Dense(50, activation='tanh')(encoded)
    decoded = Dense(200, activation='tanh')(decoded)

    ## output layer
    output_layer = Dense(X_normal.shape[1], activation='relu')(decoded)

    autoencoder = Model(input_layer, output_layer)
    autoencoder.compile(optimizer="adadelta", loss="mse")
    autoencoder.fit(X_normal, X_normal, batch_size = 256, epochs = 10, shuffle = True, validation_split = 0.20, verbose=0)

    hidden_representation = Sequential()
    hidden_representation.add(autoencoder.layers[0])
    hidden_representation.add(autoencoder.layers[1])
    hidden_representation.add(autoencoder.layers[2])
    norm_hid_rep = hidden_representation.predict(X_normal)
    fraud_hid_rep = hidden_representation.predict(X_fraud)

    X_represent_transformed = np.append(norm_hid_rep, fraud_hid_rep, axis = 0)
    y_normal = np.zeros(norm_hid_rep.shape[0])
    y_frauds = np.ones(fraud_hid_rep.shape[0])
    y_represent_transformed = np.append(y_normal, y_frauds)

    return X_represent_transformed, y_represent_transformed


def scoring_metrics_calculation(ytrn_true, yvld_true, ytrn_pred, yvld_pred, ytrn_pred_proba, yvld_pred_proba):

    accur = 0#accuracy_score(ytrn_true,ytrn_pred)
    precs = 0#precision_score(ytrn_true, ytrn_pred, average='weighted')
    recal = 0#recall_score(ytrn_true, ytrn_pred, average='weighted')
    auprc = 0#average_precision_score(ytrn_true,ytrn_pred_proba)
    conmat = 0#confusion_matrix(ytrn_true, ytrn_pred, normalize='all')
    tn, fp, fn, tp = 0,0,0,0#conmat.ravel()

    score_dict_train = dict(accur=accur, precs=precs, recal=recal, auprc=auprc, tn=tn, fp=fp, fn=fn, tp=tp)

    accur = accuracy_score(yvld_true,yvld_pred)
    precs = precision_score(yvld_true, yvld_pred, average='weighted')
    recal = recall_score(yvld_true, yvld_pred, average='weighted')
    auprc = average_precision_score(yvld_true,yvld_pred_proba)
    conmat = confusion_matrix(yvld_true, yvld_pred, normalize='true')
    tn, fp, fn, tp = conmat.ravel()

    score_dict_valid = dict(accur=accur, precs=precs, recal=recal, auprc=auprc, tn=tn, fp=fp, fn=fn, tp=tp)

    return score_dict_train, score_dict_valid


def machine_learning_model(estimator, xtrn, xvld, ytrn, yvld, ml_method='Supervised', n_splits=3, steps=1, score_met='average_precision', param_grid=None, do_gscv=False, do_recursive_elimination=False, plot_learn_curve=False, plot_con_matrix=False):

    cvld = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, train_size=None, random_state=42)

    if do_recursive_elimination in [True]:
        # estimator.fit(xtrn, ytrn)
        print("\nRecursive Feature Eliminiation.....")
        rfecv = RFECV(estimator, step=steps, min_features_to_select=1, cv=cvld, scoring=score_met, verbose=0, n_jobs=-1)
        try:
            X_train_new = rfecv.fit_transform(xtrn, ytrn)
        except RuntimeError:
            logreg_rf = LogisticRegression(C=1.0, penalty='l2', dual=False, tol=0.0001, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=42, solver='lbfgs', max_iter=1000, multi_class='auto', verbose=0, warm_start=False, n_jobs=-1, l1_ratio=None)
            rfecv = RFECV(logreg_rf, step=steps, min_features_to_select=1, cv=cvld, scoring=score_met, verbose=0, n_jobs=-1)
            X_train_new = rfecv.fit_transform(xtrn, ytrn)
        print("Optimal features: %d" % rfecv.n_features_)
        X_valid_new = rfecv.transform(xvld)
        xtrn = X_train_new
        xvld = X_valid_new

    if do_gscv in [True]:
        print("\nGrid Search CV.....")
        print(param_grid)
        gscv = GridSearchCV(estimator, param_grid=param_grid, scoring=score_met, n_jobs=-1, refit=True, cv=cvld, verbose=0, pre_dispatch='2*n_jobs', return_train_score=False)
        gscv.fit(xtrn, ytrn)
        ytrn_pred = gscv.predict(xtrn)
        yvld_pred = gscv.predict(xvld)
        ytrn_pred_proba_both = gscv.predict_proba(xtrn)
        ytrn_pred_proba = ytrn_pred_proba_both[:,1]
        yvld_pred_proba_both = gscv.predict_proba(xvld)
        yvld_pred_proba = yvld_pred_proba_both[:,1]
        print(gscv.best_estimator_)
        print(gscv.best_score_)
        print(gscv.best_params_)
        estimator = gscv.best_estimator_

    if ml_method in ['Supervised']:
        estimator.fit(xtrn, ytrn)
        ytrn_pred = estimator.predict(xtrn)
        yvld_pred = estimator.predict(xvld)
        ytrn_pred_proba_both = estimator.predict_proba(xtrn)
        ytrn_pred_proba = ytrn_pred_proba_both[:,1]
        yvld_pred_proba_both = estimator.predict_proba(xvld)
        yvld_pred_proba = yvld_pred_proba_both[:,1]

    elif ml_method in ['Unsupervised']:
        estimator.fit(xtrn)
        ytrn_pred = estimator.predict(xtrn)
        yvld_pred = estimator.predict(xvld)
        ytrn_pred = [1 if l == -1 else 0 for l in ytrn_pred]
        yvld_pred = [1 if l == -1 else 0 for l in yvld_pred]
        ytrn_pred_proba = ytrn_pred
        yvld_pred_proba = yvld_pred

    elif ml_method in ['Semisupervised']:
        X_represent_transformed, y_represent_transformed = autoencoder_supervised_classifier(xtrn, xvld)
        X_train, X_valid, y_train, y_valid = train_test_split(X_represent_transformed, y_represent_transformed, test_size=0.25)
        estimator.fit(X_train, y_train)
        ytrn_pred = estimator.predict(X_train)
        yvld_pred = estimator.predict(X_valid)
        ytrn_pred_proba_both = estimator.predict_proba(X_train)
        ytrn_pred_proba = ytrn_pred_proba_both[:,1]
        yvld_pred_proba_both = estimator.predict_proba(X_valid)
        yvld_pred_proba = yvld_pred_proba_both[:,1]
        ytrn = y_train
        yvld = y_valid

    score_dict_train, score_dict_valid = scoring_metrics_calculation(ytrn, yvld, ytrn_pred, yvld_pred, ytrn_pred_proba, yvld_pred_proba)

    if plot_learn_curve in [True]:
        print("\nPlotting Learning Curve.....")
        ### Learning Curves
        title = str(estimator)
        plot_learning_curve(estimator, title, xtrn, ytrn, ylim=(0.0, 1.1), cv=cvld, n_jobs=-1)
        plt.show()

    if plot_con_matrix in [True]:
        print("Plotting Confusion Matrix.....")
        ### Confusion Matrix
        plot_confusion_matrix(estimator, xvld, yvld, labels=None, sample_weight=None, normalize='true', display_labels=None, include_values=True, xticks_rotation='horizontal', values_format=None, cmap='viridis', ax=None)
        plt.show()

    return score_dict_train, score_dict_valid


dpi_setting=1200
plt.rcParams.update({'font.size': 7})
plt.rcParams['figure.dpi'] = 120
plot_linewidth = 1.5
startTime= datetime.now()

data_path = '/media/ashutosh/Computer Vision/Predictive_Maintenance/Bank_Loan_data_Kaggle'
# data_path = 'E:\Predictive_Maintenance\Bank_Loan_data_Kaggle'
data_csv = pd.read_csv(data_path+"//creditcard.csv")
no_features = ['V5','V6','V8','V13','V15','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28']
maybe_relevant_features = ['V7','V1','V2','V3','V16','V18']

Log_Reg = LogisticRegression(C=1.0, penalty='l2', dual=False, tol=0.0001, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=42, solver='lbfgs', max_iter=1000, multi_class='auto', verbose=0, warm_start=False, n_jobs=-1, l1_ratio=None)
SVC_Linear = SVC(C=1.0, kernel='linear', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovo', break_ties=False, random_state=42)
SVC_RBF = SVC(C=100.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovo', break_ties=False, random_state=42)
Decision_Tree = DecisionTreeClassifier(criterion='entropy', splitter='random', max_depth=16, min_samples_split=26, min_samples_leaf=2, max_features=None, random_state=42, class_weight=None)
Random_Forest = RandomForestClassifier(n_estimators=90, criterion='entropy', max_depth=10, min_samples_split=8, min_samples_leaf=2, max_features='sqrt', bootstrap=True, oob_score=False, class_weight=None, n_jobs=-1, random_state=42)
XGBoost = XGBClassifier(n_estimators=100, max_depth=5, min_child_weight=2, max_delta_step=8, learning_rate=0.1, gamma=0.1, objective='binary:logistic', scale_pos_weight=1, base_score=0.85, missing=None, n_jobs=-1, nthread=-1, random_state=42, seed=42, silent=True, subsample=1, verbosity=0)
LightGBM = LGBMClassifier(n_estimators=115, num_leaves=65, max_depth=15, min_child_samples=40, learning_rate=0.1, boosting_type='gbdt', objective='binary', random_state=42, n_jobs=- 1, silent=True)
Naive_Bayes = GaussianNB(var_smoothing=1e0)
One_Class_SVM = OneClassSVM(kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, nu=0.05, shrinking=True, cache_size=200, verbose=False, max_iter=-1)
Isolation_Forest = IsolationForest(n_estimators=100, max_samples='auto', contamination='auto', max_features=1.0, bootstrap=True, n_jobs=-1, random_state=42)
Auto_Enc_LogReg = Log_Reg
Auto_Enc_LightGBM = LightGBM


score_met = 'average_precision'
fraction_sample = 0.2
n_splits = 5
steps = 1
unsupervised_train_onfraud = False
do_gscv = False
do_over_sampling = True
do_recursive_elimination = False
plot_learn_curve = False
plot_con_matrix = False
param_grid = None


remove_features = ['Time'] + no_features #+ maybe_relevant_features ########### add no_features
remove_minmax_outliers = True ########### True


print("\n.......................................................     Start     .......................................................\n")


classifiers_text_list = ['Log_Reg', 'SVC_Linear', 'SVC_RBF', 'Decision_Tree', 'Random_Forest', 'XGBoost', 'LightGBM', 'Naive_Bayes', 'One_Class_SVM', 'Isolation_Forest', 'Auto_Enc_LogReg', 'Auto_Enc_LightGBM']
ml_methods_list = ['Supervised', 'Supervised', 'Supervised', 'Supervised', 'Supervised', 'Supervised', 'Supervised', 'Supervised', 'Unsupervised', 'Unsupervised', 'Semisupervised', 'Semisupervised']
classifiers_list = [Log_Reg, SVC_Linear, SVC_RBF, Decision_Tree, Random_Forest, XGBoost, LightGBM, Naive_Bayes, One_Class_SVM, Isolation_Forest, Auto_Enc_LogReg, Auto_Enc_LightGBM]
metrics_list = ['accur', 'precs', 'recal', 'auprc', 'tn', 'fp', 'fn', 'tp']
seed_vals_list = [13, 77, 42, 639, 41]
runs_list = ['run'+str(x) for x in seed_vals_list]

pd_df_columns = pd.MultiIndex.from_product([classifiers_text_list, metrics_list, runs_list], names=['classifiers', 'metrics', 'runs'])
zeros_init = np.zeros(pd_df_columns.shape[0])
all_clfs_valid_scores = pd.DataFrame(0.0, index=[0], columns=pd_df_columns)


for one_clf, one_clf_text, one_ml_method in zip(classifiers_list, classifiers_text_list, ml_methods_list):
    print(one_clf_text)
    if one_ml_method in ['Semisupervised']:
        multiplier_factor_sample = 250
    else:
        multiplier_factor_sample = 20

    for one_seed, one_run in zip(seed_vals_list, runs_list):

        np.random.seed(one_seed)
        ml_method = one_ml_method

        X_train, X_valid, y_train, y_valid = data_preprocessing(data_csv, features_drop_list=remove_features, ml_method=ml_method, unsupervised_train_onfraud=unsupervised_train_onfraud, remove_minmax_outliers=remove_minmax_outliers, fraction_sample=fraction_sample, multiplier_factor_sample=multiplier_factor_sample, do_over_sampling=do_over_sampling)

        score_dict_train, score_dict_valid = machine_learning_model(one_clf, X_train, X_valid, y_train, y_valid, ml_method=ml_method, n_splits=n_splits, steps=steps, score_met=score_met, param_grid=param_grid, do_gscv=do_gscv, do_recursive_elimination=do_recursive_elimination, plot_learn_curve=plot_learn_curve, plot_con_matrix=plot_con_matrix)

        for one_metric in metrics_list:
            all_clfs_valid_scores[one_clf_text, one_metric, one_run][0] = score_dict_valid[one_metric]


for one_metric in metrics_list:
    print(one_metric)
    means_list = []
    stds_list = []
    for one_clf_text in classifiers_text_list:
        mean_calc = all_clfs_valid_scores[one_clf_text, one_metric].mean(axis=1)
        means_list.append(100*mean_calc[0])
        std_calc = all_clfs_valid_scores[one_clf_text, one_metric].std(axis=1)
        stds_list.append(100*std_calc[0])

    plt.figure(num=None, figsize=None, dpi=dpi_setting, facecolor='w', edgecolor='w')
    plt.title(one_metric+' [2]')
    plt.xlim()
    plt.ylim(0,100)
    plt.xlabel('Estimators')
    plt.ylabel(one_metric)
    plt.bar(x=np.arange(len(classifiers_text_list)), height=means_list, width=0.3, yerr=stds_list)
    plt.xticks(ticks=np.arange(len(classifiers_text_list)), labels=classifiers_text_list, rotation=90)
    plt.grid(b=True, which='major', axis='both', linestyle=':', linewidth=0.5, alpha=1)
    plt.savefig(data_path+"//{txt}_removed_outliers_allfeatures.png".format(txt=one_metric), dpi=dpi_setting, facecolor='w', edgecolor='w', orientation='portrait', papertype=None, format='png', transparent=False, bbox_inches='tight', pad_inches=0.1, metadata=None)
    # plt.show()
    plt.close()


print("\n.......................................................      Done     .......................................................\n")
timeElapsed = datetime.now() - startTime
print('Time elpased (hh:mm:ss.ms) {}'.format(timeElapsed))

# os.system('spd-say "your program has finished, please check the output now"')

