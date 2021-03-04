import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import classification_report
from xgboost import *
import itertools

def get_model(k, model_path):
    if k == 'xgb':
        xgb_model = XGBClassifier()
        xgb_model.load_model(model_path)
        model = xgb_model
    else: 
        model = joblib.load(model_path)
    return model 


def make_cmx(y, predicted_classes, classes):
    cmx = metrics.confusion_matrix(y, predicted_classes) # real vs predicted
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(cmx, classes=classes, normalize=False)
    
    
def make_classif_report(y, predicted_classes, classes):    
    pred_classes_prop = pd.Series(predicted_classes).value_counts(normalize=True).sort_index()
    
    cls_report = classification_report(y, predicted_classes, labels=classes, target_names=classes, output_dict=True)
    df_cls_report = pd.DataFrame(cls_report).transpose()
    df_cls_report = round_df(df_cls_report)
    return df_cls_report
    
    aux = df_cls_report.iloc[[c-1 for c in classes]] # esta tomando el indice implicito numérico, sin prestar atencion el str real

    dict_comp_prop_clase = {}

    for index, row in aux.iterrows():
        try:
            predicted_prop = pred_classes_prop[int(index)]
        except:
            predicted_prop = 0
            
        prop = class_prop[int(index)]        
        precision = row['precision']    
        recall = row['recall']

        dict_comp_prop_clase[index] = {
            'precision': precision,             
            'recall': recall, 
            'class_prop': prop, 
            'class_prop_predicted': predicted_prop,
            'precision / class_prop': precision/prop,
            'recall / class_prop': recall/prop,
        }
    df_comp = pd.DataFrame(dict_comp_prop_clase).transpose()   
    df_comp = round_df(df_comp)
    
    return df_cls_report, df_comp


def round_df(df, cols=None, r=2):
    cols = df.columns if cols is None else cols
    for col in cols:
        df[col] = df[col].round(r)
    return df
        
        
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
#    return df


def print_pred_classes_prop(pred_classes):        
    print('Predicted classes proportions')
    print('--------------')
    print(pd.Series(pred_classes).value_counts(normalize=True).sort_index(), '\n')
