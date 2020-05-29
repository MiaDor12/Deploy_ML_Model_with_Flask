import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pydotplus
from IPython.display import display, Image, Javascript
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report

def get_qcut(df, feature, q):
    """
    | **Description:**
    | Discretize pandas series into equal-sized buckets based on rank or based on sample quantiles.
    | **Params:**
    * df (dataframe) - df that contains the column of a numeric feature and a column of the binary label.
    * feature (string) - the name of the column that contains a numeric feature.
    * q (int) - number of quantiles.
    """

    unique_values = df[feature].unique()
    num_unique_values = len(unique_values)
    if ((len(df) == 0) | (num_unique_values < 2)):
        return None

    if num_unique_values > 3:
        qCat = pd.qcut(df[feature], q=q, duplicates='drop')

    else:
        valToIntevalDict = {}
        sorted_unique_values = sorted(unique_values)
        for i in range(len(unique_values)):
            curr_val = sorted_unique_values[i]
            if i == 0:
                valToIntevalDict[curr_val] = pd.Interval(left=curr_val - 0.001, right=curr_val)
            else:
                last_val = sorted_unique_values[i - 1]
                valToIntevalDict[curr_val] = pd.Interval(left=last_val, right=curr_val)

        qCat = df[feature].apply(lambda val: valToIntevalDict.get(val)).astype('category')

    return qCat


def get_binsDf(df, CLASS_LABEL, qCat):
    """
    | **Description:**
    | This function returns a dataframe with the discrete data (bins) as it had splited by the given qCat.
    | **Params:**
    * df (dataframe)- pandas dataframe that contains the column of the label.
    * CLASS_LABEL (string) - the name of the column that contains the label.
    * qCat (series) - pandas series of of type "category", which contains the intervals of the bins.
    """

    binsDf = pd.DataFrame({'bins_right': qCat.apply(lambda x: x.right).astype(float),
                           'bins_left': qCat.apply(lambda x: x.left).astype(float),
                           CLASS_LABEL: df[CLASS_LABEL]})

    return binsDf


def get_agg_data_of_mean_label(df, feature, CLASS_LABEL, q):
    """
    | **Description:**
    | Returns pandas df with mean, sum and count of the label, per bins of the selected feature.
    | **Params:**
    * df (dataframe) - df that contains the column of the numeric feature and a column of the binary label.
    * feature (string) - the name of the column of the continuous feature.
    * CLASS_LABEL (string) - the name of the column to check lift of.
    * q (int) - number of quantiles to split.
    """

    qCat = get_qcut(df, feature, q)
    if qCat is None:
        return None

    binsDf = get_binsDf(df, CLASS_LABEL, qCat)
    aggData = binsDf.groupby('bins_right').agg({CLASS_LABEL: ['mean', 'sum'], 'bins_right': 'count'})
    aggData.columns = ['mean', 'sum', 'count']
    aggData.index.name = feature

    return aggData


def get_bar_plot_from_agg_data(x_column, y_column, title=None, xlabel=None, ylabel=None,
                               toShow=True, toSave=False, fileName=None):
    """
    | **Description:**
    | Produce the bar plot based on the given aggregated data.
    | **Params:**
    * x_column (series) - pandas series of the x axis for the bar plot.
    * y_column (series) - pandas series of the y axis for the bar plot.
    * title (string) - the title of the graph.
    * xlabel (string) - the label of the x axis of the graph.
    * ylabel (string) - the label of the y axis of the graph.
    * toShow (boolean) - indicates of whether or not showing the plot. Default True.
    * toSave (boolean) - indicates of whether or not to save the plot to a png file. Default False.
    * fileName (string) - the file name of the png file that will save if toSave=True. Default None.
    """

    sns.barplot(x_column, y_column)

    fig = plt.gcf()
    fig.set_size_inches(10, 5)
    plt.xticks(rotation=70)
    if not title:
        title = f'Label={y_column.name}, Feature={x_column.name}'
    plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)

    if toSave:
        if fileName is None:
            fileName = 'bar_plot.png'
        fileName = fileName.replace('>', '_')
        plt.savefig(fileName, pad_inches=0.2, bbox_inches='tight')

    if toShow:
        plt.show()

    plt.clf()


def get_bar_plot_of_mean_label(df, feature, CLASS_LABEL, q=10, title=None, xlabel=None, ylabel=None, toShow=True,
                               toDisplayData=False, toSave=False, fileName=None, toReturnData=False):
    """
    | **Description:**
    | This function produces a bar plot of the mean label per bin.
    | **Params:**
    * df (dataframe) - df that contains the column of the numeric feature and a column of the binary label.
    * feature (string) - name of the column of the continuous feature.
    * CLASS_LABEL (string) - the column to check lift of.
    * q (int) - number of quantiles to split.
    * title (string) - the title of the graph.
    * xlabel (string) - the label of the x axis of the graph.
    * ylabel (string) - the label of the y axis of the graph.
    * toShow (boolean) - indicates of whether or not showing the plot. Default True.
    * toDisplayData (boolean) - indicates of whether or not to display the aggData.
    * toSave (boolean) - indicates of whether or not to save the plot to a png file. Default False.
    * fileName (string) - the file name of the png file that will save if toSave=True. Default None.
    * toReturnData (boolean) - indicates of whether or not to return the aggData.
    """

    df = df.copy()[(~pd.isna(df[feature])) & (df[feature] != np.inf) & (~pd.isna(df[CLASS_LABEL]))]

    aggData = get_agg_data_of_mean_label(df, feature, CLASS_LABEL, q)

    if aggData is None:
        return None

    x_column = aggData.index
    y_column = aggData['mean']
    get_bar_plot_from_agg_data(x_column=x_column, y_column=y_column, title=title, xlabel=xlabel, ylabel=ylabel,
                               toShow=toShow, toSave=toSave, fileName=fileName)

    if toDisplayData:
        display(aggData)

    if toReturnData:
        return aggData