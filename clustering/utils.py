import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


columns = ["pubId", "is_hourly", "seqId", "on_homepage", "canonicalUrl",
                   "firstScrape", "lang_iso", "lang_reliability", "title", "text"]

def read_article_df(file):
    articles_dt = file.read().split('\n')[:-1]
    pubId, canonicalUrl,firstScrape,title,text,lang_reliability = [],[],[],[],[],[]
    lang_iso = []
    for article in articles_dt:    
        row = article.split('\t')
        pubId.append(row[0])
        canonicalUrl.append(row[4])
        firstScrape.append(row[5])
        lang_iso.append(row[6])
        lang_reliability.append(row[7])
        title.append(row[8])
        text.append(row[9])

    articles_df = pd.DataFrame()
    articles_df['pubId'] = pubId
    articles_df['canonicalUrl'] = canonicalUrl
    articles_df['firstScrape'] = firstScrape
    articles_df['title'] = title
    articles_df['text'] = text
    articles_df['lang_reliability'] = lang_reliability
    articles_df['lang_iso'] = lang_iso
    return articles_df

def load_label(label_path, file='lower_bound'):
    label1 = pd.read_csv(f'{label_path}/cave_rescue/{file}.txt', header=None)
    label1.columns = ['canonicalUrl']
    label1['label'] = 'cave_rescue'

    label2 = pd.read_csv(f'{label_path}/duckboat/{file}.txt', header=None)
    label2.columns = ['canonicalUrl']
    label2['label'] = 'duckboat'

    label3 = pd.read_csv(f'{label_path}/helsinki_summit/{file}.txt', header=None)
    label3.columns = ['canonicalUrl']
    label3['label'] = 'helsinki'

    label_df = pd.concat([label1, label2, label3])
    return label_df

def plot_cluster_sizes(cluster):
    cluster_sizes = []
    for i, c in cluster.items():
        cluster_sizes.append(len(c))
    plt.title(f'Num cluster = {len(cluster.keys())}')
    plt.hist(cluster_sizes)
    plt.show()
    
def merge_small_clusters(cluster: dict, threshold) -> dict:
    pc = {}
    pc[-1] = []
    for key, values in clusters.items():
        if len(values) > threshold:
            pc[key] = values
            
        else:
            pc[-1] += values
    return pc


def load_entities(file_path='../data/embedding/entites.txt'):
    entities = []
    with open(file_path, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            valid_json_line = line.replace(r"'", '"')
            entities.append(json.loads(valid_json_line))
            line = f.readline()
    return np.array(entities)\



def get_largest_cluster(y_pred):
    cluster_size = {}
    cluster_labels = set(y_pred)
    max_cluster_label = -1
    max_cluster_size = -1
    for cluster_label in cluster_labels:
        count = np.sum(y_pred == cluster_label)
        if count > max_cluster_size:
            max_cluster_label = cluster_label
            max_cluster_size = count
    return max_cluster_label, max_cluster_size

def max_cluster_metric(y_true, y_pred, name=None):
    labels = set(y_true)
    result = []
    for label in labels:
        if not isinstance(label, str): continue # remove NaN
        label_y_true = (y_true == label).astype(int)
        
        # get cluster with the largest population of label
        max_cluster_label, max_cluster_size = get_largest_cluster(y_pred.iloc[label_y_true])
        label_y_pred = (y_pred == max_cluster_label).astype(int)
        precision, recall, f_score, support = precision_recall_fscore_support(
            label_y_true, label_y_pred, average=None
        )
        
        scores = {
            'label': label,
            'precision': precision[1],
            'recall': recall[1],
            'f_score': f_score[1]
        }
        if name: scores['name'] = name
        result.append(scores)
        
    return result
        
def print_result_df(result_df):
    metrics = ['precision', 'recall', 'f_score']
    for metric in metrics:
        print(metric)
        print('-'*50)
        precision_df = result_df.pivot('label', 'name', metric)
        mean_series = precision_df.mean()
        mean_series.name  = 'mean'
        precision_df = precision_df.append(mean_series)
        print(precision_df)
        print('\n')
        
        
def merge_small_clusters(df, cluster_col, threshold, merge_cluster_id=-1):
    count_df = df.groupby(cluster_col).count()
    under_threshold_clusters = count_df[count_df['canonicalUrl'] < threshold].index
    small_cluster_rows = df[cluster_col].isin(under_threshold_clusters)
    df.loc[small_cluster_rows, cluster_col] = merge_cluster_id