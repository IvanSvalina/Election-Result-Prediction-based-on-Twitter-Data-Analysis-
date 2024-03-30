import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from bertopic import BERTopic

# nltk.download('stopwords')
# nltk.download('punkt')
stop_words = stopwords.words('english')


def clean_text(x):
    x = str(x)
    x = x.lower()
    x = re.sub(r'#[A-Za-z0-9]*', ' ', x)
    x = re.sub(r'https*://.*', ' ', x)
    x = re.sub(r'@[A-Za-z0-9]+', ' ', x)
    tokens = word_tokenize(x)
    x = ' '.join([w for w in tokens if not w.lower() in stop_words])
    x = re.sub(r'[%s]' % re.escape(
        '!"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~“…”’'), ' ', x)
    x = re.sub(r'\d+', ' ', x)
    x = re.sub(r'\n+', ' ', x)
    x = re.sub(r'\s{2,}', ' ', x)
    return x


# Change mccarthy.csv to biden.csv
df = pd.read_csv('biden_data.csv', encoding='utf-8')
df = df.dropna()
df['clean_text'] = df.text.apply(clean_text)
#df = df[df["clean_text"].str.contains("") == False]
print(df)
df['clean_text'].replace('', np.nan, inplace=True)
df.dropna(subset=['clean_text'], inplace=True)
df['clean_text'].replace(' ', np.nan, inplace=True)
df.dropna(subset=['clean_text'], inplace=True)
tweets = df.clean_text.to_list()
#timestamp = df.date.to_list()
topic_model = BERTopic(language="english")
topics, probs = topic_model.fit_transform(tweets)
print(topic_model.get_topic_info())
fig = topic_model.visualize_barchart()
fig.write_html(
    "D:\FER\Diplomski\sreci_semestar\Obrada prirodnog jezika\program\chart3.html")
fig = topic_model.visualize_topics()
fig.write_html(
    "D:\FER\Diplomski\sreci_semestar\Obrada prirodnog jezika\program\intertopic_distance3.html")
fig = topic_model.visualize_heatmap()
fig.write_html(
    "D:\FER\Diplomski\sreci_semestar\Obrada prirodnog jezika\program\similarity_matrix3.html")
