import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob
from nrclex import NRCLex

def get_polarity(review):
    return TextBlob(review).sentiment.polarity

def get_subjectivity(review):
    return TextBlob(review).sentiment.subjectivity

def plot_table(list, Candidate, Sentiment):
    i=0
    print(" The Most "+Sentiment+" replies to "+Candidate+" and their adjacent polarity scores")
    for x in list:
        print(list[i])
        i+=1
    print("-----------------------------------------------------------------------------------------------------------")
    print("\n")


# Reading both the csv Files
Mccarthy_tweets = pd.read_csv('mccarthy.csv', encoding='utf-8')
Biden_tweets = pd.read_csv('biden_data.csv', encoding='utf-8')

Mccarthy_tweets['Sentiment'] = Mccarthy_tweets['text'].apply(get_polarity)
Mccarthy_tweets['Subjectivity'] = Mccarthy_tweets['text'].apply(get_subjectivity)
Mccarthy_tweets['Expression_type'] = np.where(Mccarthy_tweets['Sentiment'] > 0, 'positive', 'negative')
Mccarthy_tweets['Expression_type'][Mccarthy_tweets.Sentiment == 0] = "Neutral"
Biden_tweets['Sentiment'] = Biden_tweets['text'].apply(get_polarity)
Biden_tweets['Subjectivity'] = Biden_tweets['text'].apply(get_subjectivity)
Biden_tweets['Expression_type'] = np.where(Biden_tweets['Sentiment'] > 0, 'positive', 'negative')
Biden_tweets['Expression_type'][Biden_tweets.Sentiment == 0] = "Neutral"


Mccarthy_Percentage=Mccarthy_tweets.groupby('Expression_type').count()['Sentiment']
biden_Percentage=Biden_tweets.groupby('Expression_type').count()['Sentiment']

x = list(Mccarthy_Percentage)
y = ['Neutral', 'Negative', 'Positive']
df = pd.DataFrame([(x[0],y[0]), (x[1],y[1]), (x[2],y[2])], columns=['x', 'y'])
fig = plt.figure(figsize =(10, 3))
plt.bar(df['y'],df['x'])
plt.xlabel("Expression_type")
plt.ylabel("Number of replies to McCarthy")
plt.title("McCarthy Replies Neutral/Positive/Negative Sentiment")
plt.show()


x = list(biden_Percentage)
y = ['Neutral', 'Negative', 'Positive']
df = pd.DataFrame([(x[0],y[0]), (x[1],y[1]), (x[2],y[2])], columns=['x', 'y'])
fig = plt.figure(figsize =(10, 3))
plt.bar(df['y'],df['x'])
plt.xlabel("Expression_type")
plt.ylabel("Number of replies to Biden")
plt.title("Biden Replies Neutral/Positive/Negative Sentiment")
plt.show()


Mccarthy_positive = Mccarthy_tweets[Mccarthy_tweets.Sentiment == 1].text.head()
pos_txt1 = list(Mccarthy_positive)
pos1 = Mccarthy_tweets[Mccarthy_tweets.Sentiment == 1].Sentiment.head()
pos_pol1 = list(pos1)
zip_Mccarthy_positive=list(zip(pos_pol1,pos_txt1))
plot_table(zip_Mccarthy_positive, " Kevin McCarthy", "Positive")

Mccarthy_negative = Mccarthy_tweets[Mccarthy_tweets.Sentiment == -1].text.head()
neg_txt1 = list(Mccarthy_negative)
neg1 = Mccarthy_tweets[Mccarthy_tweets.Sentiment == -1].Sentiment.head()
neg_pol1 = list(neg1)
zip_Mccarthy_negative=list(zip(neg_pol1,neg_txt1))
plot_table(zip_Mccarthy_negative, " Kevin McCarthy", "Negative")

Biden_positive = Biden_tweets[Biden_tweets.Sentiment == 1].text.tail()
pos_txt2 = list(Biden_positive)
pos2 = Biden_tweets[Biden_tweets.Sentiment == 1].Sentiment.tail()
pos_pol2 = list(pos2)
zip_biden_positive=list(zip(pos_pol2,pos_txt2))
plot_table(zip_biden_positive, " Joe Biden", "Positive")

Biden_negative = Biden_tweets[Biden_tweets.Sentiment == -1].text.head()
neg_txt2 = list(Biden_negative)
neg2 = Biden_tweets[Biden_tweets.Sentiment == -1].Sentiment.head()
neg_pol2 = list(neg2)
zip_biden_negative=list(zip(neg_pol2,neg_txt2))
plot_table(zip_biden_negative, " Joe Biden", "Negative")

colors_array = ['#ff9999','#66b3ff']

fig = plt.figure(figsize=(6,3),dpi=288)
ax1 = fig.add_subplot(121)
ax1.pie([(Mccarthy_tweets.groupby('Expression_type').count()['Sentiment'][1] / 1000) * 100, (Biden_tweets.groupby('Expression_type').count()['Sentiment'][1] / 1000) * 100], colors= colors_array, labels=['Negative McCarthy', 'Negative Biden'], autopct ='%1.1f%%', startangle=180)

ax2 = fig.add_subplot(122)
ax2.pie([(Mccarthy_tweets.groupby('Expression_type').count()['Sentiment'][2] / 1000) * 100, (Biden_tweets.groupby('Expression_type').count()['Sentiment'][2] / 1000) * 100], colors= colors_array, labels=['Positive McCarthy', 'Positive Biden'], autopct ='%1.1f%%', startangle=90)
plt.subplots_adjust(left=0.1,bottom=0.1,right=0.9,top=0.9,wspace=0.4,hspace=0.4)
plt.show()
