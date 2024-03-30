from nrclex import NRCLex
import pandas as pd
import plotly.graph_objects as go
import csv
from IPython.display import HTML, display
from plotly.subplots import make_subplots
import seaborn as sns
import datashader as ds
import colorcet as cc
import matplotlib.pyplot as plt
import sys
import matplotlib
import numpy as np
import numpy.random


Biden_tweets = pd.read_csv('biden_data.csv', encoding='utf-8')
Biden_tweets['emotionQuantity']=Biden_tweets['text'].apply(lambda x: NRCLex(x).raw_emotion_scores)
Biden_tweets['emotions'] = Biden_tweets['text'].apply(lambda x: NRCLex(x).affect_frequencies)

Mccarthy_tweets = pd.read_csv('mccarthy.csv', encoding='utf-8')
Mccarthy_tweets['emotionQuantity']=Mccarthy_tweets['text'].apply(lambda x: NRCLex(x).raw_emotion_scores)
Mccarthy_tweets['emotions'] = Mccarthy_tweets['text'].apply(lambda x: NRCLex(x).affect_frequencies)

print('\n')

UsersBiden=set()
for elem in Biden_tweets['user']:
    UsersBiden.add(elem)
setLengthUsersBiden=len(UsersBiden)
print('Biden number of users who replied out of 10000 replies',setLengthUsersBiden)

UsersMccarthy=set()
for elem in Mccarthy_tweets['user']:
    UsersMccarthy.add(elem)
setLengthUsersMccarthy=len(UsersMccarthy)
print('McCarthy number of users who replied out of 10000 replies', setLengthUsersMccarthy)
print('\n')



emotionQuantityDictBiden={'fear':0,'anger':0,'anticip':0,'trust':0,'surprise':0,'positive':0,'negative':0,'sadness':0,'disgust':0,'joy':0,'anticipation':0}
for elem in Biden_tweets['emotionQuantity']:
    for key in elem:
        emotionQuantityDictBiden[key]+=elem[key]
print("Quantity of each emotion Biden")
print(emotionQuantityDictBiden)

emotionQuantityDictMccarthy={'fear':0,'anger':0,'anticip':0,'trust':0,'surprise':0,'positive':0,'negative':0,'sadness':0,'disgust':0,'joy':0,'anticipation':0}
for elem in Mccarthy_tweets['emotionQuantity']:
    for key in elem:
        emotionQuantityDictMccarthy[key]+=elem[key]
print("Quantity of each emotion Mccarthy")
print(emotionQuantityDictMccarthy)
print('\n')

emotionMaxDictBiden={'fear':0,'anger':0,'anticip':0,'trust':0,'surprise':0,'positive':0,'negative':0,'sadness':0,'disgust':0,'joy':0,'anticipation':0}
emotionMinDictBiden={'fear':1,'anger':1,'anticip':1,'trust':1,'surprise':1,'positive':1,'negative':1,'sadness':1,'disgust':1,'joy':1,'anticipation':1}
emotionCountDictBiden={'fear':0,'anger':0,'anticip':0,'trust':0,'surprise':0,'positive':0,'negative':0,'sadness':0,'disgust':0,'joy':0,'anticipation':0}
emotionCountDictWhenMentionedBiden={'fear':0,'anger':0,'anticip':0,'trust':0,'surprise':0,'positive':0,'negative':0,'sadness':0,'disgust':0,'joy':0,'anticipation':0}
emotionWhenMentionedBiden={}

emotionMaxDictMccarthy={'fear':0,'anger':0,'anticip':0,'trust':0,'surprise':0,'positive':0,'negative':0,'sadness':0,'disgust':0,'joy':0,'anticipation':0}
emotionMinDictMccarthy={'fear':1,'anger':1,'anticip':1,'trust':1,'surprise':1,'positive':1,'negative':1,'sadness':1,'disgust':1,'joy':1,'anticipation':1}
emotionCountDictMccarthy={'fear':0,'anger':0,'anticip':0,'trust':0,'surprise':0,'positive':0,'negative':0,'sadness':0,'disgust':0,'joy':0,'anticipation':0}
emotionCountDictWhenMentionedMccarthy={'fear':0,'anger':0,'anticip':0,'trust':0,'surprise':0,'positive':0,'negative':0,'sadness':0,'disgust':0,'joy':0,'anticipation':0}
emotionWhenMentionedMccarthy={}


arrayFear=[]
arrayNegative=[]
arrayAnger=[]
arrayTrust=[]
arraySurprise=[]
arrayPositive=[]
arraySadness=[]
arrayDisgust=[]
arrayJoy=[]
arrayAnticipation=[]

for elem in Biden_tweets['emotions']:
    for key in elem:
        if key == 'fear':
            arrayFear.append(elem[key])
        if key == 'negative':
            arrayNegative.append(elem[key])
        if key == 'anger':
            arrayAnger.append(elem[key])
        if key == 'trust':
            arrayTrust.append(elem[key])
        if key == 'surprise':
            arraySurprise.append(elem[key])
        if key == 'positive':
            arrayPositive.append(elem[key])
        if key == 'sadness':
            arraySadness.append(elem[key])
        if key == 'disgust':
            arrayDisgust.append(elem[key])
        if key == 'joy':
            arrayJoy.append(elem[key])
        if key == 'anticipation':
            arrayAnticipation.append(elem[key])

        emotionCountDictBiden[key]+=elem[key]
        if elem[key]>emotionMaxDictBiden[key]:
            emotionMaxDictBiden[key]=elem[key]
        if elem[key]<emotionMinDictBiden[key]:
            emotionMinDictBiden[key]=elem[key]
        if elem[key]>0:
            emotionCountDictWhenMentionedBiden[key]+=elem[key]


for elem in Mccarthy_tweets['emotions']:
    for key in elem:
        emotionCountDictMccarthy[key]+=elem[key]
        if elem[key]>emotionMaxDictMccarthy[key]:
            emotionMaxDictMccarthy[key]=elem[key]
        if elem[key]<emotionMinDictMccarthy[key]:
            emotionMinDictMccarthy[key]=elem[key]
        if elem[key]>0:
            emotionCountDictWhenMentionedMccarthy[key]+=elem[key]

print('Average emotions Biden',list(map(lambda key: emotionCountDictBiden[key]/10000, emotionCountDictBiden.keys())))
print('Average emotions Mccarthy',list(map(lambda key: emotionCountDictMccarthy[key]/10000, emotionCountDictMccarthy.keys())))

print('\n')

print('Average emotions when detected Biden')
EmotionWDB={}
for elem in emotionCountDictWhenMentionedBiden:
    if(emotionQuantityDictBiden[elem])>0:
        EmotionWDB.update({elem: emotionCountDictWhenMentionedMccarthy[elem] / emotionQuantityDictMccarthy[elem]})
        print(elem, emotionCountDictWhenMentionedBiden[elem]/emotionQuantityDictBiden[elem])

print('\n')

print('Average emotions when detected Mccarthy')
EmotionWDM={}
for elem in emotionCountDictWhenMentionedMccarthy:
    if(emotionQuantityDictMccarthy[elem])>0:
        EmotionWDM.update({elem: emotionCountDictWhenMentionedMccarthy[elem]/emotionQuantityDictMccarthy[elem]})
        print(elem, emotionCountDictWhenMentionedMccarthy[elem]/emotionQuantityDictMccarthy[elem])


print('\n')

print("Maximum emotions Biden",emotionMaxDictBiden)
print("Maximum emotions Mccarthy",emotionMaxDictMccarthy)
print("Minimum emotions Biden",emotionMinDictBiden)
print("Minimum emotions Mccarthy",emotionMinDictMccarthy)

print('\n')
print('\n')

Biden_Keys_positive=['positive','trust','surprise','joy']
Biden_Values_positive = [EmotionWDB['positive'],EmotionWDB['trust'],EmotionWDB['surprise'],EmotionWDB['joy']]
Biden_Keys_negative = ['negat.','fear','anger','sad','disgust']
Biden_Values_negative = [EmotionWDB['negative'],EmotionWDB['fear'],EmotionWDB['anger'],EmotionWDB['sadness'],EmotionWDB['disgust']]

Mccarthy_Keys_positive=['positive','trust','surprise','joy']
Mccarthy_Values_positive = [EmotionWDM['positive'],EmotionWDM['trust'],EmotionWDM['surprise'],EmotionWDM['joy']]
Mccarthy_Keys_negative = ['negat.','fear','anger','sad','disgust']
Mccarthy_Values_negative = [EmotionWDM['negative'],EmotionWDM['fear'],EmotionWDM['anger'],EmotionWDM['sadness'],EmotionWDM['disgust']]

figure, axis = plt.subplots(2, 2)

# For Sine Function
axis[0, 0].bar(Biden_Keys_positive, Biden_Values_positive)
axis[0, 0].set_title("Biden positive emotions")

# For Cosine Function
axis[0, 1].bar(Biden_Keys_negative, Biden_Values_negative, color=['red'])
axis[0, 1].set_title("Biden negative emotions")

# For Tangent Function
axis[1, 0].bar(Mccarthy_Keys_positive, Mccarthy_Values_positive)
axis[1, 0].set_title("McCarthy positive emotions")

# For Tanh Function
axis[1, 1].bar(Mccarthy_Keys_negative, Mccarthy_Values_negative, color=['red'])
axis[1, 1].set_title("McCarthy negative emotions")
# Combine all the operations and display

plt.subplots_adjust(left=0.1,bottom=0.1,right=0.9, top=0.9,wspace=0.4, hspace=0.7)

plt.show()

#fig=(go.Figure(data=go.Scatter(x=arrayNegative,y=arrayFear,mode='markers',marker_color=arrayFear,text="Negative vs Fear")))
#fig=go.Figure(data=go.Scatter(x=arrayNegative,y=arrayJoy,mode='markers',marker_color=arrayJoy,text="Negative vs Joy"))
#fig=go.Figure(data=go.Scatter(x=arrayNegative,y=arraySadness,mode='markers',marker_color=arraySadness,text="Negative vs Sadness"))
#fig=go.Figure(data=go.Scatter(x=arrayNegative,y=arrayDisgust,mode='markers',marker_color=arrayDisgust,text="Negative vs Disgust")))
#fig=go.Figure(data=go.Scatter(x=arrayPositive,y=arrayTrust,mode='markers',marker_color=arrayTrust,text="Positive vs Trust"))
#fig=go.Figure(data=go.Scatter(x=arrayPositive,y=arraySurprise,mode='markers',marker_color=arraySurprise,text="Positive vs Surprise")))
#fig=go.Figure(data=go.Scatter(x=arrayPositive,y=arrayJoy,mode='markers',marker_color=arrayJoy,text="Positive vs Joy"))
#fig.update_layout(title='Positive vs Trust Biden')
#fig.show()

