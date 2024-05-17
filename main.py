import nltk
import streamlit as st
import glob
import plotly.express as px
import re
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
st.title("Diary Tone")

myfiles = sorted(glob.glob("files/*.txt"))
diary = []
pos = []
neg = []


def extract_date(filename):
    return re.search(r'\d{4}-\d{2}-\d{2}', filename).group()


for filepath in myfiles:
    with open(filepath, 'r') as file:
        entry = file.read()
        diary.append(entry)
    date = extract_date(filepath)
    analyzer = SentimentIntensityAnalyzer()
    mood = analyzer.polarity_scores(entry)
    pos.append({"Date": date, "Positivity": mood["pos"]})
    neg.append({"Date": date, "Negativity": mood["neg"]})

df_pos = pd.DataFrame(pos)
df_neg = pd.DataFrame(neg)

# st.write(df_pos, df_neg)

fig_pos = px.line(df_pos, x="Date", y="Positivity")
fig_neg = px.line(df_neg, x="Date", y="Negativity")

st.subheader('Positivity')
st.plotly_chart(fig_pos)
st.subheader('Negativity')
st.plotly_chart(fig_neg)

# st.write(diary, pos, neg)
