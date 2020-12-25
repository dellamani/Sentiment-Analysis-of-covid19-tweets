import pandas as pd
import plotly
import plotly.express as px  # (version 4.7.0)
import plotly.graph_objects as go

import dash  # (version 1.12.0) pip install dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import GetOldTweets3 as got

from statsmodels.tsa.arima_model import ARIMA

from datetime import datetime as dt, timedelta
import time as ti
import pickle
import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english')) # Set of stopwords (Words that doesn't give meaningful information)

lemmatizer = WordNetLemmatizer()  # Used for converting words with similar meaning to a single word.

def text_process(tweet):

    processed_tweet = [] # To store processed text

    tweet = tweet.lower() # Convert to lower case

    tweet = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', 'URL', tweet) # Replaces any URLs with the word URL

    tweet = re.sub(r'@[\S]+', 'USER_MENTION', tweet) # Replace @handle with the word USER_MENTION

    tweet = re.sub(r'#(\S+)', r' \1 ', tweet) # Removes # from hashtag

    tweet = re.sub(r'\brt\b', '', tweet) # Remove RT (retweet)

    tweet = re.sub(r'\.{2,}', ' ', tweet) # Replace 2+ dots with space

    tweet = tweet.strip(' "\'') # Strip space, " and ' from tweet

    tweet = re.sub(r'\s+', ' ', tweet) # Replace multiple spaces with a single space

    words = tweet.split()

    for word in words:

        word = word.strip('\'"?!,.():;') # Remove Punctuations

        word = re.sub(r'(.)\1+', r'\1\1', word) # Convert more than 2 letter repetitions to 2 letter (happppy -> happy)

        word = re.sub(r'(-|\')', '', word) # Remove - & '

        if (re.search(r'^[a-zA-Z][a-z0-9A-Z\._]*$', word) is not None): # Check if the word starts with an english letter

            if(word not in stop_words):                                 # Check if the word is a stopword.

                word = str(lemmatizer.lemmatize(word))                  # Lemmatize the word

                processed_tweet.append(word)

    return ' '.join(processed_tweet)

predictor = pickle.load(open("Sentiment Predictor", 'rb'))  # Code to load ML model for later use

app = dash.Dash(__name__)
app.title = 'Covid19 Sentiment Analysis'
server = app.server

date_range = {1:['2020-03-25','2020-04-14'],
              2:['2020-04-15','2020-05-03'],
              3:['2020-05-04','2020-05-17'],
              4:['2020-05-18','2020-05-31'],
              5:['2020-06-01','2020-06-14']}

phase_range = {1:'LD1',2:'LD2',3:'LD3',4:'LD4',5:'Unlock1'}

image_range = {0:'General cloud.png',
               1:'Lockdown1 cloud.png',
               2:'Lockdown2 cloud.png',
               3:'Lockdown3 cloud.png',
               4:'Lockdown4 cloud.png',
               5:'Unlock1 cloud.png'}

image_location = '/assets/'

# ----------------------------------------------------------------------------------------------------------------------------------------
#Import and clean data (importing csv into pandas)
df = pd.read_csv("Hashtag Data.csv")
df1 = pd.read_csv("Topic Data.csv")
df2 = pd.read_csv("Sentiment.csv")
df2['Date'] = pd.to_datetime(df2['Date'])
df3 = pd.read_csv("Tone Data.csv")
df4 = pd.read_csv("Graph Prediction.csv")


# ----------------------------------------------------------------------------------------------------------------------------------------
# App layout
app.layout = html.Div([

    html.Div(
    html.H1("COVID-19 Sentiment Analysis",className = "jumbotron"),
    style={'padding-bottom':'20px'}),

    html.Div(
    dcc.Slider(id="slct_period",
                min=0,
                max=5,
                marks={
                    0:{'label': 'General', 'style': {'color': '#FFFFFF','font-size':'18px','padding-left':'10px'}},
                    1:{'label': 'LD1', 'style': {'color': '#FFFFFF','font-size':'18px'}},
                    2:{'label': 'LD2', 'style': {'color': '#FFFFFF','font-size':'18px'}},
                    3:{'label': 'LD3', 'style': {'color': '#FFFFFF','font-size':'18px'}},
                    4:{'label': 'LD4', 'style': {'color': '#FFFFFF','font-size':'18px'}},
                    5:{'label': 'Unlock1', 'style': {'color': '#FFFFFF','font-size':'18px','padding-right':'10px'}}
                },
                value=0
                )),

    html.Div(
    dcc.Graph(id = 'sentiment_analysis'),style={'padding-top':'20px','padding-bottom':'20px'}),

    html.Div([

    html.Div(
    className='row',
    children = [
    html.Div(
    dcc.Graph(id = 'sentiment_analysis1'),className='column'),

    html.Div(
    dcc.Graph(id = 'sentiment_analysis2'),className='column')]),

    html.Div(
    className='row',
    children = [
    html.Div(
    dcc.Graph(id = 'sentiment_analysis3'),className='column'),

    html.Img(id = 'word_cloud',className='column',style = {"height": "450px"})])]),

    html.Div(dcc.Graph(id='sentiment_graph'),style={'padding-top':'10px','padding-bottom':'10px'}),

    html.H2("Sentiment Prediction end date: ",style={'width': '50%', 'display': 'inline-block', 'text-align': 'right','padding-right':'10px'}),

    html.Div(
    dcc.DatePickerSingle(
        id='sentiment_date',
        min_date_allowed=dt(2020, 6, 2),
        max_date_allowed=dt(2020, 9, 1),
        initial_visible_month=dt(2020,6,1),
        date='2020-6-2'),
        style={'margin-left':'-5px','width': '40%', 'display': 'inline-block'}),

    html.Div(
    html.Button('Live Tweet Sentiments', id='submit-val', n_clicks=0,className = 'livebutton'),
    className = 'livecenter'),

    html.Div(id='sentiment_live',style = {'padding-bottom':'30px','font-size':'20px','font-family':'Geneva'}),

    html.H2("Find the sentiment of text: ",style={'width': '50%', 'display': 'inline-block', 'text-align': 'right'}),

    html.Div(
    dcc.Input(
            id="sentiment_text",
            type = 'text',
            size="25",
            value='',
            placeholder="Input the text",
            style = {"height":"30px"}),
            style={'width': '45%', 'display': 'inline-block', 'text-align': 'left','padding-left':'10px'}),

    html.Div(id="sentiment_prediction",style={'text-align': 'center','padding':'15px','font-size':'25px','font-family':'Geneva'})

    ])


# ------------------------------------------------------------------------------
# Connect the Plotly graphs with Dash Components

@app.callback(
     Output('sentiment_analysis', 'figure'),
    [Input('slct_period', 'value')]
)

def update_graph(slct_period):
    if slct_period == 0:
        df_plot = df2.copy()
    else:
        start,end=map(str,date_range[slct_period])
        df_plot = df2[(df2['Date'] >= start) & (df2['Date'] <= end)]
    fig = px.line(df_plot,x='Date',y='Value',color='Sentiment')
    fig.update_layout(title={'text': "Sentiment of People",'y':0.95,'x':0.48,'xanchor': 'center','yanchor': 'top'})

    return fig

@app.callback(
     Output('sentiment_analysis1', 'figure'),
    [Input('slct_period', 'value')]
)

def update_graph(slct_period):
    if slct_period == 0:
        df_plot = df.copy().groupby('hashtag').sum().reset_index().sort_values(by='value',ascending=False)
    else:
        df_plot = df[df['Phase']==phase_range[slct_period]]

    fig = px.bar(df_plot,x='hashtag',y='value',color='value',labels={'hashtag':'Hashtags','value':"Values"},height=500)
    fig.update_layout(title={'text': "Trending Hashtags",'y':0.95,'x':0.47,'xanchor': 'center','yanchor': 'top'})

    return fig

@app.callback(
     Output('sentiment_analysis2', 'figure'),
    [Input('slct_period', 'value')]
)

def update_graph(slct_period):
    if slct_period == 0:
        df_plot = df1.copy().groupby('topic_data').sum().reset_index().sort_values(by='topic_count',ascending=False)
    else:
        df_plot = df1[df1['Phase']==phase_range[slct_period]]
    fig = px.bar(df_plot,x='topic_data',y='topic_count',color='topic_count',labels={'topic_data':'Topics','topic_count':"Values"},height=500)
    fig.update_layout(xaxis=dict(showticklabels=False),title={'text': "Trending Topics",'y':0.95,'x':0.49,'xanchor': 'center','yanchor': 'top'})

    return fig

@app.callback(
     Output('sentiment_analysis3', 'figure'),
    [Input('slct_period', 'value')]
)

def update_graph(slct_period):
    if slct_period == 0:
        df_plot = df3.copy().groupby('Tone').sum().reset_index().sort_values(by='Value',ascending=False)
    else:
        start,end=map(str,date_range[slct_period])
        df_plot = df3[df3['Phase']==phase_range[slct_period]]
    fig = px.pie(df_plot,names="Tone",values="Value")
    fig.update_layout(title={'text': "Tone of 500 Sample tweets",'y':0.95,'x':0.47,'xanchor': 'center','yanchor': 'top'})

    return fig

@app.callback(
     Output('word_cloud', 'src'),
    [Input('slct_period', 'value')]
)

def update_image(slct_period):
    return image_location+image_range[slct_period]

@app.callback(
     Output('sentiment_live', 'children'),
    [Input('submit-val', 'n_clicks')]
)

def update_graph(slct_period):
    tweetCriteria = got.manager.TweetCriteria().setQuerySearch('corona OR coronavirus OR covid-19 OR covid19 OR covid OR pandemic OR lockdown')\
                                               .setNear("Nagpur,India")\
                                               .setLang('en')\
                                               .setWithin("1500km")\
                                               .setMaxTweets(1)

    tweet = got.manager.TweetManager.getTweets(tweetCriteria)[0]
    correct_time = tweet.date + timedelta(hours=5,minutes=30)
    correct_time = correct_time.strftime("%d-%m-%Y %I:%M:%S")
    date,time = correct_time.split()
    predict = predictor.predict([text_process(tweet.text)])
    result = "Positive" if predict>0 else ("Neutral" if predict == 0 else "Negative")

    return (" Date: {} ".format(date),html.Br()," Time: {} ".format(time),html.Br()," Text: {} ".format(tweet.text),html.Br()," Sentiment: {} ".format(result))

@app.callback(
    Output('sentiment_graph', 'figure'),
    [Input('sentiment_date', 'date')])

def update_output(date):

    start = dt(2020,6,1)
    end = date
    year, month, day = map(int, end.split('-'))
    end = dt(year, month, day)

    result = df4
    train = result.iloc[:68]
    test = result.iloc[68:]
    model = ARIMA(train.Positive, order=(2,2,1))
    model_fit = model.fit(disp=-1)

    delta = end - start
    test_dates = []
    for i in range(delta.days + 1):
        test_dates.append(str(start + timedelta(days=i)).split()[0])

    forecast = model_fit.forecast(steps=delta.days+1)

    predicted_data = pd.DataFrame(forecast[0],index=test_dates,columns=['Positive'])

    #Neutral
    model = ARIMA(train.Neutral, order=(2,2,1))
    model_fit = model.fit(disp=-1)
    forecast = model_fit.forecast(steps=delta.days+1)
    predicted_data['Neutral'] = forecast[0]

    #Negative
    model = ARIMA(train.Negative, order=(2,2,1))
    model_fit = model.fit(disp=-1)
    forecast = model_fit.forecast(steps=delta.days+1)
    predicted_data['Negative'] = forecast[0]

    #Plot the data

    fig = px.line(predicted_data,labels={'index':'Date','value':"Percentage Change in Sentiments"})
    fig.update_layout(legend_title_text='Sentiment',title={'text': "Predicted Sentiment Graph",'y':0.95,'x':0.47,'xanchor': 'center','yanchor': 'top'})

    return fig


@app.callback(
     Output('sentiment_prediction', 'children'),
    [Input('sentiment_text', 'value')]
)

def update_output_div(input_text):

    predict = predictor.predict([text_process(input_text)])

    result = "Positive" if predict>0 else ("Neutral" if predict == 0 else "Negative")

    return 'Sentiment is : {}'.format(result)


# ------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)
