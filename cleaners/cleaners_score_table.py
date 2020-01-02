import pandas as pd
import numpy as np
import pandasql as ps
import os
import json
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import datetime
# the below two just for counting progress
import sys
import time

analyser = SentimentIntensityAnalyzer()
api = NewsApiClient(api_key='e5d3b48c925c4042a00f82ba31b07c53')

# use several functions to do this, to make script modular

# ** THE GOAL is to generate a score-date df for each stock


def get_api_news(name):
    """
    Get article via newsAPI.
    :param name: target company name
    :return: a dataframe
    """
    # get initial df for 1st page - 100 articles
    keyword = str(name)
    today = datetime.date.today()
    amonth_b4 = today + datetime.timedelta(-30)
    temp_dict = api.get_everything(sources='business-insider-uk, bloomberg, business-insider, '
                                           'australian-financial-review, '
                                           'cnbc, fortune, the-wall-street-journal',
                                   language='en', from_param=str(amonth_b4), to=str(today), page_size=100, q=keyword)
    temp_df = pd.DataFrame(temp_dict['articles'])
    result_df = temp_df[['publishedAt', 'title', 'description']]

    result_df.columns = ['date', 'title', 'content']  # reformat column name for consistency

    return result_df


def kaggle_filter(df, name):
    """

    :param df: processed kaggle df
    :param name: keyword string
    :return: filtered df
    """
    picked_list = []
    for i in range(df.shape(0)):  # check if name in 'title'
        if name in df.iloc[i, 0]:
            picked_list.append(i)

    filtered_df = df[picked_list, :]

    return filtered_df


def tokenize(sentence):
    """
    Use nltk to tokenize out sentences.
    :param sentence: input paragraph
    :return: a list of strings, each element is a sentence.
    """
    pass


def create_dataframe(list):
    pass


def json_reader(dir):
    """
    read all *.json file in specified directory.
    :param dir: target directory
    :return: a DataFrame with required info, each json file composes one row
    """
    # load json files and put them into a nested dictionary
    dict = {}
    file_list = os.listdir(dir)
    for i in range(len(file_list)):
        temp_json = open(f'{dir}/{file_list[i]}', encoding="utf8")  # encoding must be set
        temp_str = temp_json.read()  # store the string (this is a command that can only run once for each json)
        dict[f'json{i}'] = json.loads(temp_str)  # convert stringed dict into dictionary

    # form dataframe
    news_list = []
    news_content = []
    for news, content in dict.items():  # loop over each item in nested dict
        news_list.append(news)
        content_dict = {}
        for i in ['published', 'title', 'text']:  # pick out the required info
            content_dict[i] = content[i]  # form a new dict for pandas use
        news_content.append(pd.DataFrame(content_dict, index=[0]))  # need an index, otherwise pandas will error
    result_df = pd.concat(news_content, keys=news_list)  # use pd.concat to combine all the news data

    return result_df


def vader_scores(df, name, column='description'):
    """
    calculate VADER sentiment score with all descriptions.
    Note: this is suggested to use on df with news data of ONE company.
    :param df: big df with a/some columns contains sentiment SENTENCES
    :param column: target column name
    :param name: the name of that ONE company, used in index
    :return: a df with sentiment scores
    """
    date = df['date'].values
    pos_list = []
    neg_list = []
    neu_list = []
    for description in df[column].values:
        score_dict = analyser.polarity_scores(description)
        score_pos = score_dict['pos']
        score_neg = score_dict['neg']
        score_neu = score_dict['neu']

        pos_list.append(score_pos)
        neg_list.append(score_neg)
        neu_list.append(score_neu)

    result_df = pd.DataFrame({'name': name, 'date': date,
                              'neg_score': neg_list, 'neu_score': neu_list, 'pos_score': pos_list})

    return result_df


def average_scores(df):
    q1 = """SELECT ticker, date, avg(neg_score) AS avg_negative_score, avg(neu_score) AS avg_neutral_score, avg(pos_score) AS avg_positive_score
            FROM final_table
            GROUP BY ticker, date
            ORDER BY date
            """

    team2_table = ps.sqldf(q1, locals())


def formatting(df, name):
    """
    re-format the content, indices, etc. of the given df, **especially the date values**.
    also, sort the df in ascending time order.

    :param df: raw df, usually scraped by other function from newsapi or kaggle data
    :param name: company name of corresponding news df, since we're scraping news one by one
    :return: a re-formatted df, easy for further processing.
    """
    # part 1, change the date format and sort
    '''
    try:
        df['date'] = df['date'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
    except ValueError:
        try:
            df['date'] = df['date'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d%H:%M:%S.%f%z'))
        except ValueError:
            try:
                df['date'] = df['date'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%SZ'))
            except ValueError:
                print('ERROR: wrong date schedule!')
    '''

    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by='date', inplace=True)
    df['date'].values.astype('datetime64[D]')

    # part 2, add a column of name
    df['name'] = name

    return df


# run

if __name__ == "__main__":
    '''
    From team 2:
    
    if on one day, there's no news on this company, directly set this date's sentiment value 
    to ZERO.
    
    the date must be continuous, with a adjustable start/end date.
    '''
    # Step 1, Fetch raw data
    # 1.1, generate a tickers list
    # the ticker_df['Symbol'] contains the tickers, ticker_df['security'] contains name
    # use name or ticker, the search result from newsApi may be different.
    # so we try use both, then merge them together (newsapi)

    table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    ticker_df = table[0]
    ticker_df = ticker_df.loc[:, ['Symbol', 'Security', 'GICS Sector']]
    # so we have a df with tickers, name and industry sector.
    # 1.2, loop over list and fetch company news data

    # load kaggle data
    print('Loading Kaggle data... please wait')
    kaggle_1 = pd.read_csv('all-the-news/articles1.csv')
    kaggle_1 = kaggle_1[['date', 'title', 'content']]

    kaggle_2 = pd.read_csv('all-the-news/articles2.csv')
    kaggle_2 = kaggle_2[['date', 'title', 'content']]

    kaggle_3 = pd.read_csv('all-the-news/articles3.csv')
    kaggle_3 = kaggle_3[['date', 'title', 'content']]
    print('Success! \n')

    # load json data
    print('Loading *.json data... please wait')
    dir_list = ['/Users/vincentz/Desktop/Vincent/Python_Projects/investment_fund/us-financial-news-articles/2018_01_11'
                ]
    for i in dir_list:  # need further development to fetch multiple files
        json_data = json_reader(i)

    print('Success! \n')

    i = 0
    for ticker, name in zip(ticker_df['Symbol'].values, ticker_df['Security'].values):

        # print live progress
        i += round(100/ticker_df.shape[0], 2)
        sys.stdout.write(f'Searching data with {name}...\r%d%%' % i)
        sys.stdout.flush()

        # api: scrap data one by one
        api_data = get_api_news(ticker)
        # format with formatting
        api_data = formatting(api_data, ticker)

        # kaggle: use filter function to generate a df
        kaggle_data1 = kaggle_filter(kaggle_1, name)
        kaggle_data2 = kaggle_filter(kaggle_2, name)
        kaggle_data3 = kaggle_filter(kaggle_3, name)
        # can use a dictionary to avoid repeating
        for data in [kaggle_data1, kaggle_data2, kaggle_data3]:
            data = formatting(data, ticker)

        # 2. merge dataset into one huge df

        # 3. calculate sentiment score

        # 4. calculate average daily score with pandasql

        # output


    '''
    kaggle_1 = pd.read_csv('all-the-news/articles1.csv')
    kaggle_1 = kaggle_1[['date', 'title', 'content']]

    kaggle_2 = pd.read_csv('all-the-news/articles2.csv')
    kaggle_2 = kaggle_2[['date', 'title', 'content']]

    kaggle_3 = pd.read_csv('all-the-news/articles3.csv')
    kaggle_3 = kaggle_3[['date', 'title', 'content']]
    '''
    # df_list = [kaggle_1, kaggle_2, kaggle_3]

    # test json reader
    json_test = json_reader('newsjson')





