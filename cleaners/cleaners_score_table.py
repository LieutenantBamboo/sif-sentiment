import pandas as pd
import numpy as np
from newsapi import NewsApiClient
import pandasql as ps
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import datetime

analyser = SentimentIntensityAnalyzer()
api = NewsApiClient(api_key='e5d3b48c925c4042a00f82ba31b07c53')


# List of tickers
# generate a list of all the updated tickers for the S&P 500 (from Wikipedia, since they update it often)

table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
ticker_df = table[0]
# the ticker_df['Symbol'] contains the tickers, ticker_df['security'] contains name
# use name or ticker, the search result from newsApi may be different.
# so we use both, then merge them together (newsapi)

# use several functions to do this, to make script modular

# ** THE GOAL is to generate a score-date df for each stock

# api generate

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
    result_df = temp_df[['publishedAt', 'title', 'description', 'content']]
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


def content_scores(df):
    pass


def formatting(df, name):
    """
    re-format the content, indices, etc. of the given df, **especially the date values**.
    also, sort the df in ascending time order.
    :param df: raw df, usually scraped by other function from newsapi or kaggle data
    :param name: company name of corresponding news df, since we're scraping news one by one
    :return: a re-formatted df, easy for further processing.
    """
    datetime_object = datetime.strptime(test.iloc[0, 0], '%Y-%m-%dT%H:%M:%S.%f%z')


# run
if __name__ == "__main__":
    df = get_api_news('AAPL')
    df2 = get_api_news('Apple')

kaggle_1 = pd.read_csv('all-the-news/articles1.csv', names=['title', 'date', 'content'])
kaggle_2 = pd.read_csv('all-the-news/articles2.csv', names=['title', 'date', 'content'])
kaggle_3 = pd.read_csv('all-the-news/articles3.csv', names=['title', 'date', 'content'])

df_list = [kaggle_1, kaggle_2, kaggle_3]

