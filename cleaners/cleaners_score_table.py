"""
Sif-sentiment: Cleaners team code
@author: Farah & Vincent
"""

import pandas as pd
import numpy as np
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

    # case of empty data set, i.e., NO NEWS within 30 days
    if temp_dict['totalResults'] == 0:
        result_df = pd.DataFrame({'date': np.nan, 'title': np.nan, 'content': np.nan}, index=[0])
    # normal case
    else:
        temp_df = pd.DataFrame(temp_dict['articles'])
        result_df = temp_df.loc[:, ['publishedAt', 'title', 'description']]

        result_df.columns = ['date', 'title', 'content']  # reformat column name for consistency

    return result_df


def kaggle_filter(df, name):
    """

    :param df: processed kaggle df
    :param name: keyword string, usually be a specific security name instead of a ticker
    :return: filtered df
    """
    df_naFree = df.dropna()  # drop na
    df_naFree.reset_index(drop=True, inplace=True)

    picked_list = []
    for index in range(df_naFree.shape[0]):
        if name in df_naFree.iloc[index, 1] or name in df.iloc[index, 2]:  # check if name in 'title' or 'content'
            picked_list.append(index)

    # empty list case
    if len(picked_list) == 0:
        filtered_df = pd.DataFrame({'date': np.nan, 'title': np.nan, 'content': np.nan}, index=[0])
    else:
        filtered_df = df_naFree.iloc[picked_list, :]

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

    # format column name
    result_df.columns = ['date', 'title', 'content']

    return result_df


def vader_scores(df, name, column='title'):
    """
    calculate VADER sentiment score with all descriptions.
    Note: this is suggested to use on df with news data of ONE company.
    :param df: big df with a/some columns contains sentiment SENTENCES
    :param column: target column name
    :param name: the name of that ONE company, used in index
    :return: a df with sentiment scores
    """
    # drop na, and if no news at all, if no data, return a one-row df with value 0
    df.dropna(inplace=True)
    if df.shape[0] == 0:
        return pd.DataFrame(pd.DataFrame({'name': name, 'date': str(datetime.date.today()),
                                          'neg_score': 0, 'neu_score': 0, 'pos_score': 0}))

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
    """
    replaced through groupby(), reindex()
    :param df:
    :return:
    """
    '''
    q1 = """SELECT ticker, date, avg(neg_score) AS avg_negative_score, avg(neu_score) AS avg_neutral_score, avg(pos_score) AS avg_positive_score
            FROM final_table
            GROUP BY ticker, date
            ORDER BY date
            """
    '''
    # team2_table = ps.sqldf(q1, locals())
    pass


def formatting(df, *name):
    """
    re-format the content, indices, etc. of the given df, **especially the date values**.
    also, sort the df in ascending time order.

    :param df: raw df, usually scraped by other function from newsapi or kaggle data
    :param name: (OPTIONAL) company name of corresponding news df, since we're scraping news one by one
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
    df['date'] = df['date'].values.astype('datetime64[D]')  # drop all hour/min/sec value, we only need date

    # part 2, add a column of name, if name param exists
    for i in name:
        df['name'] = i

    return df


def initialize():
    pass


# run
if __name__ == "__main__":

    # time it!
    start_stopwatch = time.process_time()
    print(f'started at: {datetime.datetime.today()}')

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

    # initialize output lists
    score_list = []
    count_list = []

    # 1.2, loop over list and fetch company news data
    concat_data_list = []
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
    json_list = []
    for i in dir_list:
        #   # need further development to fetch multiple files
        temp_json_data = json_reader(i)
        json_list.append(temp_json_data)

    json_data = pd.concat(json_list)

    print('Success! \n')

    j = 0
    # fetch newsAPI data and filtering
    print('filtering & scoring, please wait...')
    for ticker, security in zip(ticker_df['Symbol'].values, ticker_df['Security'].values):
    # for ticker, security in zip(['AAPL', 'GOOG'], ['apple', 'google']):  # example test
        # in each loop, we scrap/filter specific company's news from newsAPI and kaggle data, and form a huge df

        # print live progress
        j += round(100/ticker_df.shape[0], 2)
        sys.stdout.write(f'Searching data with {security}...\r%d%%' % j)
        sys.stdout.flush()

        # api: scrap data
        api_data = get_api_news(ticker)
        api_data_sec = get_api_news(security)

        # format with formatting function
        api_data = formatting(api_data, ticker)
        api_data_sec = formatting(api_data_sec, ticker)  # here not using security since we need to put them together

        concat_data_list.append(api_data)  # test if list works, if error change into dict
        concat_data_list.append(api_data_sec)

        # kaggle: use filter function to generate a df
        kaggle_data1 = kaggle_filter(kaggle_1, security)
        kaggle_data2 = kaggle_filter(kaggle_2, security)
        kaggle_data3 = kaggle_filter(kaggle_3, security)
        # can use a dictionary to avoid repeating

        # json:
        filtered_json = kaggle_filter(json_data, security)

        # reformat dataframes
        for data in [kaggle_data1, kaggle_data2, kaggle_data3, filtered_json]:
            data = formatting(data, ticker)  # name consistent
            concat_data_list.append(data)  # add to concat list

        # 2. merge dataset into one huge df
        current_data = pd.concat(concat_data_list)
        current_data = formatting(current_data)
        current_data.reset_index(inplace=True, drop=True)  # reformat it again

        # current_data.dropna(inplace=True)

        # 3. calculate sentiment score (for the specific company)
        raw_score = vader_scores(current_data, ticker)

        # make two copy to avoid mixed reference
        score_copy1 = raw_score.copy()
        score_copy2 = raw_score.copy()
        # when groupby, use .agg([]) to apply multiple methods (not used, since nested columns are hard to handle

        # use groupby twice to deal with different team requests
        score_current_df = score_copy1.groupby(by='date', as_index=False).mean()
        score_current_df['name'] = ticker  # seems the name column is lost in groupby

        # the count_df should be like in the overleaf share doc, with only one ticker column
        # trick: only groupby with one score, and change that score count column name as ticker
        current_count_df = score_copy2.loc[:, ['date', 'neg_score']].groupby(by='date', as_index=False).count()
        current_count_df.columns = ['date', f'{ticker}']

        # 4. format the current dfs to make date consistent with different companies, with pd.date_range()
        # first make a date range as requested
        start_date = '2019-12-01'  # example date
        end_date = '2020-01-06'
        time_range = pd.bdate_range(start=start_date, end=end_date)  # bdate for business/working date

        # need to set index to use reindex()
        current_count_df.set_index('date', inplace=True)
        score_current_df.set_index('date', inplace=True)

        # fill the no-data value with 0
        current_count_df_full = current_count_df.reindex(time_range, fill_value=0).copy()
        score_current_df_full = score_current_df.reindex(time_range).copy()
        for col in ['neg_score', 'neu_score', 'pos_score']:
            score_current_df_full[col].fillna(value=0, inplace=True)
        score_current_df_full['name'].fillna(value=ticker, inplace=True)

        count_list.append(current_count_df_full)
        score_list.append(score_current_df_full)

        # 5. output

    # need out of loop
    result_score_df = pd.concat(score_list)
    result_count_df = pd.concat(count_list, axis=1)

    # end stopwatch!
    print('\nTime used: ' + str(time.process_time() - start_stopwatch))

    '''
    kaggle_1 = pd.read_csv('all-the-news/articles1.csv')
    kaggle_1 = kaggle_1[['date', 'title', 'content']]

    kaggle_2 = pd.read_csv('all-the-news/articles2.csv')
    kaggle_2 = kaggle_2[['date', 'title', 'content']]

    kaggle_3 = pd.read_csv('all-the-news/articles3.csv')
    kaggle_3 = kaggle_3[['date', 'title', 'content']]
    '''
    # df_list = [kaggle_1, kaggle_2, kaggle_3]
    print('everything done!')
    return result_score_df





