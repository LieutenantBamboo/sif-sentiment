from newsapi import NewsApiClient
import pandas as pd
import math
import datetime


api = NewsApiClient(api_key='e5d3b48c925c4042a00f82ba31b07c53')
'''
b = api.get_everything(sources='business-insider-uk, bloomberg, business-insider, australian-financial-review, '
                               'cnbc, fortune, the-wall-street-journal'
                       , language='en', from_param='2019-11-16', to='2019-12-16', page_size=100)  # no further than 30 days
df = pd.DataFrame(b['articles'])
'''

# newsapi only allow 240 chars for Developer plan users, so our data for each article contains:
# source; title; description; content
# will use all of them for future processing

# also, newsapi seems not to cover Financial Times

# we can limit sources, limit Keywords

# the default is 20 per page, and the page size could be changed & set by pageSize param


def get_raw_news(name):
    """
    Get article via newsAPI.
    :param name: target company name
    :return: a dataframe
    """
    # get initial df for 1st page
    keyword = str(name)
    today = datetime.date.today()
    amonth_b4=today+ datetime.timedelta(-30)
    temp_dict = api.get_everything(sources='business-insider-uk, bloomberg, business-insider, '
                                           'australian-financial-review, '
                                           'cnbc, fortune, the-wall-street-journal',
                                   language='en', from_param=str(amonth_b4), to=str(today), page_size=100, q=keyword)
    temp_df = pd.DataFrame(temp_dict['articles'])
    result_df = temp_df[['title', 'description', 'content']]
    return result_df


'''
next steps:
1. get a list of stock tickers with corresponding company names
2. re-arrange df to extract data that able to process
3. developer accounts only limited to 100 results...
'''

# test
df = get_raw_news('Apple')


