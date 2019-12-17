import requests
import pandas as pd
import numpy as np
from pprint import pprint
url = ('https://newsapi.org/v2/sources?language=en&category=business&apiKey=e5d3b48c925c4042a00f82ba31b07c53')
response = requests.get(url)
# a = pd.DataFrame(response.json())
# b = pd.DataFrame(a['articles'][0])
pprint(response.json())

'''
"id": "the-wall-street-journal",
"name": "The Wall Street Journal",
"description": "WSJ online coverage of breaking news and current headlines from the US and around the world. Top stories, photos, videos, detailed analysis and in-depth reporting.",
"url": "http://www.wsj.com",
"category": "business",
"language": "en",
"country": "us"

'''

