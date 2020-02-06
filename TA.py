import ta
import yfinance as yf


def technical_analysis(ticker, start_date, end_date):
    """
    Function to return technical analysis
    :param ticker: Ticker
    :param start_date: Start date for TA
    :param end_date: End date for TA
    :return: Dataframe containing all TA information
    """
    # getting yahoo data
    data = yf.download(ticker, start_date, end_date)

    # adding all TA features
    data = ta.add_all_ta_features(
        data, open="Open", high="High", low="Low", close="Close", volume="Volume")

    return data



if __name__ == "__main__":
    # test run on AAPL
    ta = technical_analysis('AAPL', '2000-01-01', '2019-03-01')

    # list of technical analysis indicators
    print(ta.columns)

    # table of ta
    print(ta)
