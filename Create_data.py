import ta
import yfinance as yf
import pandas as pd


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


def merge_data(ticker, ta):
    filename1 = str(ticker) + "_1.xlsx"
    df1 = pd.read_excel(filename1, index_col=0, header=None)
    df1.columns = ["MARKET_CAPITALIZATION_TO_EV", "GR_MRGN_RETURN_ON_INVTRY_INVEST", "APPLIED_BETA"
        , "REL_SHR_PX_MOMENTUM", "VOLATILITY_10D", "VOLATILITY_30D"]

    filename2 = str(ticker) + "_2.xlsx"
    df2 = pd.read_excel(filename2, index_col=0, header=None)
    df2.columns = ["VOLATILITY_60D", "VOLATILITY_90D", "VOLATILITY_180D", "DVD_PAYOUT_RATIO", "INTEREST_COVERAGE_RATIO",
                   "RETURN_ON_ASSET"]

    filename3 = str(ticker) + "_3.xlsx"
    df3 = pd.read_excel(filename3, index_col=0, header=None)
    df3.columns = ["NORMALIZED_ROE", "PE_RATIO", "PX_TO_BOOK_RATIO", "TRAIL_12M_EPS", "PX_TO_SALES_RATIO",
                   "EV_TO_T12M_EBITDA"]

    filename4 = str(ticker) + "_4.xlsx"
    df4 = pd.read_excel(filename4, index_col=0, header=None)
    df4.columns = ["MARKET_CAPITALIZATION_TO_BV", "EV_TO_T12M_EBIT", "FREE_CASH_FLOW_PER_SH", "CUR_RATIO", "CASH_RATIO",
                   "TOT_DEBT_TO_TOT_EQY"]

    filename5 = str(ticker) + "_5.xlsx"
    df5 = pd.read_excel(filename5, index_col=0, header=None)
    df5.columns = ["ASSET_TURNOVER", "INVENT_TURN", "GROSS_MARGIN", "RETURN_COM_EQY", "GROSS_PROFIT", "NET_INCOME"]

    merged_df = pd.concat([df1, df2, df3, df4, df5, ta], axis=1, join='inner')

    return (merged_df)


if __name__ == "__main__":
    ta = technical_analysis('AAPL', '1990-01-01', '2020-01-01')
    # if want to merge without prices in ta, uncomment the following line
    # ta = ta.iloc[:,5:]
    merged_df = merge_data('AAPL', ta)
    print(merged_df)
