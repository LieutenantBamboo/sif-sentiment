import xlsxwriter


def make_workbook(filename, search_bar):
    """
    Make a new workbook in excel
    :param filename: name of excel file
    :param search_bar: the equation to put in the first cell of excel
    """
    workbook = xlsxwriter.Workbook(filename)
    worksheet = workbook.add_worksheet()
    worksheet.write(0, 0, search_bar)
    workbook.close()


def fundamentals(ticker, start_date, end_date):
    """
    Function to return technical analysis
    :param ticker: Ticker
    :param start_date: Start date
    :param end_date: End date
    :return: None
    """

    filename1 = str(ticker) + "_1.xlsx"
    make_workbook(filename1, '=BDH("' + str(ticker) + ' US EQUITY",' + '"MARKET_CAPITALIZATION_TO_EV, '
                                                                       'GR_MRGN_RETURN_ON_INVTRY_INVEST, APPLIED_BETA'
                                                                       ' , REL_SHR_PX_MOMENTUM, VOLATILITY_10D, '
                                                                       'VOLATILITY_30D ","' + start_date + '",'
                                                                                                           '"' +
                  end_date + '","curr=USD","per=cm","cols=11;rows=1792")')

    filename2 = str(ticker) + "_2.xlsx"
    make_workbook(filename2, '=BDH("' + str(ticker) + ' US EQUITY",' + '"VOLATILITY_60D, VOLATILITY_90D, '
                                                                       'VOLATILITY_180D, DVD_PAYOUT_RATIO, '
                                                                       'INTEREST_COVERAGE_RATIO, RETURN_ON_ASSET","' +
                  start_date + '","' + end_date + '","curr=USD","per=cm","cols=11;rows=1792")')

    filename3 = str(ticker) + "_3.xlsx"
    make_workbook(filename3, '=BDH("' + str(ticker) + ' US EQUITY",' + '"NORMALIZED_ROE, PE_RATIO, PX_TO_BOOK_RATIO, '
                                                                       'TRAIL_12M_EPS, PX_TO_SALES_RATIO, '
                                                                       'EV_TO_T12M_EBITDA","' +
                  start_date + '","' + end_date + '","curr=USD","per=cm","cols=11;rows=1792")')

    filename4 = str(ticker) + "_4.xlsx"
    make_workbook(filename4, '=BDH("' + str(ticker) + ' US EQUITY",' + '"MARKET_CAPITALIZATION_TO_BV, '
                                                                       'EV_TO_T12M_EBIT, FREE_CASH_FLOW_PER_SH'
                                                                       ' , CUR_RATIO, CASH_RATIO, '
                                                                       'TOT_DEBT_TO_TOT_EQY","' +
                  start_date + '","' + end_date + '","curr=USD","per=cm","cols=11;rows=1792")')

    filename5 = str(ticker) + "_5.xlsx"
    make_workbook(filename5, '=BDH("' + str(ticker) + ' US EQUITY",' + '"ASSET_TURNOVER, '
                                                                       'INVENT_TURN, GROSS_MARGIN'
                                                                       ' , RETURN_COM_EQY, GROSS_PROFIT, '
                                                                       'NET_INCOME","' +
                  start_date + '","' + end_date + '","curr=USD","per=cm","cols=11;rows=1792")')



if __name__ == "__main__":
    # test run on AAPL
    fundamentals('AAPL', '1990-01-01', '2020-01-01')

