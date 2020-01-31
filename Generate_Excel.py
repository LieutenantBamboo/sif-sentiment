import xlsxwriter


def make_workbook(filename, search_bar):
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
    make_workbook(filename1, '=BDH("' + str(ticker) + ' US EQUITY",' + \
                  '"DVD_PAYOUT_RATIO, PE_RATIO,PX_TO_BOOK_RATIO, TRAIL_12M_EPS,\
                  PX_TO_SALES_RATIO, MARKET_CAPITALIZATION_TO_EV","' + \
                  start_date + '","' + end_date + '","curr=USD","cols=11;rows=1792")')

    filename2 = str(ticker) + "_2.xlsx"
    make_workbook(filename2, '=BDH("' + str(ticker) + ' US EQUITY",' + \
                  '"GR_MRGN_RETURN_ON_INVTRY_INVEST, EV_TO_T12M_EBITDA,\
                  MARKET_CAPITALIZATION_TO_BV, EV_TO_T12M_EBIT, INTEREST_COVERAGE_RATIO","' + \
                  start_date + '","' + end_date + '","curr=USD","cols=11;rows=1792")')

    filename3 = str(ticker) + "_3.xlsx"
    make_workbook(filename3, '=BDH("' + str(ticker) + ' US EQUITY",' + \
                  '"RETURN_ON_ASSET, ENORMALIZED_ROE,\
                  MARKET_CAPITALIZATION_TO_BV, EV_TO_T12M_EBIT, INTEREST_COVERAGE_RATIO","' + \
                  start_date + '","' + end_date + '","curr=USD","cols=11;rows=1792")')


if __name__ == "__main__":
    # test run on AAPL
    fundamentals('AAPL', '2000-01-01', '2019-03-01')
