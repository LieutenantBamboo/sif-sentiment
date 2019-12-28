import numpy as np
import pandas as pd
import yfinance as yf


class Team3:
    def __init__(self, team1_output, team2_output, date):
        # team 2 output is a data frame with tickers and probability of up/down
        self.team2_output = team2_output
        self.team1_output = team1_output
        self.date = date

    def getclose(self,stocks_to_invest):
        """
        Function to get close from Yahoo
        :param stocks_to_invest: filtered stocks
        :return: table with date as index and ticker close prices as columns
        """
        comp_returns = pd.DataFrame()
        ticker = stocks_to_invest['Ticker'].values

        # get data for each company from Yahoo
        for comp in ticker:
            yahoo = yf.Ticker(comp)
            historical_stock = yahoo.history(period="1y", interval="1mo")["Close"].to_frame().reset_index()
            historical_stock['Ticker'] = comp
            if len(comp_returns) == 0:
                comp_returns = historical_stock
            else:
                comp_returns = comp_returns.append(historical_stock)

        # Set index to date
        df = comp_returns.set_index('Date')

        # pivot data frame by ticker
        table = df.pivot(columns='Ticker')

        # Tidy up by dropping Nans
        table.columns = [col[1] for col in table.columns]
        table = table.dropna()

        # return Table
        return table

    def portfolio_annualised_performance(self, weights, mean_returns, cov_matrix):
        """
        Calculate portfolio annualised performance
        :param weights: randomly generated weights
        :param mean_returns: average returns
        :param cov_matrix: covariance matrix
        :return: standard deviation & returns
        """
        returns = np.sum(mean_returns * weights) * 252
        std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        return std, returns

    def random_portfolios(self, stocks_to_invest,num_portfolios, mean_returns, cov_matrix, risk_free_rate):
        """
        Generate random portfolios for optimisation later on
        :param stocks_to_invest: filtered stocks
        :param num_portfolios: number of randomly generated portfolios
        :param mean_returns: average returns
        :param cov_matrix: covariance matrix
        :param risk_free_rate: risk free rate
        :return: results (contains std deviation, returns, sharpe ratio), weights
        """
        results = np.zeros((3, num_portfolios))
        weights_record = []
        for i in range(num_portfolios):
            weights = np.random.uniform(low=-1.0, high=1.0, size=len(stocks_to_invest))
            weights /= np.sum(weights)
            weights_record.append(weights)
            portfolio_std_dev, portfolio_return = self.portfolio_annualised_performance(weights, mean_returns,
                                                                                        cov_matrix)
            results[0, i] = portfolio_std_dev
            results[1, i] = portfolio_return
            results[2, i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
        return results, weights_record

    def display_simulated_ef_with_random(self, stocks_to_invest, table, mean_returns, cov_matrix, num_portfolios,
                                         risk_free_rate):
        """
        Display results
        :param stocks_to_invest: filtered stocks
        :param table: table generated from getclose(). Data frame of close values
        :param mean_returns: average returns
        :param cov_matrix: covariance matrix
        :param num_portfolios: number of randomly generated portfolios
        :param risk_free_rate: risk free rate
        :return: -
        """
        results, weights = self.random_portfolios(stocks_to_invest, num_portfolios, mean_returns, cov_matrix,
                                                  risk_free_rate)
        # maximise sharpe ratio
        max_sharpe_idx = np.argmax(results[2])
        sdp, rp = results[0, max_sharpe_idx], results[1, max_sharpe_idx]
        max_sharpe_allocation = pd.DataFrame(weights[max_sharpe_idx], index=table.columns, columns=['allocation'])
        max_sharpe_allocation.allocation = [round(i * 100, 2) for i in max_sharpe_allocation.allocation]
        max_sharpe_allocation = max_sharpe_allocation.T

        # minimise volatility
        min_vol_idx = np.argmin(results[0])
        sdp_min, rp_min = results[0, min_vol_idx], results[1, min_vol_idx]
        min_vol_allocation = pd.DataFrame(weights[min_vol_idx], index=table.columns, columns=['allocation'])
        min_vol_allocation.allocation = [round(i * 100, 2) for i in min_vol_allocation.allocation]
        min_vol_allocation = min_vol_allocation.T

        print("-" * 80)
        print("Maximum Sharpe Ratio Portfolio Allocation\n")
        print("Annualised Return:", round(rp, 2))
        print("Annualised Volatility:", round(sdp, 2))
        print("\n")
        print(max_sharpe_allocation)
        print("-" * 80)
        print("Minimum Volatility Portfolio Allocation\n")
        print("Annualised Return:", round(rp_min, 2))
        print("Annualised Volatility:", round(sdp_min, 2))
        print("\n")
        print(min_vol_allocation)

    def equalweight(self, stocks_to_invest):
        nticker = len(stocks_to_invest)
        nlong = len(stocks_to_invest[stocks_to_invest['Prob Up']>stocks_to_invest['Prob Down']])
        nshort = nticker - nlong
        weightlong = 1/nlong
        weightshort = -1/nshort
        stocks_to_invest['Equal Weight'] = np.where(stocks_to_invest['Prob Up'] > stocks_to_invest['Prob Down'],
                                                    weightlong, weightshort)
        return stocks_to_invest


    def filter(self, quality, sentiment):
        """
        Function to filter stocks based on a quality variable and sentiment variable
        :param: quality: A number signifying the quality value
        :return: a list of stocks that is good to invest
        """
        # get today's quality
        today_stocks = self.team1_output[(self.team1_output['Date'] == self.date)]
        today_stocks.index = pd.Index(['Quality'])

        # filter by quality
        today_stocks = today_stocks.transpose()[1:]
        quality_stocks = today_stocks[today_stocks['Quality'] > quality].index.tolist()

        # filter by sentiment
        # get today's data
        today_stocks_2 = self.team2_output[(self.team2_output['Date'] == self.date)]

        # get stocks that fit sentiment factor (high prob of up or down)
        sentiment_stocks = today_stocks_2[(today_stocks_2['Prob Up'] > sentiment)
                                          | (today_stocks_2['Prob Down'] > sentiment)]

        # stocks that fit in both factors are the ones we will invest in
        stocks_to_invest = sentiment_stocks[sentiment_stocks['Ticker'].isin(quality_stocks)]

        # Thinking to use this to replace confidence intervals in Black Litterman
        stocks_to_invest["Prob"] = stocks_to_invest.apply(lambda row: max(row["Prob Up"], row["Prob Down"]), axis=1)

        return stocks_to_invest


def main(date, quality, sentiment):
    # ---- LATER ON PASS THIS AS PARAMETERS ----
    # Mock team1's output
    # Data frame containing Date and Tickers
    team1_output = pd.DataFrame(data={'Date': ['14-12-19', '15-12-19'],
                                      'AAPL': [1000, 700],
                                      'GOOG': [1000, 500]})
    # Mock team2's output
    # Data frame containing Date, Tickers, Probability of Up and Down
    team2_output = pd.DataFrame(data={'Date': ['14-12-19', '14-12-19', '15-12-19'],
                                      'Ticker': ['AAPL', 'GOOG', 'MSFT'],
                                      'Prob Up': [0.9, 0.2, 0.9],
                                      'Prob Down': [0.1, 0.8, 0.1]})
    # -------------------------------------------
    inputs = Team3(team1_output, team2_output, date)
    # Filter stocks based on quality factor and sentiment factor

    stocks_to_invest = inputs.filter(quality, sentiment)

    # Added Equal Weights
    stocks_to_invest = inputs.equalweight(stocks_to_invest)
    print("-" * 80)
    print("Equal Weight Allocation\n")
    print(stocks_to_invest)

    table = inputs.getclose(stocks_to_invest)

    returns = table.pct_change()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_portfolios = 25000
    risk_free_rate = 0.0178

    # Maximum Sharpe Ratio Portfolio Allocation
    # Minimum Volatility Portfolio Allocation
    # adapted from https://towardsdatascience.com/efficient-frontier-portfolio-optimisation-in-python-e7844051e7f
    inputs.display_simulated_ef_with_random(stocks_to_invest, table, mean_returns, cov_matrix, num_portfolios,
                                            risk_free_rate)

    return


if __name__ == "__main__":
    date = '14-12-19'
    quality = 500
    sentiment = 0.75
    main(date, quality, sentiment)
