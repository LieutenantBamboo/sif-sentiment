import yfinance as yf
import numpy as np
import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier


#### READ FIRST ####

# This code is not complete. Reason being that BL model needs predicted returns.
# Our MVP uses probabilities. How should we go about this.

# Checklist
# [/] Filter based on Quality & Sentiment strength
# [ ] BL optimisation
# [ ] Backtesting

####################

class Team3:
    def __init__(self, team1_output, team2_output, date):
        # team 2 output is a data frame with tickers and probability of up/down
        self.team2_output = team2_output
        self.team1_output = team1_output
        self.date = date

    def get_capm_param(self, companies_ticker, riskless_domestic_cash, verbose=1):

        """
        This is a helper function for BL_Optimization function
        :param companies_ticker: a list of company tickers
        :param riskless_domestic_cash: the risk free rate
        :param verbose: a binary variable for extra comments
        :return: parameters of the CAPM model formed from the companies given
        """

        comp_returns = []
        comp_market_cap = []

        # get data for each company from Yahoo
        for comp in companies_ticker:
            if verbose == 1:
                print(comp)

            yahoo = yf.Ticker(comp)
            historical_stock = yahoo.history(period="1y", interval="1mo").pct_change(1)
            info_stock = yahoo.info
            historical_stock[comp] = historical_stock["Close"]
            df3 = historical_stock[comp].dropna(0)
            df3 = df3.loc[~df3.index.duplicated(keep='first')]
            comp_returns.append(df3)
            comp_market_cap.append(info_stock["marketCap"])

        # returns
        all_returns = pd.concat(comp_returns, 1).dropna(0)
        returns = all_returns.values.reshape(len(companies_ticker), len(all_returns))

        # market cap
        market_caps = pd.DataFrame({"Market Capitalizations": comp_market_cap}, index=companies_ticker)
        market_caps["%"] = market_caps / np.sum(market_caps)
        market_caps = market_caps["%"]

        # risk aversion rate
        risk_aversion_rate = (np.mean(all_returns.values.reshape(-1)) - riskless_domestic_cash / 12) / np.var(
            all_returns.values.reshape(-1))
        covar_matrix = np.cov(returns)

        # average returns
        mu = np.mean(all_returns)

        # efficient frontier weights
        ef = EfficientFrontier(mu, covar_matrix, weight_bounds=(-1, 1))
        raw_weights = ef.max_sharpe()
        weights_c = [[raw_weights[n]] for n in raw_weights]
        weights = pd.DataFrame(weights_c, index=companies_ticker, columns={"Pi"})
        pi = pd.DataFrame(risk_aversion_rate * np.matmul(covar_matrix, weights))

        # monthly performance
        performance_monthly = ef.portfolio_performance(risk_free_rate=riskless_domestic_cash / 12)

        return all_returns, market_caps, covar_matrix, risk_aversion_rate, weights, pi, performance_monthly

    def bl_optimization(self, companies_ticker, riskless_domestic_cash, tau, volatility_constraint=0, verbose=1):
        """
        Black-Litterman Portfolio Optimization

        """
        # Used Filter from output of team 2, so no longer need this method
        # if verbose == 1:
        #     print("Processing Sentiment Analysis and Scoring Returns-Sentiment Relationship ...")
        # predicted_returns_and_Omega = companies_in_sentiment_basket(companies_ticker)

        if verbose == 1:
            print("Sentiment Analysis and Scoring Returns-Sentiment Relationship Performed")
            print("Performing Initial Portfolio Optimization Without Bias  ...")

        all_returns, market_caps, sigma, risk_aversion_rate, weights, pi, performance = self.get_capm_param(
            companies_ticker,
            riskless_domestic_cash,
            verbose=verbose)
        if verbose == 1:
            print("Initial Optimization Performed")
            print("Performing Black-Litterman Portfolio Optimization ...")

        # SINCE WE NO LONGER GET PREDICTED RETURNS FROM TEAM 2, SHOULD WE STILL BE USING BL??
        Q = np.array([n.tolist()[0][0] for n in predicted_returns_and_Omega["Prediction returns"]]).reshape(
            len(predicted_returns_and_Omega), 1)

        Omega = np.diag(predicted_returns_and_Omega["Confidence BL"])

        sigma_inv = np.linalg.inv(sigma)
        sigma_inv_pi = np.matmul(sigma_inv, pi)

        E_R = np.matmul(np.linalg.inv(1 / tau * sigma_inv + np.linalg.inv(Omega)),
                        (1 / tau * sigma_inv_pi + np.matmul(np.linalg.inv(Omega), Q)))

        new_weights = np.matmul(sigma_inv, E_R) * 1 / risk_aversion_rate

        if volatility_constraint != 0:
            new_weights = np.sqrt(volatility_constraint) * new_weights / np.matmul(
                new_weights.values.reshape(1, len(new_weights)),
                np.matmul(sigma, new_weights.values.reshape(len(new_weights), 1)))

        new_weights = new_weights / np.sum(new_weights)
        if verbose == 1:
            print("Black-Litterman Portfolio Optimization Performed")

        return pd.DataFrame({"Companies": companies_ticker, "Returns": all_returns.mean(), "Prediction": Q.reshape(-1),
                             "Confidence Prediction": predicted_returns_and_Omega["Confidence BL"].values.reshape(-1),
                             "Pi": pi["Pi"], "E(R)": E_R["Pi"], "Markovitz Weights": weights["Pi"],
                             "CS-SIMVO Weights": new_weights["Pi"]}).set_index("Companies")

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
    print(stocks_to_invest)
    return


if __name__ == "__main__":
    date = '14-12-19'
    quality = 500
    sentiment = 0.75
    main(date, quality, sentiment)
