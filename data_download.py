import yfinance

def download_data():
    #download data needed
    data_finance_spy = yfinance.download("SPY", start="2010-01-01", end="2024-12-31")
    data_finance_vix = yfinance.download("^VIX", start="2010-01-01", end="2024-12-31")

    # flatten MultiIndex columns
    data_finance_spy.columns = data_finance_spy.columns.get_level_values(0)
    data_finance_vix.columns = data_finance_vix.columns.get_level_values(0)

    # save data in data folder
    data_finance_spy.to_csv("data/spy.csv")
    data_finance_vix.to_csv("data/vix.csv")

if __name__ == "__main__":
    download_data()
    print("END")
