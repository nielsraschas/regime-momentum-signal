import yfinance

START_DATE = "2010-01-01"
END_DATE = "2024-12-31"

def download_data():
    #download data needed
    data_finance_spy = yfinance.download("SPY", start=START_DATE, end=END_DATE)
    data_finance_vix = yfinance.download("^VIX", start=START_DATE, end=END_DATE)
    data_finance_irx = yfinance.download("^IRX", start=START_DATE, end=END_DATE)

    # flatten MultiIndex columns
    data_finance_spy.columns = data_finance_spy.columns.get_level_values(0)
    data_finance_vix.columns = data_finance_vix.columns.get_level_values(0)
    data_finance_irx.columns = data_finance_irx.columns.get_level_values(0)

    # save data in data folder
    data_finance_spy.to_csv("data/spy.csv")
    data_finance_vix.to_csv("data/vix.csv")
    data_finance_irx.to_csv("data/irx.csv")

if __name__ == "__main__":
    download_data()
    print("END")
