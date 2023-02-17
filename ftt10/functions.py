from dataclasses import dataclass
import sys, smtplib, json, urllib3
import matplotlib.pyplot as plt
from datetime import datetime
from pytrends.request import TrendReq
import numpy as np
import pandas as pd
import yfinance as yf
import tqdm.notebook as tq

urllib3.disable_warnings()


# testing
# https://stackoverflow.com/questions/20115693/correct-way-to-address-pyside-qt-widgets-from-a-ui-file-via-python
# pyside2-uic Trends.ui -o ui_mainwindow.py2
# -------------------------------- Example of two way threading ----------------------------------------
# https://stackoverflow.com/questions/35527439/pyqt4-wait-in-thread-for-user-input-from-gui/35534047#35534047
# ---------------------   Example of combining dataframes and dropping duplicates            -----------------------
# https://pandas.pydata.org/docs/reference/api/pandas.Index.drop_duplicates.html
# pyinstaller -w --clean  FollowTheTrends3.py


def run(Keywords, Parameters, TimeFrame):
    """Long-running task."""

    small_dfs = pd.DataFrame()
    for i, item in tq.tqdm(enumerate(Keywords), total=len(Keywords)):
        scaled_position = np.ceil((i / (len(Keywords) - 1)) * 100)
        timeframe = TimeFrame['1wk']
        stock_as_list = item  # RETURN [] LATER
        try:
            stock_data = interestData(stock_as_list, timeframe)
        except Exception as e:
            print(e)
            # UpdateLists({'item': e, 'the_list': 'Status'})
            # print('Could not get trends', item)
        else:
            # print('Got the trend for ', item)
            if isinstance(stock_data, pd.DataFrame) and len(stock_data) > 5:
                for symbol in item:
                    if isinstance(stock_data[symbol][len(stock_data) - 1], np.int64):
                        if np.max(stock_data[symbol]) > 2:
                            analyse_data(symbol, stock_data, Parameters)
                        # else:
                        # print('values are too small')
                    else:
                        print('no dataframe in int64 format')
            if Parameters['saveData']:
                stock_data = stock_data.iloc[:, :-1]  # remove the last column "isPartial"
                if len(small_dfs) < 2:
                    small_dfs = stock_data
                else:
                    small_dfs = pd.merge(small_dfs, stock_data, how='inner', left_index=True,
                                         right_index=True)  # merge only items with the same DateTimeIndex (losing some on the edges - but keeping it nice ..)
            else:
                print(item, "Was not able to analyze data  does not look like a dataframe :( ")
        finally:
            pass
            # print('moving to the next 5  ..')
        # print(str(scaled_position) + '%  ')
    print('job is done .. - saving results now')
    UpdateLists({'item': 'job is done .. - saving results now', 'the_list': 'Status'})

    # saving the data:
    try:
        tStamp = str(datetime.now().strftime('%Y_%m_%d_%H_%M'))
        if Parameters['saveData']:  # saving hourly data
            csv_file = Parameters['base_name'] + '/Hourly_Data/_hourly_data_' + tStamp  ############## FIX PATH
            small_dfs.to_csv(csv_file + '.csv', index=True)

    except Exception as e:
        UpdateLists({'item': e, 'the_list': 'Status'})
        print(e)
    else:
        UpdateLists({'item': 'Results saved - All done', 'the_list': 'Status'})


def analyse_data(symbol, stock_data, Parameters):
    rows = len(stock_data)
    last_value = stock_data[symbol][rows - 1]
    weekly_max = np.max(stock_data[symbol])
    relative_last_value = round(100 * (last_value / weekly_max))
    # UpdateLists(
    #     {'item': symbol + ': last value is: ' + str(relative_last_value) + '%', 'the_list': 'Status'})
    # print(symbol + ': last value is: ' + str(relative_last_value) + '%')
    if relative_last_value < Parameters['w_minimum_value']:  # ignore cases where the last value is lower then ~ 85%
        return
    # taking the Mean and STDV of the earlier data without the most recent points

    w_earlier = stock_data.head(round(Parameters['early'] * len(stock_data)))

    w_mean = round(np.mean(w_earlier[symbol]), 2)
    w_std = round(np.std(w_earlier[symbol]), 2)
    if w_std == 0:  # ignore cases where STDV is 0 ... not sure how it arrived so far - but probably represents low quality data
        return
    # print('last_value ' + str(last_value) + 'w_mean ' + str(w_mean) + 'w_std ' + str(w_std))
    w_data = list(stock_data[symbol])

    if w_data.count(0) > 5:  # ignore cases where the data is scarce
        return
    w_sigma = round((last_value - w_mean) / w_std, 1)
    w_time = list(stock_data.index)
    before_last_value = stock_data[symbol][rows - 2]

    last_two_hr = (last_value + before_last_value) / 2
    # -----------------------------  conditions for success  -----------------
    # w_zeros = w_data.count(0) < 5
    # over_minimum = relative_last_value > self.uiParameters['w_minimum_value']
    high_sigma = last_two_hr > w_mean + (Parameters[
                                             'w_winnerCoeff'] * w_std)  # lots of false alarms with only the last value very high -> taking the average of two hours to be a winner
    low_sigma = last_value > w_mean + (Parameters['w_maybeCoeff'] * w_std)

    if high_sigma:
        print("GOT a winner here. ", symbol)
        avvol = avegave_volume(symbol)
        UpdateLists(
            {'item': symbol + ':  Sigma is: ' + str(w_sigma) + ' Av. vol(M): ' + str(avvol), 'the_list': 'Winners'})
        sendMail(symbol)
        plot_winners({
            'symbol': symbol,
            'STDV': w_std,
            "mean": w_mean,
            'data': w_data,
            'timeline': w_time,
            'volume': avvol})


    elif low_sigma:
        avvol = avegave_volume(symbol)
        UpdateLists(
            {'item': symbol + ':  Sigma is: ' + str(w_sigma) + ' Av. vol(M): ' + str(avvol), 'the_list': 'Maybe'})


def avegave_volume(symbol):
    try:
        ticker_object = yf.Ticker(symbol)
        avvol = round(ticker_object.history('10d').loc[:, 'Volume'].mean() / 1000000, 2)
        # print(avvol)
    except Exception as e:
        print(e)
        avvol = 'could not get volume'
    finally:
        return avvol


def interestData(stock_as_list, timeframe):
    try:
        pytrends = TrendReq(hl='en-US', tz=360, timeout=(10, 25), retries=11, backoff_factor=0.1, requests_args={
            'verify': False})  # proxies=['https://34.203.233.13:80']  132.72.138.1:8081
        plotdata = pytrends.build_payload(stock_as_list, cat=0, timeframe=timeframe['trends'], geo='',
                                          gprop='')  # geo='US'
        plotdata = pytrends.interest_over_time()  # dataframe to hold the data
        return plotdata
    except Exception as e:
        print(e)
        # print("couldn't get google trend")
    else:
        pass
        # print("GOT the Trend correctly")


# -------------------------- FUNCTION LIST --------------------------------------------------
def UpdateLists(results):
    df = pd.DataFrame([results])
    base_name = 'drive/MyDrive/Colab Notebooks/results/NASDAQ_' + str(datetime.now().strftime('%Y_%m_%d_%H'))
    df.to_csv(base_name + '.csv', mode='a', index=False, header=False)


# ----------------------------------------------------------------------------------------------------------

def sendMail(stock):
    email_list = ['amir.golan3@gmail.com']  # , 'fredrik.andersson@changeloop.se', 'romangutin82@gmail.com']
    # =============================================================================
    # SET EMAIL LOGIN REQUIREMENTS
    # =============================================================================
    gmail_user = 'follow.the.trends2021@gmail.com'
    gmail_app_password = 'srzkfruzurjzaldq'  # 'FortuneTeller'  # GOOGLE-PASSWORD!!!!'

    # =============================================================================
    # SET THE INFO ABOUT THE SAID EMAIL
    # =============================================================================
    sent_from = gmail_user
    sent_to = email_list  # ['THE-TO@gmail.com', 'THE-TO@gmail.com']
    sent_subject = "hey, You should check out: " + stock
    sent_body = ("Hey, what's up? \n\n" + "I hope you have been well!\n" + "You should check out: " + stock + "\n"
                                                                                                              "\n"
                                                                                                              "Cheers,\n"
                                                                                                              "A\n")

    email_text = """\
From: %s
To: %s
Subject: %s

%s
    """ % (sent_from, ", ".join(sent_to), sent_subject, sent_body)

    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.ehlo()
        server.login(gmail_user, gmail_app_password)
        server.sendmail(sent_from, sent_to, email_text)
        server.close()

        print('Email sent!')
        UpdateLists({'item': 'Sent mail to members ', 'the_list': 'Status'})
    except Exception as exception:
        print("Error: %s!\n\n" % exception)
        UpdateLists({'item': 'ERROR sending e-mail', 'the_list': 'Status'})


def plot_winners(winner):
    X = winner['data']
    t = winner['timeline']
    average = [winner['mean'] for i in range(len(t))]
    # the 1 sigma upper and lower analytic population bounds
    lower_bound = winner['mean'] - winner['STDV']
    upper_bound = winner['mean'] + winner['STDV']
    fig, ax = plt.subplots(1, figsize=(12, 6))
    fig.canvas.manager.set_window_title(winner['symbol'])
    ax.plot(t, X, lw=2, label='trend of: ' + winner['symbol'], color='blue')
    ax.tick_params(axis='y', labelcolor='blue')
    ax.plot(t, average, lw=1, label='mean value', color='black', ls='--')
    ax.fill_between(t, lower_bound, upper_bound, facecolor='yellow', alpha=0.5, label='1 sigma range')
    ax.legend(loc='upper left')

    # here we use the where argument to only fill the region where the
    # walker is above the population 1 sigma boundary
    ax.fill_between(t, upper_bound, X, where=X > upper_bound, facecolor='blue', alpha=0.5)
    ax.set_xlabel('date')
    ax.set_ylabel('Google search intensity')
    ax.grid()

    try:
        marketdata = yf.download(tickers=winner['symbol'], period='1wk', interval='1m')
        if (marketdata.index.tz is not None) and len(marketdata) > 5:
            marketdata.index = marketdata.index.tz_convert(None).tz_localize(None)

        ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        color = '#CC00CC'  # 'tab:red'
        ax2.set_ylabel('stock price / $', color=color)  # we already handled the x-label with ax1
        ax2.plot(marketdata.index, marketdata['Close'], label=winner['symbol'] + ' stock price', color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(bottom=min(marketdata['Close']), top=None)
        ax2.fill_between(marketdata.index, 0, marketdata['Close'], where=marketdata['Close'] > 0, facecolor=color,
                         alpha=0.5)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        fig.savefig(fname='drive/MyDrive/Colab Notebooks/results/NASDAQ_' + winner['symbol'], dpi='300', format='png',
                    bbox_inches='tight')
    finally:
        plt.show()