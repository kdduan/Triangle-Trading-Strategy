from __future__ import print_function
import re
import sys
import time
import datetime
from datetime import timedelta
import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
from six.moves import urllib
import time
import os.path
import xlwt

# This is added to prevent timeout while running file
headers={"User-Agent": "Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2049.0 Safari/537.36"}

def split_crumb_store(v):
    return v.split(':')[2].strip('"')


def find_crumb_store(lines):
    for l in lines:
        if re.findall(r'CrumbStore', l):
            return l
    print("Did not find CrumbStore")


def get_cookie_value(r):
    return {'B': r.cookies['B']}


def get_page_data(symbol):
    url = "https://finance.yahoo.com/quote/%s/?p=%s" % (symbol, symbol)
    r = requests.get(url, headers = headers)
    cookie = get_cookie_value(r)

    lines = r.content.decode('unicode-escape').strip(). replace('}', '\n')
    return cookie, lines.split('\n')


def get_cookie_crumb(symbol):
    cookie, lines = get_page_data(symbol)
    crumb = split_crumb_store(find_crumb_store(lines))
    return cookie, crumb


def get_data(symbol, start_date, end_date, cookie, crumb):
    filename = '%s.csv' % (symbol)
    url = "https://query1.finance.yahoo.com/v7/finance/download/%s?period1=%s&period2=%s&interval=1d&events=history&crumb=%s" % (symbol, start_date, end_date, crumb)
    response = requests.get(url, cookies=cookie, headers = headers)
    with open (filename, 'wb') as handle:
        for block in response.iter_content(1024):
            handle.write(block)


def get_now_epoch():
    return int(time.time())


def download_quotes(symbol):
    start_date = 0
    end_date = get_now_epoch()
    cookie, crumb = get_cookie_crumb(symbol)
    get_data(symbol, start_date, end_date, cookie, crumb)

# Build the cookie handler
cookier = urllib.request.HTTPCookieProcessor()
opener = urllib.request.build_opener(cookier)
urllib.request.install_opener(opener)

# Cookie and corresponding crumb
_cookie = None
_crumb = None

# Headers to fake a user agent
_headers={
        'User-Agent': 'Mozilla/5.0 (X11; U; Linux i686) Gecko/20071127 Firefox/2.0.0.11'
}

def _get_cookie_crumb():
        '''
        This function perform a query and extract the matching cookie and crumb.
        '''

        # Perform a Yahoo financial lookup on SP500
        req = urllib.request.Request('https://finance.yahoo.com/quote/^GSPC', headers=_headers)
        f = urllib.request.urlopen(req)
        alines = f.read().decode('utf-8')

        # Extract the crumb from the response
        global _crumb
        cs = alines.find('CrumbStore')
        cr = alines.find('crumb', cs + 10)
        cl = alines.find(':', cr + 5)
        q1 = alines.find('"', cl + 1)
        q2 = alines.find('"', q1 + 1)
        crumb = alines[q1 + 1:q2]
        _crumb = crumb

        # Extract the cookie from cookiejar
        global cookier, _cookie
        for c in cookier.cookiejar:
                if c.domain != '.yahoo.com':
                        continue
                if c.name != 'B':
                        continue
                _cookie = c.value


def load_yahoo_quote(ticker, begindate, enddate, info = 'quote', format_output = 'list'):
        '''
        This function load the corresponding history/divident/split from Yahoo.
        '''
        # Check to make sure that the cookie and crumb has been loaded
        global _cookie, _crumb
        if _cookie == None or _crumb == None:
                _get_cookie_crumb()

        # Prepare the parameters and the URL
        tb = time.mktime((int(begindate[0:4]), int(begindate[4:6]), int(begindate[6:8]), 4, 0, 0, 0, 0, 0))
        te = time.mktime((int(enddate[0:4]), int(enddate[4:6]), int(enddate[6:8]), 18, 0, 0, 0, 0, 0))

        param = dict()
        param['period1'] = int(tb)
        param['period2'] = int(te)
        param['interval'] = '1d'
        if info == 'quote':
                param['events'] = 'history'
        elif info == 'dividend':
                param['events'] = 'div'
        elif info == 'split':
                param['events'] = 'split'
        param['crumb'] = _crumb
        params = urllib.parse.urlencode(param)
        url = 'https://query1.finance.yahoo.com/v7/finance/download/{}?{}'.format(ticker, params)
        #print(url)
        req = urllib.request.Request(url, headers=_headers)

        # Perform the query
        # There is no need to enter the cookie here, as it is automatically handled by opener
        
        f = urllib.request.urlopen(req)
        alines = f.read().decode('utf-8')
        #print(alines)
        if format_output == 'list':
                return alines.split('\n')
        
        if format_output == 'dataframe':
                nested_alines = [line.split(',') for line in alines.split('\n')[1:]]
                cols = alines.split('\n')[0].split(',')
                adf = pd.DataFrame.from_records(nested_alines[:-1], columns=cols)
                return adf


symbol_folder = r'/home/kdduan/Documents/Projects/Triangle Strategy/Market Database/' 

# Write initial file of ticker information
def write_csv_all(ticker_list):
    for ticker in ticker_list:
        download_quotes(ticker)
        print("Downloaded: "+ticker)
        
# Load the quote information -- returns a list of strings
def load_quote(ticker, start_date, end_date):
        return load_yahoo_quote(ticker, start_date, end_date)


# Daily update for the larger database
def update_csv(ticker_list):
    
    
    # Get today's date
    date_today = datetime.datetime.now() - timedelta(1)
    
    # Since this updates every day at 9:15, the only available information that hasn't been added
    # is yesterday's
    last_update_date = date_today - timedelta(5465)
    date_today = date_today.strftime("%Y%m%d")
    last_update_date = last_update_date.strftime("%Y%m%d")
    print(last_update_date, date_today)
    
    # Update each ticker
    for ticker in ticker_list:
        # Get DF from non-updated spreadsheet
        original_df = pd.read_csv(symbol_folder + str(ticker)+".csv")   
        # Create DF for new info
        print(last_update_date, date_today)
        information_list = load_yahoo_quote(ticker, last_update_date, date_today)
        #print(information_list)
        data = [x.split(",") for x in information_list] # Split each string into list into its own list
        new_df = pd.DataFrame(data[1:], columns=data[0]) 

        # Combine the two
        updated_df = original_df.append(new_df, ignore_index = True)
        # Drop duplicates -- these exist because subtracting 3 days rather than 1 to account for weekends
        updated_df = updated_df.drop_duplicates(subset = ['Date'], keep = 'last')
        
        updated_df = pd.DataFrame(updated_df, columns = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"])
        updated_df = updated_df.dropna()
        filepath = symbol_folder + str(ticker) + ".csv"
        if os.path.isfile(symbol_folder + filepath):
            # Delete the older one
            os.remove(filepath)
        
        # Write combined to CSV
        updated_df.to_csv(filepath, index = False)
    
# if you updated or created at the wrong time (during market hours) delete that entry
def fix_data(ticker_list):
    date_today = datetime.datetime.now() 
    date_today = date_today.strftime("%m/%d/%Y")
    for ticker in ticker_list:
        print(ticker)
        df = pd.read_csv(symbol_folder + str(ticker)+".csv") 
        df.Date = pd.to_datetime(df.Date)
        df = df[df.Date != date_today]
        os.remove(symbol_folder + str(ticker) + ".csv")
        df = pd.DataFrame(df, columns = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"])
        df.to_csv(symbol_folder + str(ticker) + ".csv", index = False)

# If you want to add more symbols (not in S&P 500), input the Excel file with those symbols
# IMPORTANT: the new file must have a column with the label "Ticker" containing all the symbols
def add_symbols(file):
    # Get list of original symbols
    original_list = get_list(ticker_file)
    # Get list of new symbols
    new_list_df = pd.read_excel(file)
    new_list = list(new_list_df['Ticker'])
    # Add new symbols to ticker list if they're not already in the list
    for new_symbol in new_list:
        if new_symbol not in original_list:
            original_list.append(new_symbol)
    # Create dataframe
    updated_df = pd.DataFrame({'Ticker': original_list})
    # Remove the previous files
    os.remove(ticker_file)
    os.remove(file)
    # Create the updated file
    updated_df.to_excel(ticker_file)



def get_list(url):
    # Get S&P list from Wikipedia site
    website_url = requests.get(url, verify = False).text 
    soup = BeautifulSoup(website_url,'lxml')
    soup = soup.prettify()
    
    # Get the 1st table on page to make into dataframe
    df = pd.read_html(soup)[0]
    #print (df)
    #print(df["Symbol"])
    
    # Make list out of first column (ticker symbol)
    ticker_list = list(df["Symbol"])
    ticker_list = ticker_list[1:506]

   
    return ticker_list


# Main function -- updates if the file for the symbol already exists, creates a new one file if it doesn't
def main(ticker_list):
    print(ticker_list)
    ticker_list.remove("BRK.B")
    ticker_list.remove("BF.B")
    ticker_list.append("BRK-B")
    ticker_list.append("BF-B")
    #print(ticker_list)
    for ticker in ticker_list:
        #time.sleep(2)
        error_list = ['RMD','ABMD']
        if ticker in error_list :
            continue
        
        filepath = symbol_folder + str(ticker) + ".csv"
        if os.path.isfile(filepath):
            print("UPDATING", ticker)
            update_csv([ticker])
        else:
            print("WRITING", ticker)
            write_csv_all([ticker])
            fix_data([ticker])
            
if __name__ == '__main__':
    main(get_list('https://en.wikipedia.org/wiki/List_of_S&P_500_companies'))
