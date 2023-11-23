import pandas as pd
import requests
import calendar
from persiantools import characters
import jalali_pandas


def get_daily_price_data(symbol, start_date='1402-01-01', end_date='1395-01-01'):
    
    #search webid of symbol
    try:
        symbol_id = search_ticker_webid(symbol).loc[symbol, 'WEB-ID'].values[0]
    except:
        print('stock is not founded!')
        
    #get price history
    r = requests.get(f'http://members.tsetmc.com/tsev2/data/InstTradeHistory.aspx?i={symbol_id}&Top=999999&A=0')
    df=pd.DataFrame(r.text.split(';'))
    columns=['Date','High','Low','Final','Close','Open','Y-Final','Value','Volume','No']
    
    #split data into defined columns
    df[columns] = df[0].str.split("@",expand=True)
    df.drop(columns=[0],inplace=True)
    df.dropna(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df['jdate'] = df['Date'].jalali.to_jalali().apply(lambda x: str(x).split(' ')[0])
    df = df.set_index('jdate')
    df['Ticker'] = symbol
    df[columns[1:]] = df[columns[1:]].astype(float).astype(int)
    
    #adjust the prices
    adjusted_price = get_adjusted_price(df.iloc[::-1])
    
    #filter on date
    adjusted_price = adjusted_price[(adjusted_price['jdate'] > start_date) & (adjusted_price['jdate'] < end_date)]
    
    return adjusted_price

def get_adjusted_price(price_df):
    
    df = price_df
    df.rename({'adjClose': 'Final', 'yesterday': 'Y-Final'}, axis=1, inplace=True)
    df = df.reset_index()
    #add columns
    # df['ticker'] = characters.ar_to_fa(ticker)
    df['Weekday']=df['Date'].dt.weekday
    df['Weekday'] = df['Weekday'].apply(lambda x: calendar.day_name[x])
    # df['J-Date'] = df['date'].apply(lambda x: str(jdatetime.date.fromgregorian(date=x.date())))
    df['Final(+1)'] = df['Final'].shift(+1)          # final prices shifted forward by one day
    df['temp'] = df.apply(lambda x: x['Y-Final'] if((x['Y-Final']!=0)and(x['Y-Final']!=1000)) 
                                            else (x['Y-Final'] if((pd.isnull(x['Final(+1)']))) 
                                            else x['Final(+1)']),axis = 1)
    df['Y-Final'] = df['temp']
    df.drop(columns=['Final(+1)','temp'],inplace=True)
    #price adjusting:
    df['COEF'] = (df['Y-Final'].shift(-1)/df['Final']).fillna(1.0)
    df['ADJ-COEF']=df.iloc[::-1]['COEF'].cumprod().iloc[::-1]
    df['Adj Open'] = (df['Open']*df['ADJ-COEF']).apply(lambda x: int(x))
    df['Adj High'] = (df['High']*df['ADJ-COEF']).apply(lambda x: int(x))
    df['Adj Low'] = (df['Low']*df['ADJ-COEF']).apply(lambda x: int(x))
    df['Adj Close'] = (df['Close']*df['ADJ-COEF']).apply(lambda x: int(x))
    df['Adj Final'] = (df['Final']*df['ADJ-COEF']).apply(lambda x: int(x))
    df.drop(columns=['COEF','ADJ-COEF'],inplace=True)
    return df

def search_ticker_webid(name):
    page = requests.get(f'http://old.tsetmc.com/tsev2/data/search.aspx?skey={name}')
    data = []
    for i in page.text.split(';') :
        try :
            i = i.split(',')
            data.append([i[0],i[1],i[2],i[7],i[-1]])
        except :
            pass
    data = pd.DataFrame(data, columns=['Ticker','Name','WEB-ID','Active','Market'])
    data['Name'] = data['Name'].apply(lambda x : characters.ar_to_fa(' '.join([i.strip() for i in x.split('\u200c')]).strip()))
    data['Ticker'] = data['Ticker'].apply(lambda x : characters.ar_to_fa(''.join(x.split('\u200c')).strip()))
    data['Name-Split'] = data['Name'].apply(lambda x : ''.join(x.split()).strip())
    data['Symbol-Split'] = data['Ticker'].apply(lambda x : ''.join(x.split()).strip())
    data['Active'] = pd.to_numeric(data['Active'])
    data = data.sort_values('Ticker')
    data = pd.DataFrame(data[['Name','WEB-ID','Name-Split','Symbol-Split','Market']].values, columns=['Name','WEB-ID',
                        'Name-Split','Symbol-Split','Market'], index=pd.MultiIndex.from_frame(data[['Ticker','Active']]))
    data.drop(['Name-Split', 'Symbol-Split'], inplace=True, axis=1)
    return data

def get_stock_dps(stock):
    url = f"https://www.sahamyab.com/api/proxy/symbol/getSymbolExtData?v=0.1&extData=tseDPS&code={stock}&"
    payload = ""
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    }

    response = requests.request("GET", url, data=payload, headers=headers)
    data = pd.DataFrame(response.json()['result'])
    data = pd.json_normalize(data.T[0]).dropna().replace('/', '-', regex=True)
    data['symbol'] = stock
    data = data.rename({'salMali': 'year', 'soodHarSahm': 'DPS'}, axis=1)
    return data[['symbol', 'year', 'publish', 'DPS']].sort_values('year').set_index('year')
    