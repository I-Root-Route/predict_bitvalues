import datetime

apiKey = 'A70FAF6E-FACE-4227-920F-7C93E130F6D3'
consURL = 'https://rest.coinapi.io/v1/ohlcv/'
slash = '/'
time = datetime.datetime.now().isoformat()

#https://rest.coinapi.io/v1/ohlcv/BTC/JPY/history?period_id=1MIN&time_start=2016-01-01T00:00:00&apikey=A70FAF6E-FACE-4227-920F-7C93E130F6D3

def get_coin_data(crypto, currency):
    request_url = consURL + crypto + slash + currency + slash + 'history?period_id=1DAY&time_end=' + time + '&apikey=' + apiKey
    return request_url

print(get_coin_data('BTC','JPY'))
