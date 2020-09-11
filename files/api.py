import datetime

apiKey = 'ご自身でcoinAPIなどのサイトでAPIキーを取得してください(無料でできます)'
consURL = 'https://rest.coinapi.io/v1/ohlcv/'
slash = '/'
time = datetime.datetime.now().isoformat()


def get_coin_data(crypto, currency):
    request_url = consURL + crypto + slash + currency + slash + 'history?period_id=1DAY&time_end=' + time + '&apikey=' + apiKey
    return request_url

print(get_coin_data('BTC','JPY'))
