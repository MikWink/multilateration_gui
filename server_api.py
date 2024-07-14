from promql_http_api import PromqlHttpApi
import pytz
from datetime import datetime, timedelta
import base64

from requests.auth import HTTPBasicAuth
class ServerApi:
    def __init__(self):
        self.url = 'https://os-statistics.entras.iis.fhg.de/'
        self.username = 'inserter'
        self.password = 'BoIAmR9VfRgVh54oTDDIOKwvUEkj56Xr'
        self.encoded_credentials = base64.b64encode(f'{self.username}:{self.password}'.encode()).decode()
        self.api = PromqlHttpApi(self.url, headers={'Authorization': f'Basic {self.encoded_credentials}'})
        self.tz = pytz.timezone('Europe/Berlin')

    def get(self, metric, start, end, step="10m"):
        q = self.api.query_range(metric, start, end, step)
        q()
        promql_response = q.response
        http_response = promql_response.response
        print(f'HTTP response status code  = {http_response.status_code}')
        print(f'HTTP response encoding     = {http_response.encoding}')
        print(f'PromQL response status     = {promql_response.status()}')
        print(f'PromQL response data       = {promql_response.data()}')
        print(f'PromQL response error type = {promql_response.error_type()}')
        print(f'PromQL response error      = {promql_response.error()}')
        return promql_response.data()







