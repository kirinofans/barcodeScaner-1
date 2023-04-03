import requests

url = 'https://www.baidu.com'
response = requests.get(url)
print(response.status_code)