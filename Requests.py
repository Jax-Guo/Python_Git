import json
from multiprocessing import Pool
import requests
from requests.exceptions import RequestException
from bs4 import BeautifulSoup
def get_one_page(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US; rv:1.9.1.5) Gecko/20091102 Firefox/3.5.5 (.NET CLR 3.5.30729)"}
        res = requests.get(url,headers = headers)
        res.encoding = 'utf-8'
        if(res.status_code == 200 or res.status_code == 304):
            return res.text
        return None
    except RequestException:
        return None

def parse_one_page(html):
    soup = BeautifulSoup(html,'lxml')
    items = soup.find_all('dd')
    for item in items:
       yield{
           "index": item.select('.board-index')[0].text,
           "image": item.select('.board-img')[0].get('data-src'),
           "title": item.select('.name')[0].text,
           "actor": item.select('.star')[0].text.strip()[3:],
           "time":  item.select('.releasetime')[0].text.strip()[5:],
           "score": item.select('.score')[0].text + item.select('.fraction')[0].text
       }

def write_to_file(content):
    with open('result.txt','a',encoding='utf-8') as f:
        f.write(json.dumps(content,ensure_ascii=False)+'\n')
        f.close()

def main(offset):
    url = "http://maoyan.com/board/4?offset=" + str(offset)
    html = get_one_page(url)
    [{print(item),write_to_file(item)} for item in parse_one_page(html)]

if __name__ == '__main__':
   # for i in range(10):
   #     main(i * 10)
   pool = Pool()
   pool.map(main,[i*10 for i in range(10)])