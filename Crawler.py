# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 19:53:28 2019

@author: jason
"""

import time
from datetime import datetime
from bs4 import BeautifulSoup
from operator import itemgetter
import requests
import sys

def crawl():     
    Start = time.time() 
    payload = { 'from':'/bbs/Beauty/M.1514740613.A.FF1.html', 'yes': 'yes' }
    rs = requests.session()
    res = rs.post("https://www.ptt.cc/ask/over18", data=payload)
    all_popular = open('all_popular.txt', 'w', encoding = 'utf-8')
    numberli = list(range(2324,2603))+list(range(2615,2758)) #numberli = list(range(2324, 2758))
    all_article = open('all_articles.txt', 'w', encoding = 'utf-8') 
    
    for index in numberli:    
        url = 'https://www.ptt.cc/bbs/Beauty/index' + str(index) + '.html' 
        r = rs.get(url)
        time.sleep(0.1) 
        soup = BeautifulSoup(r.text, 'html.parser')   
        articles = soup.find_all('div', 'r-ent') 
    
        for article in articles:             
                if index < 2330 and article.find('div','date').string[0] == "1":
                    continue
                href = str(article.find('a')['href'])
                title = article.find('a').text
                date = article.find('div','date').string
                date_str = date[1] + date[3] + date[4]
                if date[0] != " ":
                    date_str = date[0] + date_str
                all_article.write(date_str + "," + title + ",https://www.ptt.cc" + href + "\n")
                if article.find('div','nrec').string == "爆":
                    all_popular.write(date_str + "," + title + ",https://www.ptt.cc" + href + "\n")    
    rs.close()
    all_article.close()
    all_popular.close()   
    End = time.time() 
    print("Total %f sec" % (End - Start)) 
    
#crawl()

def push(start_date, end_date):
    Start = time.time()
    rs = requests.session()
    payload = {'from':'/bbs/Beauty/index2323.html','yes':'yes'}
    rs.post("https://www.ptt.cc/ask/over18", data = payload)
    all_article = open("all_articles.txt", encoding = 'utf8')
    likelist = {}
    boolist = {}
    for line in all_article:
        date = line[:line.find(",")]
        if int(date) < int(start_date):
            continue
        if int(date) > int(end_date):
            break
        url = line[line.rfind(",") + 1:].split()[0]
        soup = BeautifulSoup(rs.get(url).text, 'html.parser')
        pushes = soup.find_all('div','push')
        for data in pushes:
            if data.find('span','f3 hl push-userid'):
                user_id = data.find('span','push-userid').string
                pushtag = data.find('span','push-tag').string
                if pushtag == "推 ":
                    if user_id in likelist:
                        likelist[user_id] -= 1
                    else:
                        likelist[user_id] = -1
                elif pushtag  == "噓 ":
                    if user_id in boolist:
                        boolist[user_id] -= 1
                    else:
                        boolist[user_id] = -1
        time.sleep(0.1)
    pushfile = open('push[' + str(start_date) + '-' + str(end_date) + '].txt', "w", encoding = 'utf8')
    pushfile.write("all like: " + str(-sum(likelist.values())) + "\n")
    pushfile.write("all boo: " + str(-sum(boolist.values())) + "\n")
    likelist = sorted(likelist.items(), key = itemgetter(1,0))
    boolist = sorted(boolist.items(), key = itemgetter(1,0))
    for rank in range(10):
        pushfile.write("like #" + str(rank + 1) + ": " + str(likelist[rank][0]) + " " + str(-likelist[rank][1]) + "\n")
    for rank in range(10):
        pushfile.write("boo #" + str(rank + 1) + ": " + str(boolist[rank][0]) + " " + str(-boolist[rank][1]) + "\n")
    pushfile.close()
    End = time.time() 
    print("Total %f sec" % (End - Start)) 
 
#push(start_date = 101, end_date = 1231)  

def popular(start_date, end_date):
    tStart = time.time()
    FormData = {'from':'/bbs/Beauty/index.html', 'yes':'yes'} 
    OverAgeUrl = 'https://www.ptt.cc/ask/over18' 
    session = requests.session() 
    session.post(OverAgeUrl, data = FormData) 
    allpopular = open('all_popular.txt', 'r', encoding = 'utf-8')  
    content_popular = allpopular.readlines()
    allpopular.close()
    allpopular = []
    for item in content_popular:
        allpopular.append(item.split('\n')[0]) 
    popular_count = 0 
    FilenameExtension = ['jpg', 'jpeg', 'png', 'gif'] 
    pic_href = []
    for item in allpopular:
        date = int(item.split(',')[0]) 
        if (date - start_date) * (date - end_date) <= 0: 
            popular_count += 1
            temp = item.split(',')
            url = temp[len(temp) - 1] 
            r = session.get(url)
            time.sleep(0.1) 
            content = r.text
            soup = BeautifulSoup(content, 'html.parser')
            a = soup.find_all('a') 
            for item2 in a:
                a_href = item2['href'].lower() 
                a_href_split = a_href.split('.') 
                if a_href_split[len(a_href_split) - 1] in FilenameExtension: 
                     pic_href.append(item2['href'])
    popularfile = open('popular[' + str(start_date) + '-' + str(end_date) + '].txt', 'w', encoding = 'utf-8')
    print('number of popular articles: %d'%popular_count, file = popularfile)
    for item in pic_href:
        print(item, file = popularfile)
    session.close()
    popularfile.close()
    tEnd = time.time() 
    print("Total %f sec" % (tEnd - tStart)) 

#popular(start_date=101, end_date=1231)  

def keyword(key, start_date, end_date):
    Start = time.time()
    rs = requests.session()
    payload = {'from':'/bbs/Beauty/index2323.html','yes':'yes'}
    rs.post("https://www.ptt.cc/ask/over18", data = payload)
    allarticle = open("all_articles.txt", encoding = 'utf8')
    keywordFile = open("keyword(" + key + ")[" + str(start_date) + "-" + str(end_date) + "].txt", "w", encoding = 'utf8')
    for line in allarticle:
        date = line[:line.find(",")]
        if int(date) < int(start_date):
            continue
        if int(date) > int(end_date):
            break
        url = line[line.rfind(",") + 1:].split()[0]
        soup = BeautifulSoup(rs.get(url).text, 'html.parser')
        main_content = soup.find('div',{'id':'main-content'})
        content = str(main_content.text)
        content = content[:content.find('--\n※ 發信站: 批踢踢實業坊')]
        if content.find(key) == -1:
            continue
        hrefs = main_content.find_all('a')
        for href in hrefs:
            url = str(href.text)
            url_low = url.lower()
            if url_low.endswith('.jpg') or url_low.endswith('.jpeg') or url_low.endswith('.png') or url_low.endswith('.gif'):
                keywordFile.write(url + "\n")
        time.sleep(0.1)
    keywordFile.close();
    End = time.time() 
    print("Total %f sec" % (End - Start))  
    
#keyword('正妹', start_date = 101, end_date = 1231)