import sys
import requests
from bs4 import BeautifulSoup
import re
import time
import urllib3
from collections import Counter
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
aaaa=[]
bbbb=[]
cccc=[]
dddd=[]
eeee=[]
push_date_title_html=[]
total=[]
i=0
all_popular = open("all_popular.txt","w",encoding = 'utf8')
all_article=open("all_article.txt","w",encoding = 'utf8')
payload={
            'from':'/bbs/SportLottery/index.html',
            'yes':'yes'
            }
url='https://www.ptt.cc/bbs/SportLottery/index.html'
rs=requests.session()
res=rs.post('https://www.ptt.cc/ask/over18',verify=False,data=payload)
while(i<1):
    
    res=rs.get(url,verify=False)
    soup = BeautifulSoup(res.text,'html.parser')
    #print(soup)
    article = soup.select('div.title')
     
    paging=soup.select('div.btn-group-paging a')
    #print(paging)
    date=soup.select('div.date')
    
    bomb=soup.select('div.nrec')
    #print(bomb)
    author=soup.select('div.author')
    
    keyword="[公告]" 
    
    page2 = paging[1]["href"]   
    #print(page2)
    
    last='bbs/SportLottery/M.1514736182.A.E10.html'
    
    
    next_url = 'http://www.ptt.cc'+page2
    
    url = next_url
    
    pattern='\d+' 
    a=0
    for art in article:
        art1=art.text
        bbbb.append(art1.strip())
        
        if art.a==None:
            url_s='無網址'
        else :
            url_s=art.a.get('href')
        
        url_html='http://www.ptt.cc'+url_s
        cccc.append(url_html)
    for bombb in bomb:
        bomb1=bombb.text
        push_list=re.findall('[\d爆]+',bomb1)
        ya=str(push_list)
        push_list1=re.sub('\[\]','[0]',ya)
        yaaa=re.findall('[\d爆]+',push_list1)
        yaaaa=str(yaaa)
        haha=yaaaa.strip('\[').strip('\]')
        dddd.append(haha)
    for authors in author:
        author1=authors.text
        author_list=re.findall('\w+',author1)
        author_str=str(author_list)
        author_str_final=author_str.strip('\[\'').strip('\'\]')
        eeee.append(author_str_final)
    for day in date:
        day1=day.text
        date_list=re.findall(pattern,day1)
        date_num=date_list[0]+date_list[1]
        aaaa.append(date_num)
        a=a+1
    for t_num in range(a):
        if'[籃球]' in bbbb[t_num]:
                total.append(aaaa[t_num]+','+eeee[t_num]+','+bbbb[t_num]+','+cccc[t_num]+'\n')
                if '爆'in dddd[t_num]:
                    push_date_title_html.append(aaaa[t_num]+','+eeee[t_num]+','+bbbb[t_num]+','+cccc[t_num]+'\n')
        if aaaa[t_num]=='0131':         
            i+=1         
    for one in total:
        push_all_articles=one
        all_article.write(one)
    for two in push_date_title_html:
        push_all_articles=two
        all_popular.write(two)
    aaaa=[] #日期
    bbbb=[] #文章標題
    cccc=[] #url
    dddd=[] #推文數 
    eeee=[]#作者    
    date_title_html=[]
    push_date_title_html=[]
    total=[]
    time.sleep(0.5)        
    #all_articles.close()
all_popular.close()
all_article.close()









j=0
authorid_true=[]
while(j<1):

    keyword=['暴龍','76人','七六人','賽爾提克','籃網','尼克',
              '公鹿','溜馬','活賽','公牛','騎士',
              '黃蜂','魔術','熱火','巫師','老鷹',
              '金塊','雷霆','拓荒者','爵士','灰狼',
              '勇士','湖人','快艇','國王','太陽',
              '灰熊','小牛','馬刺','火箭','鵜鶘','水鳥']
    search_articles=[]
    search_popular=[]
    authorid=[]
    trouble=[]
    all_articles = open("C:/Users/jason/Desktop/Crawl/all_article.txt","r",encoding = 'utf8')
    all_author = open('all_author.txt','w',encoding='utf8')
    
    
    '''找全部作者'''
    
    for dates in all_articles.readlines():
            date=int(re.findall('\d+',dates)[0])
            search_articles.append(dates)
    for a in search_articles:
        web_url=re.findall('http[s]?://www.ptt.cc/bbs/SportLottery/\w+.\w+.\w+.\w+.html',a)
            #print(web_url[0])
        for b in web_url:
            c=b.strip()
            if c=='http://www.ptt.cc/bbs/SportLottery/M.1514762846.A.B4B.html':
                j+=1
            d=c.strip('http://www.ptt.cc')
            #print(c)
            payload={
                    'from':'/'+c,
                    'yes':'yes'
                    }
            rs=requests.session()
            res=rs.post('https://www.ptt.cc/ask/over18',verify=False,data=payload)
            web_res=rs.get(b,verify=False)
            soup = BeautifulSoup(web_res.text,'html.parser')
            author_list=soup.find_all(class_="article-meta-value")
            #print(web_url[0])
            content=soup.find(id="main-content").text
            target_content =  u'※ 發信站: 批踢踢實業坊(ptt.cc),'
            if content.find(target_content) != -1:
#                #print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
                contentxxx = content.split(target_content)
#                # if contentxxx[0].find(keywords) == -1:
            onoff=0
            for keywords in keyword:
                if onoff==0:
                    if keywords in contentxxx[0]:
                        if author_list==[]:
                            lalalala=1
                            trouble.append(b)
                        else:
                            onoff=1
                            for userid in author_list[0]:
                                authorid.append(userid+'\n')
                            
            for author_id in authorid:
                all_author.write(author_id)
            authorid=[]                   
all_author.close()        


k=0
while(k<1):

    keyword=['暴龍','76人','七六人','賽爾提克','籃網','尼克',
              '公鹿','溜馬','活賽','公牛','騎士',
              '黃蜂','魔術','熱火','巫師','老鷹',
              '金塊','雷霆','拓荒者','爵士','灰狼',
              '勇士','湖人','快艇','國王','太陽',
              '灰熊','小牛','馬刺','火箭','鵜鶘','水鳥']
    comment=['感謝過關','感謝大大','過三關','過一關','過兩關','全過','猛','強','超準','超強','轟','全轟','四過一','四過二','四過三','三過一','三過二']
    search_articles=[]
    search_popular=[]
    authorid_t=[]
    trouble=[]
    all_articles = open("C:/Users/jason/Desktop/Crawl/all_article.txt","r",encoding = 'utf8')
    all_author_t = open('all_author_t.txt','w',encoding='utf8')
    
    
    '''找全部準作者'''
    
    for dates in all_articles.readlines():
            date=int(re.findall('\d+',dates)[0])
            search_articles.append(dates)
    for a in search_articles:
        web_url=re.findall('http[s]?://www.ptt.cc/bbs/SportLottery/\w+.\w+.\w+.\w+.html',a)
            #print(web_url[0])
        for b in web_url:
            c=b.strip()
            if c=='http://www.ptt.cc/bbs/SportLottery/M.1514762846.A.B4B.html':
                k+=1
            d=c.strip('http://www.ptt.cc')
            #print(c)
            payload={
                    'from':'/'+c,
                    'yes':'yes'
                    }
            rs=requests.session()
            res=rs.post('https://www.ptt.cc/ask/over18',verify=False,data=payload)
            web_res=rs.get(b,verify=False)
            soup = BeautifulSoup(web_res.text,'html.parser')
            author_list_t=soup.find_all(class_="article-meta-value")
            #print(web_url[0])
            content=soup.find(id="main-content").text
            target_content =  u'※ 發信站: 批踢踢實業坊(ptt.cc),'
            if content.find(target_content) != -1:
                #print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
                contentxxx = content.split(target_content)
                # if contentxxx[0].find(keywords) == -1:
            onoff=0
            onoffc=0
            for keywords in keyword:
                if onoff==0:
                    for comments in comment:
                        if onoffc==0:
                            if keywords in contentxxx[0]:
                                if comments in contentxxx[1]:
                                    if author_list==[]:
                                        lalalala=1
                                        trouble.append(b)
                                    else:
                                        onoff=1
                                        onoffc+=1
                                        for userid in author_list_t[0]:
                                            authorid_t.append(userid+'\n')
                            
            for author_id_t in authorid_t:
                all_author_t.write(author_id_t)
            authorid=[]                   
all_author_t.close() 


TOTAL=[]
all_author_t = open('all_author_t.txt','r',encoding='utf8')

for dates in all_author_t.readlines():
        date=re.findall('\w+',dates)
        TOTAL.append(date)
xxx=date.Counter


total=[]
all_author = open('all_author.txt','r',encoding='utf8')

for dates in all_author.readlines():
        date=re.findall('\w+',dates)
        total.append(date)
yyy=date.Counter
