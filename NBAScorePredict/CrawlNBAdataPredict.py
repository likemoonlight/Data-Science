# -*- coding: utf-8 -*-
"""
Created on Sat Jan 04 05:24:54 2020

@author: jason
"""
import sys
import requests
import re
from bs4 import BeautifulSoup
from datetime import datetime


team1 = sys.argv[1] #球隊1
team2 = sys.argv[2] #球隊2

#team1 = '老鹰' #球隊1
#team2 = '黄蜂' #球隊2

# 球隊字典
teamlist ={'老鹰':'ATL','黄蜂':'CHA','热火':'MIA','魔术':'ORL','奇才':'WAS','公牛':'CHI','骑士':'CLE','活塞':'DET','步行者':'IND','雄鹿':'MIL','篮网':'BKN','凯尔特人':'BOS','尼克斯':'NYK','76人':'PHI','猛龙':'TOR','勇士':'GSW','快船':'LAC','湖人':'LAL','太阳':'PHO','国王':'SAC','掘金':'DEN','森林狼':'MIN','雷霆':'OKC','开拓者':'POR','爵士':'UTA','独行侠':'DAL','火箭':'HOU','灰熊':'MEM','鹈鹕':'NOH','马刺':'SAS'}

#print(teamlist[team1])
#print(teamlist[team2])
a=0 ; b=0 ; c=0 ; d=0 ; e=0 ; m=0 ; g=0 ; h=0 ; n=0 ; j=0 ; k=0 ; l=0
response = requests.get("http://www.stat-nba.com/query_team.php?crtcol=date_out&order=1&QueryType=game&Team_id="+teamlist[team1]+"&TOpponent_id="+teamlist[team2]+"&PageNum=10000") #主URL
#print(response.status_code)
#爬蟲狀態
if response:
	print('Success!')
else:
	print('An error has occurred.')

soup = BeautifulSoup(response.content,'html.parser') #解析主URL,當前只爬取第一頁的内容
urls = [] #待爬取URL集合
for i in soup.find_all(href = re.compile("game/\w+.html"),target="_blank"): #匹配符合條件的標籤
	href = re.sub(r'^\.',"",i['href']) #提取標籤中href数據並去掉URL開始的點（.）
	urls.append("http://www.stat-nba.com" + href) #每個URL添加到集合
f = open(team1+'_'+team2+'.csv',"w",encoding='utf_8_sig')
f.write("時間,客隊,主隊,客隊得分,主隊得分,總分,總分單雙,第一節單雙,第二節單雙,上半場單雙,第三節單雙,第四節單雙\n")
num = 0
for url in urls:
	num += 1
	print(num)
	print(url)
	response = requests.get(url)
	if response:
		print('Success!')
	else:
		print('An error has occurred.')
	
	soup = BeautifulSoup(response.content,'html.parser')
	time = soup.find(text = re.compile(r'(\d{4}-\d{1,2}-\d{1,2})')) #比赛时间
	time = re.sub(r'\s+',"",time) #去掉时间上的空字符
	team = soup.find_all(href = re.compile('/team/'),target="_blank")
	kedui = team[1].string #客隊名稱
	#print(kedui)
	zhudui = team[3].string #主隊名稱
	#print(zhudui)
	kedui_score = soup.find_all(class_ ="score")[0].string #客隊得分
	zhudui_score = soup.find_all(class_ ="score")[1].string #主隊得分
	zongfen = int(kedui_score) + int(zhudui_score) #總分
	#print(zongfen)
	if(zongfen % 2 == 0): #判断總分單雙
		zongfendanshuang = "雙"
	else:
		zongfendanshuang = "單"      
	#print(zongfendanshuang)
	danjiescore = soup.find_all(class_ ="number")
	onescore = int(danjiescore[0].string) + int(danjiescore[4].string) #第一節總得分
	twoscore = int(danjiescore[1].string) + int(danjiescore[5].string) #第二節總得分
	thrscore = int(danjiescore[2].string) + int(danjiescore[6].string) #第三節總得分
	fouscore = int(danjiescore[3].string) + int(danjiescore[7].string) #第四節總得分
	shangbanjiescore = onescore + twoscore
	if(onescore % 2 == 0):
		danshuang1 = "雙"
	else:
		danshuang1 = "單"        
	if(twoscore % 2 == 0):
		danshuang2 = "雙"
	else:
		danshuang2 = "單"        
	if(thrscore % 2 == 0):
		danshuang3 = "雙"
	else:
		danshuang3 = "單"
	if(fouscore % 2 == 0):
		danshuang4 = "雙"
	else:
		danshuang4 = "單"        
	if(shangbanjiescore % 2 == 0): #判断上半場單雙
		bandanshuang = "雙"
	else:
		bandanshuang = "單"        
	# 寫數據	
	f.write(time+','+kedui+','+zhudui+','+str(kedui_score)+','+str(zhudui_score)+','+str(zongfen)+','+zongfendanshuang+','+danshuang1+','+danshuang2+','+bandanshuang+','+danshuang3+','+danshuang4 + "\n")
	if(zongfen % 2 == 0):a += 1 #統計單雙數量        
	else: b=b+1
	if(onescore % 2 == 0):c=c+1        
	else:d=d+1
	if(twoscore % 2 == 0):e=e+1
	else:m=m+1
	if(thrscore % 2 == 0):g=g+1        
	else: h=h+1
	if(fouscore % 2 == 0): n=n+1       
	else:j=j+1       
	if(shangbanjiescore % 2 == 0): k=k+1      
	else:l=l+1
f.close()
NBAPredict=open("NBAPredict投注單雙  %s.txt"%('%s vs %s'%(team1,team2)),"w",encoding = 'utf8')
if(a>=b): #判断投注單雙  
    NBAPredict.write('總分應該投注買雙')    
else: 
    NBAPredict.write('總分應該投注買單')
if(c>=d):
    NBAPredict.write('\n'+'\n'+'第一節總得分應該投注買雙')    
else:
    NBAPredict.write('\n'+'\n'+'第一節總得分應該投注買單')
if(e>=m):
    NBAPredict.write('\n'+'\n'+'第二節總得分應該投注買雙') 
else:
    NBAPredict.write('\n'+'\n'+'第二節總得分應該投注買單') 
if(g>=h):
    NBAPredict.write('\n'+'\n'+'第三節總得分應該投注買雙')    
else: 
    NBAPredict.write('\n'+'\n'+'第三節總得分應該投注買單')
if(n>=j): 
    NBAPredict.write('\n'+'\n'+'第四總得分應該投注買雙')    
else: 
    NBAPredict.write('\n'+'\n'+'第四節總得分應該投注買單') 
if(k>=l): 
    NBAPredict.write('\n'+'\n'+'上半場應該投注買雙')    
else: 
    NBAPredict.write('\n'+'\n'+'上半場總得分應該投注買單') 
NBAPredict.close()       





