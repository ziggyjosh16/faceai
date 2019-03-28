import os
import requests
from bs4 import BeautifulSoup

def getimages(searchname,filedir,num):
    dataurl='https://www.google.com/search?ei=IKF8XOX8J9uAr7wPxKaj-Ag&yv=3&q={}&tbm=isch&vet=10ahUKEwjljPKMy-fgAhVbwIsBHUTTCI8QuT0IdygB.IKF8XOX8J9uAr7wPxKaj-Ag.i&ved=0ahUKEwjljPKMy-fgAhVbwIsBHUTTCI8QuT0IdygB&ijn=1&start={}&asearch=ichunk&async=_id:rg_s,_pms:s,_fmt:pc'
    count=1
    try: 
         os.mkdir(filedir)
    except:
        print("Directory Already Exists.")
    for i in range(0,num): #num=3 就是前三百           #q        start
        res=requests.get(dataurl.format(searchname,i*100))
        soup=BeautifulSoup(res.text, "lxml")        
        for ele in soup.select('img'):    
            imgurl=ele.get('data-src') or ele.get('src') #抓src的tag
            with open(filedir+'/'+filedir+str(count)+'.jpg','wb') as f:  
                res2=requests.get(imgurl) #抓取圖片內容   
                f.write(res2.content)
                f.close()
            count=count+1            



getimages("rihanna","rihanna",1)
getimages("beyonce", "beyonce",1)
     






