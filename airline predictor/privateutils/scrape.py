"""
pip install selenium
"""

from selenium import webdriver
from bs4 import BeautifulSoup
import urllib.request as req
import time
import pandas as pd
import os

AIRLINE_CODE = 'OZ'
AIRLINE_NUMBER = '9355'

start_page = 2 #2

CSV_FILENAME = AIRLINE_CODE+'.csv'
#PAGE = 'http://www.airliners.net/search?photoCategory=12&sortBy=dateAccepted&sortOrder=desc&perPage=10&display=detail'
PAGE = 'http://www.airliners.net/search?airline='+AIRLINE_NUMBER+'&photoCategory=39&sortBy=dateAccepted&sortOrder=desc&perPage=50&display=detail'
PHOTO_LOCATION = AIRLINE_CODE+'/'

"""
---- Pipeline

---- Doing


---- Done
20883 - EK - Emirates
50561 - SQ - Singapore Airlines
7677 - NH - ANA
36811 - MH - Malaysian Airlines
54791 - TP - TAP
55557 - TG - Thai
18647 - DL - Delta
45547 - QF - Qantas
34139 - KE - Korean Air
4587 - CA - Air China
15053 - CX - Cathay
9355 - OZ - Asiana
"""


"""
Load the requested HTML page from airliners.net
"""
def load_page(driver,page_number=1):
    page = PAGE
    page = page+'&page='+str(page_number)
    print(page)
    
    driver.get(page)
    html = driver.page_source
    
    soup = BeautifulSoup(html, "lxml")
    
    return soup

"""
Dump Data into CSV
"""
def dump_csv_local(csv_data):
    """
    Output to CSV on local file system
    """
    headers = ['id','airline','model','reg','msn','airport','location','datetime','photo']
    df = pd.DataFrame(csv_data, columns = headers)
    
    # Write DataFrame to csv 
    if not os.path.isfile(CSV_FILENAME):
       df.to_csv(CSV_FILENAME,header = headers)
    else: # else it exists so append without writing the header
        df.to_csv(CSV_FILENAME,mode = 'a',header=False)  


"""
Download the specified photo
"""
def download_photo(photo,photoid):
        
    remaining_download_tries = 15

    while remaining_download_tries > 0 :
        try:
            req.urlretrieve(photo, PHOTO_LOCATION+str(photoid)+".jpg")
            print("successfully downloaded: " + photo)
            time.sleep(0.1)
            break
        except:
            print("error downloading " + photo +" on trial no: " + str(16 - remaining_download_tries))
            remaining_download_tries = remaining_download_tries - 1
            continue
        else:
            break  


def is_photo_in_csv(photoid):
    if not os.path.isfile(CSV_FILENAME):
        return False
    
    df = pd.read_csv(CSV_FILENAME)
    
    return len(df[df.id==photoid])>0

"""
Scrape the relevant metadata from the HTML page
"""
def extract_data(soup):
    photo_csv = []
    for row in soup.find_all("div",{"class":"ps-v2-results-row"}):
        #print(row)
        photoid = row.find_all("div",{"class":"ps-v2-results-col-title-half-width ps-v2-results-col-title-photo-id"})[0].text.replace('\n','')
        
        aircraft_text = row.find_all("div",{"class":"ps-v2-results-col ps-v2-results-col-aircraft"})[0]
        if len(aircraft_text.find_all('a')) > 1:
            airline = aircraft_text.find_all('a')[0].text.replace('\n','').replace('\n','')
            model = aircraft_text.find_all('a')[1].text.replace('\n','').replace('\n','')
        else:
            airline = 'Nothing'
            model = aircraft_text.find_all('a')[0].text.replace('\n','').replace('\n','')          
        
        registration_text = row.find_all("div",{"class":"ps-v2-results-col ps-v2-results-col-id-numbers"})[0]
        if len(registration_text.find_all('a')) > 0:
            reg = registration_text.find_all('a')[0].text.replace('\n','').replace('\n','')
        else:
            reg = 'Nothing'
            msn = 'Nothing'
        if len(registration_text.find_all('a')) > 1:
            msn = registration_text.find_all('a')[1].text.replace('\n','').replace('\n','')    
        else:
            msn = 'Nothing'
        
        locdate_text = row.find_all("div",{"class":"ps-v2-results-col ps-v2-results-col-location-date"})[0]
        airport = locdate_text.find_all('a')[0].text.replace('\n','').replace('\n','')
        if len(locdate_text.find_all('a'))>1:
            location = locdate_text.find_all('a')[1].text.replace('\n','').replace('\n','')
        else:
            location = 'Nothing'
        if len(locdate_text.find_all('a'))>2:
            datetime = locdate_text.find_all('a')[2].text.replace('\n','').replace('\n','')
        else:
            datetime = 'Nothing'
        #Fix Bug with US
        if datetime=='USA':
            location = location+'|'+datetime
            if len(locdate_text.find_all('a'))>3:
                datetime = locdate_text.find_all('a')[3].text.replace('\n','').replace('\n','')
            else:
                datetime = 'Nothing'

        # Download airliner photo    
        photo = row.find_all("div",{"class":"ps-v2-results-photo"})[0].img['src']
        
        if not os.path.isfile(PHOTO_LOCATION+str(photoid).strip() +".jpg"):
#            print('<'+photo+'>')
#            print('<'+str(photoid).strip() +'>')
#            break
            download_photo(photo,photoid.strip())
        
        # Append CSV data if not already on it
        if not is_photo_in_csv(str(photoid)):
            photo_csv.append([photoid,airline,model,reg,msn,airport,location,datetime,photo])
        
    # Write CSV data to file        
    dump_csv_local(photo_csv)
        
    return photo_csv


if __name__ == "__main__":
    driver = webdriver.PhantomJS()
    
    #Load first page alone so I can know who many pages I need to run
    soup = load_page(driver)
    extract_data(soup)
    
    num_pages = int(soup.find_all("div",{"class":"pagination"})[0].find_all('li')[-2].text)
    
    #num_pages = 200
    
    # Last successfull run was up to (inclusive) 599
    # Next batch starts at last + 1
    
    print('MAX PAGES = '+str(num_pages))
    
    #for page in range(2,num_pages):
    for page in range(start_page,num_pages):
        soup = load_page(driver,page)
        extract_data(soup)

   
    driver.close()
    driver.quit()