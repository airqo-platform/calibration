import pandas as pd
from google.cloud import bigquery
import numpy as np
import pickle
import gpflow
from datetime import datetime,timedelta
import matplotlib.pyplot as plt
from gpflow.utilities import print_summary
import download_airnow_web
import pyproj

def loaddata(bigquery_account_json):
    client = bigquery.Client.from_service_account_json(bigquery_account_json)
    try:
        df = pickle.load(open('alldata.p','rb'))
        mostrecent = df['created_at'].max()
        mostrecent = mostrecent.strftime('%Y-%m-%d')
        print("Downloading data since %s." % mostrecent)
        sql = """SELECT * FROM `airqo-250220.thingspeak.clean_feeds_pms` WHERE created_at >= DATETIME('%s')""" % mostrecent
        d = client.query(sql).to_dataframe()
        print("Data downloaded spans %s to %s." % (d['created_at'].min().strftime('%Y-%m-%d'),d['created_at'].max().strftime('%Y-%m-%d')))
        #TODO The drop_duplicates call is really slow.
        df = pd.concat([df,d]).drop_duplicates() #we are likely to have duplicates as the create_at threshold is a bit vague
    except FileNotFoundError:
        print("cache file ('alldata.p') not found. Downloading entire dataset, this may take some time.")
        sql = """SELECT * FROM `airqo-250220.thingspeak.clean_feeds_pms`"""
        df = client.query(sql).to_dataframe()
    pickle.dump(df,open('alldata.p','wb'))
    return df

def combinedatasets(df,embassydf,distfromkampalabox = 40e3):
    #Sort out timezones.
    df['created_at']=df['created_at'].apply(lambda x: pd.Timestamp(x,tz='UTC'))
    embassydf['created_at'] = embassydf['Date (LT)'].apply(lambda x: pd.Timestamp(x,tz='Africa/Kampala').astimezone('UTC'))

    #Add embassy location
    #AirQo Installation (Unit 51)
    #Lat: 0.29925, Long: 32.5927
    #US Embassy BAM (Estimate)
    #Lat: 0.299333, Long: 32.592539
    embassydf['latitude'] = 0.299333
    embassydf['longitude'] = 32.592539

    #Concatenate embassy data into main df.
    embassydf['channel_id']=-1
    embassydf['pm2_5'] = embassydf['Raw Conc.']
    df = pd.concat([df, embassydf[['created_at','channel_id','pm2_5','latitude','longitude']]])

    #Convert to northings and eastings
    epsg3857 = pyproj.Proj(init='epsg:3857') #EPSG:3857 -- WGS84 Web Mercator [used by websites]
    wgs84 = pyproj.Proj(init='EPSG:4326') #WGS 84 [used by GPS satellite system]
    kampala = pyproj.transform(wgs84,epsg3857,32.581111,0.313611)
    df['x'],df['y'] = pyproj.transform(wgs84,epsg3857,df['longitude'].tolist(),df['latitude'].tolist())

    #Only keep items that are in a box around Kampala
    df = df[(df['x']>kampala[0]-distfromkampalabox) & (df['x']<kampala[0]+distfromkampalabox) & (df['y']>kampala[1]-distfromkampalabox) & (df['y']<kampala[1]+distfromkampalabox)]
    #save the whole lot
    #pickle.dump(df,open('alldataprocessed.p','wb'))
    return df

def build_encounters(df,prox = 40,timeprox = 30):
    """
    prox = proximity in metres
    timeprox = proximity in minutes
    """
    dfs = []
    encounters = None
    for cid in df['channel_id'].unique():
        df['created_at_2']=df['created_at']
        dfs.append(df[df['channel_id']==cid])
    for i,d1 in enumerate(dfs):
        #for d2 in dfs[i+1:]:
        for j,d2 in enumerate(dfs):
            if i>=j:continue
            newdf = pd.merge_asof(d1.sort_values("created_at"),d2.sort_values("created_at"),on='created_at',tolerance=pd.Timedelta(timeprox,'minutes'),direction='nearest',suffixes=('_sensorA', '_sensorB')). \
                dropna(subset=["created_at","created_at_2_sensorA",'latitude_sensorA','longitude_sensorA', 'latitude_sensorB', 'longitude_sensorB','pm2_5_sensorA','pm2_5_sensorB'])
            #leaving this line in might make it quicker, as I compute a sqrt (somewhat unnecessarily for a lot of data)
            #newdf = newdf[(np.abs(newdf['x_sensorA']-newdf['x_sensorB'])<prox) & (np.abs(newdf['y_sensorA']-newdf['y_sensorB'])<prox)]
            newdf['dist'] = np.sqrt((newdf['x_sensorA']-newdf['x_sensorB'])**2 + (newdf['y_sensorA']-newdf['y_sensorB'])**2)
            newdf = newdf[newdf['dist']<prox]
            newdf['timedelta'] = np.abs(newdf['created_at_2_sensorA']-newdf['created_at_2_sensorB'])
            encounters = pd.concat([encounters,newdf])
        print("%d of %d (%d encounters recorded)" % (i+1,len(dfs),len(encounters)))
    return encounters

def plotsensorencounters(df,box=100,boxsensorids=None,plotsensorids=None,subplotshape=None):
    """
    For each sensor in boxsensorids, plot a subplot centred on its median location.
    Of size +/- box metres. In this box, plot all the sensors that pass nearby from
    the plotsensorids list.    
    Leave either as None to plot all the sensors.
    """
    i=0
    if boxsensorids is None:
        boxsensorids = df['channel_id'].unique()
    if plotsensorids is None:
        plotsensorids = df['channel_id'].unique()
    if subplotshape is None:
        width = np.trunc(np.sqrt(len(boxsensorids)))
        height = np.trunc(len(boxsensorids)/width+0.99)
        subplotshape = [width,height]
    #plt.figure(figsize=[subplotshape[0]*4,subplotshape[1]*4])
    for cid in boxsensorids:
        i+=1
        boxc=[0,0]
        boxc[0] = np.median(df[df['channel_id']==cid]['x'])
        boxc[1] = np.median(df[df['channel_id']==cid]['y'])
        plt.subplot(subplotshape[1],subplotshape[0],i)
        #if cid in [930428,930430]: continue

        for channel_id in plotsensorids:
            #print(channel_id)
            chandf = df[df['channel_id']==channel_id]
            chandf = chandf[(chandf['x']>boxc[0]-box) & (chandf['x']<boxc[0]+box) & (chandf['y']>boxc[1]-box) & (chandf['y']<boxc[1]+box)]
            #if channel_id in [-1,930428,930430]:
            #    dotsize = 3
            #else:
            dotsize = 3
            if len(chandf)>0:
                plt.scatter(chandf['x'],chandf['y'],dotsize,label="%d (%d)" % (channel_id,len(chandf)))
        plt.legend()
        plt.xlim([boxc[0]-box,boxc[0]+box])
        plt.ylim([boxc[1]-box,boxc[1]+box])
        plt.title(cid)
        #plt.plot(embassyxy['x'],embassyxy['y'],'x',markersize=20)
        plt.grid()
        
def getdatelist(start,delta,steps,spaces=1):
    t = start
    dates = []
    strings = []
    for i in range(steps):
        dates.append(t)
        if i%spaces==0:
            strings.append(t.strftime('%b%d'))
        else:
            strings.append("")
        if delta.days<30:
            t=t+delta
        else:
            month=t.month+1
            year = t.year
            if month>12:
                year=year+1
                month = 1
            t = pd.datetime(year,month,1)
            
    return dates,strings
 
