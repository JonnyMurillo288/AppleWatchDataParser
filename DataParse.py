from xml.etree import ElementTree as ET
import pandas as pd
import numpy as np
<<<<<<< HEAD
import datetime
import os
import time
import re
import sys
from pathlib import Path
import sqlite3
from multiprocessing.pool import ThreadPool
from collections import defaultdict
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
=======
import os
from pathlib import Path
import sqlite3
>>>>>>> 133b804febe13d4f04ea20f6a85eedc9e06be932


CURR = Path.cwd()


class Watch:
    """
    CLASS for parsing through and collecting data for each apple watch workout
    RUN one at a time with each file and return a DF with stats
    """

    def __init__(self,workout_file,export_file):
        self.workout_file = workout_file
        self.export_file = export_file
<<<<<<< HEAD


    def workout_tree(self,date):
        """ Parse through an XML tree for the workout file
=======
        # self.df_stats = self.create_df(self.get_tree())

    def workout_tree(self,workout_date=None):
        """ Parse through an XML tree of data
>>>>>>> 133b804febe13d4f04ea20f6a85eedc9e06be932
        RETURN: dictionary of stats (TIME, LONGITUDE, LATITUDE, SPEED)
        """
        workout_stats = {}
        tree = ET.parse(self.workout_file)
        root = tree.getroot()

        print('Parsing through workout XML file')
        for child in root: #accesses metadata and trk
            for trkseg in child: #accesses name and trkseg
                for seg in trkseg: #accesses trkpt, seg.attrib is longitud and latitude
                    for i in seg: #inside of trkpt get ele, time and extensions (exten contains speed hAcc and vAcc)
                        if i.tag[-4:] == 'time':
                            time = i.text[:-1].split('T')
                            time = ' '.join(time)
<<<<<<< HEAD
                            time = str(datetime.datetime.fromisoformat(time) - datetime.timedelta(hours=7))
                            if date not in time:
                                continue
=======
>>>>>>> 133b804febe13d4f04ea20f6a85eedc9e06be932
                        for x in i:
                            if x.tag[-5:] == 'speed':
                                speed = x.text
                                workout_stats[time] = [seg.attrib,speed]
        return workout_stats

<<<<<<< HEAD
    def parseRecords(self,record):
        """ Parse through the export file data to find the date of the workout 
            and return the records

        EX: HKQuantityTypeIdentifierHeartRate """

        record_stats = {}
        tree = ET.parse(self.export_file)
        root = tree.getroot()

        print('Parsing through export file to find', record)
        for child in root: #child is record 
            try:
                if child.attrib['type'] == record:
                    start = child.attrib['startDate'] 
                    record_stats[start[:-6]] = round(float(child.attrib['value']),1)
            except KeyError:
                pass
        return record_stats


    def get_data(self,date,record,*args,**kwargs):
        """ concatenates the specified data and the workout data to a df """
        t = time.time()
        conn = sqlite3.connect('Workout_data.db')
        c = conn.cursor()
        stats = self.workout_tree(date)
        wo_stats = {k:stats[k] for k in stats.keys() if date in k}
        stats = self.parseRecords(record)
        hr_stats = {k:stats[k] for k in stats.keys() if date in k}
        last_hr = 0
        if len(hr_stats) == 0 or len(wo_stats) == 0:
            print("Empty hr_stats/wo_stats, returning")
            return True
        i = Interpolator(hr_stats,wo_stats)
        print("Interpolated Time:",time.time() - t)
        for dt,v in wo_stats.items(): # {date, ({lon:__, lat:__}, speed)}
            date = re.sub("-","_",dt[:10])
            create = f"CREATE TABLE IF NOT EXISTS Workouts_{date} (timestamp TEXT, heartrate FLOAT, longitude FLOAT, latitude FLOAT, speed FLOAT)"
            c.execute(create)
            q = f"INSERT INTO Workouts_{date} VALUES (?,?,?,?,?)"
            try:
                hr = round(i.interpolated[dt],4)
            except KeyError:
                hr = None
                pass
            loc = v[0]
            speed = v[1]
            if hr == np.nan:
                hr = last_hr
            last_hr = hr
            c.execute(q,(dt,hr,loc['lat'],loc['lon'],speed))
        print("This took",time.time() - t,"Seconds")
        conn.commit()
        del i
        return True


class Interpolator:

    def __init__(self,hr_stats,wo_dates):
        # ls = [k for k in hr_stats.keys()]
        # for i in range(len(ls)-1):
            # print("Time Between HR Records")
            # print(datetime.datetime.fromisoformat(ls[i+1]) - datetime.datetime.fromisoformat(ls[i]))
        print(len(hr_stats),"Records for heartrate")
        df = pd.DataFrame(hr_stats.items(),columns=['times','heartrate']).set_index('times')
        wo_df = pd.DataFrame(wo_dates.items(),columns=['times','  '])['times']
        wo = [x for x in pd.date_range(start=wo_df.min(),
                            end=wo_df.max(),
                            freq='1S')]
        df_index = pd.DataFrame(index=pd.date_range(start=df.index.min(),
                                                  end=df.index.max(),
                                                  freq='1S'))
        df_index = df_index.drop([w for w in df_index.index if w not in wo])
        new_df = df.join(df_index,how='right')
        self.interpolated = self.interpolate(new_df)['heartrate']
    
    def interpolate(self,df):
        """ Interpolate the heartrate from a range of stats """
        try:
            ridge = Ridge()
            int_hr = df.interpolate(method='slinear').dropna()['heartrate']
            df = df.fillna(method='pad').dropna().reset_index()
            X,_,y,_ = train_test_split(df[['index']],df[['heartrate']],test_size=.2)
            ridge.fit(X,y)
            df['heartrate'] = (ridge.predict(df[['index']].values.astype(float)) + int_hr + df['heatrate']) / 3 # Mean of the interpolated, padded, and predicted
            df['index'] = pd.to_datetime(df['index']).astype(str) # change index to str datetime for get data method
            df = df.set_index('index')
            return df.to_dict()
        except ValueError:
            return {'heartrate':{None:np.nan}}


#=============================================================#

def heartrate():
    path = sys.argv[1]
    exp_file = path + '/export.xml'
    watch = Watch(None,exp_file)
    tree = watch.parseRecords('HKQuantityTypeIdentifierHeartRate')
    breakpoint()


def main(ident):
    path = sys.argv[1]
    workout_path = path + '/workout-routes'
    wo_files = sorted(os.listdir(workout_path))
    file_path = wo_files[1:]
    exp_file = path + '/export.xml'

    for f in file_path: # Loop through the files in the workout routes folder
        date = f[6:-11]
        if date[-1] == '-': 
            date = f[6:-12]
        if date[-1] == '_':
            date = f[6:-12]

        print('\nstarting for:', date)
        watch = Watch(workout_path + '/' + f,exp_file)
        watch.get_data(date,ident)

def displayError():
    print("run format:\npython3 DataParse.py AppleExportFolder Identifier")
    print("Copy and paste one of the following as Identifier:\n")
    print("HKQuantityTypeIdentifierBasalEnergyBurned")
    print("HKQuantityTypeIdentifierHeartRate")
    print("HKQuantityTypeIdentifierStepCount")
    print("HKQuantityTypeIdentifierDistanceWalkingRunning")
    print("HKQuantityTypeIdentifierWalkingSpeed")
    print("HKQuantityTypeIdentifierWalkingStepLength")
    
if __name__ == '__main__':
    identifiers = ['HKQuantityTypeIdentifierBasalEnergyBurned','HKQuantityTypeIdentifierHeartRate'\
                   'HKQuantityTypeIdentifierDistanceWalkingRunning','HKQuantityTypeIdentifierStepCount'\
                   'HKQuantityTypeIdentifierWalkingSpeed','HKQuantityTypeIdentifierWalkingStepLength']
    try:
        if sys.argv[2] not in identifiers:
            displayError()
            sys.exit(1)
    except IndexError:
        displayError()
        sys.exit(1)

    if sys.argv[2] == 'HKQuantityTypeIdentifierHeartRate':
        heartrate()
    main(sys.argv[2])
=======
    def heartrate_tree(self,workout_date):
        """ Parse through the export file data to find the date of the workout 
        and return the heartrate with the given times """
        from collections import defaultdict

        heartrate_stats = {}
        tree = ET.parse(self.export_file)
        root = tree.getroot()

        print('Parsing through export file to find heart rate')
        for child in root: #child is record 
            try:
                if child.attrib['type'] == 'HKQuantityTypeIdentifierHeartRate':
                    start = child.attrib['startDate'] 
                    if start[:-15] == workout_date:
                        heartrate_stats[start[:-6]] = child.attrib['value']
            except KeyError:
                pass
        return heartrate_stats

    def create_workout_df(self,dict_of_stats):
        """
        RETURN: df with stats
        """

        df = pd.DataFrame.from_dict(dict_of_stats,orient='index',columns=['location','speed'])
        speed = df['speed'] #separate speed
        df = df['location'].apply(pd.Series) # let lon and lat separate into new col, this cuts off speed
        df = df.join(speed) # add speed back
        return df
    
    def create_heartrate_df(self,dict_of_stats):
        df = pd.DataFrame.from_dict(dict_of_stats,orient='index',columns=['HeartRate'])
        return df

    def save_db(self,df,db_name,table_name,path=CURR):
        """
        PARAMS:
        path: path to db, default cwd()
        db_name: name of db
        table_name: name of table, use as date of workout
        """
        print('Saving to DB')
        conn = sqlite3.connect(f'{path}/{db_name}.db')

        df.to_sql(name=table_name,con=conn,if_exists='replace')


def main():
    path = '/Users/jonnymurillo/Desktop/apple_health_export/workout-routes'
    p = sorted(os.listdir('/Users/jonnymurillo/Desktop/apple_health_export/workout-routes'))
    file_path = p[1:]
    exp_file = '/Users/jonnymurillo/Desktop/apple_health_export/export.xml'


    for f in file_path:
        print('\nstarting for:', f[6:-11])
        watch = Watch(path + '/' + f,exp_file)
        hr_stats = watch.heartrate_tree(f[6:-11])
        wo_stats = watch.workout_tree()
        hr_df = watch.create_heartrate_df(hr_stats)
        wo_df = watch.create_workout_df(wo_stats)
        # Now join the dataframes by time
        df = pd.concat([hr_df,wo_df],axis=1).reset_index().fillna(np.nan)
        df = df.rename(columns={'index':'Timestamp'})
        # df = df.fillna(method='ffill')


        watch.save_db(df,db_name='Workout_data',table_name=f[6:-11])


if __name__ == '__main__':
    main()
>>>>>>> 133b804febe13d4f04ea20f6a85eedc9e06be932
