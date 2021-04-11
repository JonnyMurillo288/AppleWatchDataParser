from xml.etree import ElementTree as ET
import pandas as pd
import numpy as np
import os
from pathlib import Path
import sqlite3


CURR = Path.cwd()


class Watch:
    """
    CLASS for parsing through and collecting data for each apple watch workout
    RUN one at a time with each file and return a DF with stats
    """

    def __init__(self,workout_file,export_file):
        self.workout_file = workout_file
        self.export_file = export_file
        # self.df_stats = self.create_df(self.get_tree())

    def workout_tree(self,workout_date=None):
        """ Parse through an XML tree of data
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
                        for x in i:
                            if x.tag[-5:] == 'speed':
                                speed = x.text
                                workout_stats[time] = [seg.attrib,speed]
        return workout_stats

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