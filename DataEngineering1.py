#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 17:43:22 2022

@author: Andrew1

features:
    distance to finish line (race length - distance traveled)
    distance from start (distance traveled))
    speed (MPS)
    acceleration (MPS^2)
    leader speed 
    leader acceleration
    odds (%)
    sum of odds infron 
    sum of odds behind
    distance from leader (l distance traveled - distance traveled)
    some measure of crowdedness (squared dist) - horses within grid square
    in bend / straight (binary)
    position 
    track type
    race type
"""

#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import math

import warnings

warnings.simplefilter("ignore")

#tracking
path_to_file = "/Users/Andrew1/Desktop/Kaggle/big-data-derby-2022/nyra_tracking_table.csv"

tracking = pd.read_csv(path_to_file)

#race
path_to_file = "/Users/Andrew1/Desktop/Kaggle/big-data-derby-2022/nyra_race_table.csv"

race = pd.read_csv(path_to_file)

#start
path_to_file = "/Users/Andrew1/Desktop/Kaggle/big-data-derby-2022/nyra_start_table.csv"

start = pd.read_csv(path_to_file, header=None)

start_cols = ["track_id", "race_date","race_number","program_number",
              "weight_carried","jockey","odds","position_at_finish"]

rename_dict = dict(zip(range(8),start_cols))
start = start.rename(columns=rename_dict)

#complete
path_to_file = "/Users/Andrew1/Desktop/Kaggle/big-data-derby-2022/nyra_2019_complete.csv"
complete = pd.read_csv(path_to_file, header=None,parse_dates=([1]),dtype=({3 : str}))
complete_cols = ["track_id", "race_date","race_number","program_number",
                 "trakus_index","latitude","longitude","distance_id",
                 "course_type","track_condition","run_up_distance",
                 "race_type","purse","post_time",
             "weight_carried","jockey","odds","position_at_finish"]
rename_dict = dict(zip(range(18),complete_cols))
i_col = dict(zip(complete_cols,range(18)))

complete = complete.rename(columns=rename_dict)

#Indexes where there is a tracking issue ie one horse disappears
path_to_file = "/Users/Andrew1/Desktop/Kaggle/big-data-derby-2022/bad_index.csv"
bad_indices = pd.read_csv(path_to_file).iloc[:,0]

#finished races
path_to_file = "/Users/Andrew1/Desktop/Kaggle/Derby/finished_races.csv"
finished_races = pd.read_csv(path_to_file).iloc[:,1]
#%%

def sortDF(df):
    new = df.copy()
    
   # new = new.sort_values("program_number")
    new = new.sort_values("trakus_index",kind = "mergesort")
    new = new.sort_values("race_number",kind = "mergesort")
    new = new.sort_values("race_date",kind = "mergesort")
    new = new.sort_values("track_id", kind = "mergesort")
        
    return new

def firstAndLast(df, value, column):
    found = False
    col_num = i_col[column]
    #print("hello")
    
    for i in range(len(df)):
        
        if(df.iloc[i,col_num] == value):
            
            first_row = i
            found = True
            break
        
        
    if(not found):
        print("value not found")
        return
    last_row = first_row
    for n in range(i+1, len(df),1):
        
        if(df.iloc[n,col_num] != value):
            break
        last_row = n
    
    return first_row, last_row
    
def feetPerDegree(latitude):
    eq_mi_per_lon = 69.172

    lat_radian = (math.pi  * latitude)/180
    cos_radian = math.cos(lat_radian)

    mi_per_lon= eq_mi_per_lon * cos_radian

    mi_per_lat = 69

    feet_per_longitude = mi_per_lon * 5280
    feet_per_latitude = mi_per_lat * 5280
    
    return feet_per_longitude, feet_per_latitude
    
def coordinatePlane(df_old):
    df = df_old.copy()
    
    for track in df.track_id.unique():
        track_data = df.loc[df.track_id == track]
        lon_mult, lat_mult = feetPerDegree(df.latitude.mean())
        
        track_data["longitudeFT"] = track_data.longitude - min(track_data.longitude)
        track_data["longitudeFT"] = track_data["longitudeFT"]  * lon_mult
            
        track_data["latitudeFT"] = track_data.latitude - min(track_data.latitude)
        track_data["latitudeFT"] = track_data["latitudeFT"] * lat_mult
        
        df.loc[track_data.index, ["longitudeFT","latitudeFT"]] = track_data[["longitudeFT","latitudeFT"]]
        
    return df

def furlongToFeet(df,column):
    new = df.copy()
    
    new["distanceFT"] = new[column] * 6.6
    
    return new

def oddsToProb(df,column):
    new = df.copy()
    
    new["winProb"] = 100 / (df[column] + 100)
    
    return new

def winBin(df, column):
    new = df.copy()
    
    new["won"] = 0
    new.loc[new[column] == 1, "won"] = 1
    
    return new

def rotateTrack(df, theta):
    new = df.copy()
    
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)
    
    new["x_coord"] = (new["longitudeFT"] * cos_theta) + (new["latitudeFT"] * sin_theta)
    new["y_coord"] = (-1 * new["longitudeFT"] * sin_theta) + (new["latitudeFT"] * cos_theta)

    return new
def orientTrack(df,track):
    new = df.copy()
    
    best_thetas = {"AQU" : (math.pi / 180) * 71,
                   "BEL" : (math.pi / 180) * 161,
                   "SAR" : (math.pi / 180) * 27}
    

    theta = best_thetas[track]
    
    new = rotateTrack(new, theta)
    
    return new, theta

def rotationPoint(race):
    new = race.copy()
    track = new.track_id.iloc[0]
    
    new, theta = orientTrack(new, track)
    
    point_y = new.y_coord.mean()
    
    point_x = new.iloc[-1].x_coord
    
    point_df = pd.DataFrame({"longitudeFT":[point_x],  "latitudeFT":[point_y]})
    
    point_df = rotateTrack(point_df, -1*theta)
    
    point = (float(point_df.x_coord),float(point_df.y_coord))
    
    return point
    
def euclidianDistance(x1,y1, x2,y2):
    distance = (((x1 - x2) **2) +\
               ((y1 - y2) **2))**.5

    return distance

def angleCalc(side_a,side_b,side_c):
    angle_a = np.arccos((side_b**2 + side_c**2 - side_a**2)/(2*side_b*side_c))
    return angle_a

def angularSpeed(race,horses,point):
    num_horses = len(horses)
    
    race["lonLag"] = race.longitudeFT.shift(num_horses)
    race["latLag"] = race.latitudeFT.shift(num_horses)
    
    side_a = euclidianDistance(race.longitudeFT,race.latitudeFT , race.lonLag , race.latLag)
    side_b =  euclidianDistance(race.longitudeFT, race.latitudeFT, *point)
    side_c = euclidianDistance(*point, race.lonLag, race.latLag)
    
    return angleCalc(side_a, side_b, side_c)
def viewTrack(df, x_axis, y_axis):
    sns.set(rc={'figure.figsize':(5,5)})
    g= sns.scatterplot(data= df, x = x_axis, y= y_axis,hue="program_number"\
                 ,legend=True)

    upper_long = max(df[x_axis]) +500 
    lower_long = min(df[x_axis]) - 500 

    range_axis = upper_long - lower_long

    lower_lat = min(df[y_axis]) -500 
    upper_lat = lower_lat + range_axis 

    g.set(xlim=(lower_long, upper_long))
    g.set(ylim=(lower_lat,  upper_lat))
    
    return g


def relationalFeatures(df_old):
    df = df_old.copy()
    
    i = 0
    

    new_cols = ["distance","distanceToLeader","finishDistanceLeader","infront",
                "behind","leaderEuclidianDist","winnerPos",]
    
    for date in df.race_date.unique():
        day = df.loc[df.race_date == date]
        
        for race_num in day.race_number.unique():
            race = day.loc[day.race_number == race_num]
            
            #sns.scatterplot(data=race, x = "longitudeFT",y="latitudeFT")
            point = rotationPoint(race) 
            #point_start = (float(race.iloc[0].longitudeFT), float(race.iloc[0].latitudeFT))
            
            horses = race.program_number.unique()
            
            race["angularSpeed"] = angularSpeed(race, horses, point)
            
            race = calculateSpeed(race,horses).fillna(0)
            df.loc[race.index,"speed"] = race.speed
            
            race = calculateAccel(race,horses).fillna(0)
            df.loc[race.index,"accel"] = race.accel
            

            dist_dict = dict(zip(horses,[0 for i in range(len(horses))]))
            ang_dict = dict(zip(horses,[0 for i in range(len(horses))]))
            
            for ind in race.trakus_index.unique():
                trakus = race.loc[race.trakus_index == ind]
                
                trakus[new_cols] = 0
                
                for horse in horses:

                    dist_dict[horse] = dist_dict[horse] \
                    + float(trakus.loc[trakus.program_number==horse,"speed"])
                    
                    ang_dict[horse] = ang_dict[horse] \
                    + float(trakus.loc[trakus.program_number==horse,"angularSpeed"])

                    trakus.loc[trakus.program_number==horse,"distance"] = dist_dict[horse]
                    trakus.loc[trakus.program_number==horse,"angularDistance"] = ang_dict[horse]
                
                trakus = trakus.sort_values("angularDistance")
                lead_horse = str(trakus.iloc[-1]["program_number"])
                
                leader_distance = dist_dict[lead_horse]
                leader_lon = trakus.iloc[len(trakus)-1,:].longitudeFT
                leader_lat = trakus.iloc[len(trakus)-1,:].latitudeFT
                trakus["leaderEuclidianDist"] = (((trakus.longitudeFT - leader_lon) **2) +\
                    ((trakus.latitudeFT - leader_lat) **2))**.5
                
                trakus["distanceToLeader"] = leader_distance - trakus["distance"] 
                trakus["finishDistanceLeader"] = trakus["distanceFT"] - leader_distance
                trakus["behind"] = range(len(horses))
                trakus["infront"] = range(len(horses)-1,-1,-1 )
                trakus["sparseWinnerPos"] = trakus.won * (trakus.infront+1)
                trakus["winnerPos"] = trakus["sparseWinnerPos"].sum()
                
                
                df.loc[trakus.index, new_cols+["angularDistance","angularSpeed"]] = trakus[new_cols+["angularDistance","angularSpeed"]]
                i+=1
                
                if(False): #DELETE
                    print(point)
                    print(lead_horse)
                    #sns.scatterplot(data = trakus, x = "longitudeFT", y="latitudeFT",hue="program_number")
                    viewTrack(trakus, "longitudeFT", "latitudeFT")
                    #sns.scatterplot(x=point[0], y=point[1])
                    return trakus #DEleete
    return df
                
def calculateSpeed(race,horses):
    num_horses = len(horses)
    
    race["lonLag"] = race.longitudeFT.shift(num_horses)
    race["latLag"] = race.latitudeFT.shift(num_horses)
    
    race["speed"]  = (((race.longitudeFT - race["lonLag"]) **2) +\
        ((race.latitudeFT - race["latLag"]) **2))**.5
        
    return race

def calculateAccel(race,horses):
    num_horses = len(horses)
    
    race["speedLag"] = race.speed.shift(num_horses)
    race["accel"] = race.speed - race.speedLag
    return race

def speedToCover(df):
    new = df.copy()
    new["finishDistance"] = new.finishDistanceLeader + new.leaderEuclidianDist
    new["speedToCover"] = new.speed - ((new.speed * new.finishDistanceLeader)/(new.finishDistance))
    
    return new
#%%
data = complete.copy()

data = data.drop(["jockey","purse","post_time","race_type"],1)
data = data.drop(bad_indices)
data = data.drop(finished_races)

data["track_condition"]= data.track_condition.astype("category").cat.codes
data["course_type"]= data.course_type.astype("category").cat.codes

data = data.loc[(data.trakus_index % 4) == 1]

data = sortDF(data)

data = coordinatePlane(data)

data = furlongToFeet(data, "distance_id")

data = oddsToProb(data, "odds")

data = winBin(data, "position_at_finish")

data = relationalFeatures(data)

data = speedToCover(data)

drop_cols = ["latitude","longitude","distance_id","position_at_finish",
             'track_id', 
             #'race_date', 'race_number', 'program_number','trakus_index',"odds", 'longitudeFT', 'latitudeFT', 'distanceFT', 
             'run_up_distance',
             #'weight_carried',
             ]

data_csv = data.drop(drop_cols,1)

data_csv.to_csv("/Users/Andrew1/Desktop/Kaggle/Derby/engineeredDataV5.csv")


