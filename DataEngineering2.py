#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 21:48:11 2022

@author: Andrew1
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import random 

import math

import warnings

np.random.seed(123)

warnings.simplefilter("ignore")

#tracking
path_to_file = "/Users/Andrew1/Desktop/Kaggle/Derby/engineeredDataV5.csv"

data = pd.read_csv(path_to_file, index_col=(0))



#race
path_to_file = "/Users/Andrew1/Desktop/Kaggle/big-data-derby-2022/nyra_race_table.csv"

race = pd.read_csv(path_to_file)

#%%

def leaderPathEfficiency(race_slice):
    if((race_slice.distance == 0).any()):
        return 1
    leader_path_eff = race_slice.loc[race_slice.index[0],"angularDistance"] / race_slice.loc[race_slice.index[0],"distance"]
    avg_path_eff = race_slice.loc[race_slice.index[1:],"angularDistance"].mean() / race_slice.loc[race_slice.index[:1],"distance"].mean()
    #print("leader:",leader_path_eff)
    #print("everyone else:",avg_path_eff)
    
    return leader_path_eff / avg_path_eff
    

def retrieveSampleLeader(df,races):
    i = 0
    sample_columns = ["avgSpeedToCover","leaderFinsihDistance",
                                    "avgLeaderDistance", "minLeaderDistance",
                                    "avgLeaderDistanceEuc", "minLeaderDistanceEuc",
                                    "frontWins","trakus_index","priorWinProb",
                                    "proportionWinProb","numHorses","minSpeedToCover",
                                    "leaderSpeed","maxSpeed","avgSpeed","race_index"
                                    ,"leaderPathEfficiency","leadHorse"]
    
    sample = pd.DataFrame(columns= sample_columns)
    for index in races.index:
        slice_dict = dict()
        date = races.loc[index,"race_date"]
        race_num = races.loc[index,"race_number"]
        
        full_race = df.loc[(df.race_date == date) & (df.race_number == race_num)]
        #assert not full_race.empty
        
        for race_point in full_race.trakus_index.unique():
            race_slice = full_race.loc[full_race.trakus_index == race_point]
            #assert not race_slice.empty
            
            race_slice = race_slice.sort_values("infront")
        
  
        
        
            slice_dict["avgSpeedToCover"] = race_slice.speedToCover.sum() / (len(race_slice) - 1)
            slice_dict["leaderFinsihDistance"] = race_slice.finishDistanceLeader.mean()
            slice_dict["avgLeaderDistance"] = race_slice.distanceToLeader.sum() / (len(race_slice) - 1)
            slice_dict["avgLeaderDistanceEuc"] = race_slice.leaderEuclidianDist.sum() / (len(race_slice) - 1)
            slice_dict["frontWins"] = race_slice.frontWins.sum()
        
            slice_dict["numHorses"] = len(race_slice)
            
            slice_dict["leaderSpeed"] = race_slice.loc[race_slice.index[0],"speed"]
            slice_dict["priorWinProb"] = race_slice.loc[race_slice.index[0],"winProb"]
            slice_dict["leaderPathEfficiency"] = leaderPathEfficiency(race_slice)
            
            slice_dict["leadHorse"] = race_slice.loc[race_slice.index[0],"program_number"]
            race_slice = race_slice.drop(race_slice.index[0])
            
            lower_win_prob = race_slice.winProb.sum()
            
            slice_dict["maxSpeed"] =  max(race_slice.speed)
            slice_dict["avgSpeed"] = race_slice.speed.mean()
            slice_dict["proportionWinProb"] = slice_dict["priorWinProb"] / lower_win_prob
            slice_dict["minLeaderDistance"] = min(race_slice.distanceToLeader)
            slice_dict["minLeaderDistanceEuc"] = min(race_slice.leaderEuclidianDist)
            slice_dict["minSpeedToCover"] = min(race_slice.speedToCover)
            slice_dict["trakus_index"] = race_slice.trakus_index.mean()
            slice_dict["race_index"] = index

            sample = sample.append(slice_dict, ignore_index=True)
            i+=1

        

    return sample
def retrieveSampleLower(df,races,pos):
    i = 0
    sample_columns = ["avgSpeedToCover","leaderFinsihDistance",
                                    "avgLeaderDistance", "minLeaderDistance",
                                    "avgLeaderDistanceEuc", "minLeaderDistanceEuc",
                                    "frontWins","trakus_index","priorWinProb",
                                    "proportionWinProb","numHorses","minSpeedToCover",
                                    "leaderSpeed","maxSpeed","avgSpeed","race_index"
                                    ,"leaderPathEfficiency","leadHorse"]
    
    sample = pd.DataFrame(columns= sample_columns)
    for index in races.index:
        slice_dict = dict()
        date = races.loc[index,"race_date"]
        race_num = races.loc[index,"race_number"]
        
        full_race = df.loc[(df.race_date == date) & (df.race_number == race_num)]
        if(full_race.empty ):
            continue

        
        full_race = full_race.loc[full_race.infront >= (pos-1)]
        if (len(full_race.program_number.unique()) <= 2):
            continue
        
        for race_point in full_race.trakus_index.unique():
            race_slice = full_race.loc[full_race.trakus_index == race_point]
            
            race_slice = race_slice.sort_values("infront")
        
        
            leader_lon = float(race_slice.iloc[0,:].longitudeFT)        
            leader_lat = float(race_slice.iloc[0,:].latitudeFT)
            leader_dist = float(race_slice.iloc[0,:].distance)
        
            race_slice['leaderEuclidianDist'] = (((race_slice.longitudeFT - leader_lon) **2) +\
                                                 ((race_slice.latitudeFT - leader_lat) **2))**.5
            race_slice["distanceToLeader"] = leader_dist - race_slice["distance"] 
            race_slice["finishDistanceLeader"] = race_slice["distanceFT"] - leader_dist
        
            race_slice = speedToCover(race_slice)    
        
        
            slice_dict["avgSpeedToCover"] = race_slice.speedToCover.sum() / (len(race_slice) - 1)
            slice_dict["leaderFinsihDistance"] = race_slice.finishDistanceLeader.mean()
            slice_dict["avgLeaderDistance"] = race_slice.distanceToLeader.sum() / (len(race_slice) - 1)
            slice_dict["avgLeaderDistanceEuc"] = race_slice.leaderEuclidianDist.sum() / (len(race_slice) - 1)
            slice_dict["frontWins"] = race_slice.frontWins.sum()
        
            slice_dict["numHorses"] = len(race_slice)
            
            slice_dict["leaderSpeed"] = race_slice.loc[race_slice.index[0],"speed"]
            slice_dict["priorWinProb"] = race_slice.loc[race_slice.index[0],"winProb"]
            slice_dict["leaderPathEfficiency"] = leaderPathEfficiency(race_slice)
            
            slice_dict["leadHorse"] = race_slice.loc[race_slice.index[0],"program_number"]
            race_slice = race_slice.drop(race_slice.index[0])
            
            lower_win_prob = race_slice.winProb.sum()
            
            
            slice_dict["maxSpeed"] =  max(race_slice.speed)
            
            slice_dict["avgSpeed"] = race_slice.speed.mean()
            slice_dict["proportionWinProb"] = slice_dict["priorWinProb"] / lower_win_prob
            slice_dict["minLeaderDistance"] = min(race_slice.distanceToLeader)
            slice_dict["minLeaderDistanceEuc"] = min(race_slice.leaderEuclidianDist)
            slice_dict["minSpeedToCover"] = min(race_slice.speedToCover)
            slice_dict["trakus_index"] = race_slice.trakus_index.mean()
            slice_dict["race_index"] = index

            sample = sample.append(slice_dict, ignore_index=True)
            i+=1
            #if(i==100):
                #return sample
        

    return sample
def speedToCover(df):
    new = df.copy()
    new["finishDistance"] = new.finishDistanceLeader + new.leaderEuclidianDist
    new["speedToCover"] = new.speed - ((new.speed * new.finishDistanceLeader)/(new.finishDistance))
    
    return new

def positionWins(df,pos):
    new = df.copy()
    
    new["frontWins"] =0
    new.loc[(new.won == 1) & (new.infront == pos-1), "frontWins"] =1
    
    return new

#%%


data = positionWins(data, 1)

test_races = race.sample(int(.33 * len(race)))
train_races = race.drop(test_races.index)


train_data = retrieveSampleLeader(data, train_races)
test_data = retrieveSampleLeader(data, test_races)

#%%

def reduceDataTest(data, next_pos,train_races, test_races):
    #data_reduced = data.loc[data.winnerPos != next_pos-1]
    data_reduced = positionWins(data, next_pos)

    test_data_reduced = retrieveSampleLower(data_reduced, test_races,next_pos)
    #train_data_reduced = retrieveSampleLower(data_reduced, train_races,next_pos)
    
    return  test_data_reduced

test_reduced_1 = reduceDataTest(data, 2, train_races, test_races)

test_reduced_2 = reduceDataTest(data, 3, train_races, test_races)

def reduceDataTrain(data, next_pos,train_races, test_races):
    data_reduced = data.loc[data.winnerPos != next_pos-1]
    data_reduced = positionWins(data_reduced, next_pos)

   # test_data_reduced = retrieveSampleLower(data_reduced, test_races,next_pos)
    train_data_reduced = retrieveSampleLower(data_reduced, train_races,next_pos)
    
    return  train_data_reduced, data_reduced

train_reduced_1, reduced_1 = reduceDataTrain(data, 2, train_races, test_races)

train_reduced_2, reduced_2 = reduceDataTrain(reduced_1, 3, train_races, test_races)

#reduced_3,test_reduced_3,train_reduced_3 = reduceData(reduced_2, 4, train_races, test_races)

#reduced_4,test_reduced_4,train_reduced_4 = reduceData(reduced_3, 5, train_races, test_races)

#%%
test_data.to_csv("/Users/Andrew1/Desktop/Kaggle/Derby/test_data_V6.csv")
train_data.to_csv("/Users/Andrew1/Desktop/Kaggle/Derby/train_data_V6.csv")

test_reduced_1.to_csv("/Users/Andrew1/Desktop/Kaggle/Derby/test_reduced_1_V6.csv")
train_reduced_1.to_csv("/Users/Andrew1/Desktop/Kaggle/Derby/train_reduced_1_V6.csv")

test_reduced_2.to_csv("/Users/Andrew1/Desktop/Kaggle/Derby/test_reduced_2_V6.csv")
train_reduced_2.to_csv("/Users/Andrew1/Desktop/Kaggle/Derby/train_reduced_2_V6.csv")

#test_reduced_3.to_csv("/Users/Andrew1/Desktop/Kaggle/Derby/test_reduced_3_V6.csv")
#train_reduced_3.to_csv("/Users/Andrew1/Desktop/Kaggle/Derby/train_reduced_3_V6.csv")

#test_reduced_4.to_csv("/Users/Andrew1/Desktop/Kaggle/Derby/test_reduced_4_V6.csv")
#train_reduced_4.to_csv("/Users/Andrew1/Desktop/Kaggle/Derby/train_reduced_4_V6.csv")



