#-*- encoding=utf-8 -*-

import pandas as pd
import numpy as np
import re

inv_df = pd.read_csv('../../../data/JD815/inv.csv',names = ['binId','sku','qty'])

location_df = pd.read_csv('../../../data/JD815/locations.csv',names = ['binId','x','y','1','2','3','4','5'])

location_df.drop(columns = ['1','2','3','4','5'],inplace = True) 


inv_df = pd.merge(inv_df,location_df,how = 'left',on = ['binId'])
inv_df['时间'] = 0
inv_df['箱子位置'] = '货架'
inv_df['位置编号'] = inv_df.binId.copy()
inv_df.rename(columns = {'binId':'箱子编号','x':'箱子物理位置x','y':'箱子物理位置y','qty':'件数'},inplace = True)
columns = ['箱子编号','箱子位置','位置编号','sku','件数','箱子物理位置x','箱子物理位置y']
inv_df = inv_df[columns]

#inv_df
columns = ['时间','箱子编号','箱子位置','位置编号','箱子物理位置x','箱子物理位置y','机器人编号','Sku','件数','操作台编号']

#操作台编号，第一个unload口的映射，x，y
stationInfo = pd.read_csv('station.csv')

ReserveStock = pd.read_csv('longDis/kubot_reserve_tock.csv',
                           names = ['Time','KubotID','BinID','BinLoc','SKUID','Amount','DestID','ReqID'])
del ReserveStock['ReqID']
ReserveStock.rename(columns = {
    'Time':'时间',
    'KubotID':'机器人编号',
    'BinID':'箱子编号',
    'BinLoc':'位置编号',
    'SKUID':'Sku',
    'Amount':'件数',
    'DestID':'映射点'
},inplace = True)

ReserveStock['箱子位置'] = '货架(预约)'
location_df.rename(columns = {'binId':'位置编号'},inplace = True)
ReserveStock = pd.merge(ReserveStock,location_df,how = 'left',on = ['位置编号'])
ReserveStock.rename(columns = {'x':'箱子物理位置x','y':'箱子物理位置y'},inplace = True)


stationdf = stationInfo[['操作台编号','映射点']]

ReserveStock = pd.merge(ReserveStock,stationdf,how = 'left', on =['映射点'])
del ReserveStock['映射点']
ReserveStock = ReserveStock[columns]


BinLoadedFromShel = pd.read_csv('longDis/bin_loaded_from_shelf.csv',
                               names = ['Time','BinID','KubotID','Loc'])
BinLoadedFromShel.rename(columns = {
    'Time':'时间',
    'BinID':'箱子编号',
    'KubotID':'机器人编号',
    'Loc':'位置编号'
},inplace = True)

BinLoadedFromShel['箱子位置'] = '机器人(load)'
BinLoadedFromShel['Sku'] = np.nan
BinLoadedFromShel['件数'] = np.nan
BinLoadedFromShel['操作台编号'] = np.nan
location_df.rename(columns = {'binId':'位置编号'},inplace = True)
BinLoadedFromShel = pd.merge(BinLoadedFromShel,location_df,how = 'left',on = ['位置编号'])
BinLoadedFromShel.rename(columns = {'x':'箱子物理位置x','y':'箱子物理位置y'},inplace = True)
columns = ['时间','箱子编号','箱子位置','位置编号','箱子物理位置x','箱子物理位置y','机器人编号','Sku','件数','操作台编号']
BinLoadedFromShel = BinLoadedFromShel[columns]


processBin = pd.read_csv('longDis/bin_start_process.csv',names = ['Time','BinID','Sku','Qty','StationNo'])
processBin.rename(columns = {
    'Time':'时间',
    'BinID':'箱子编号',
    'Sku':'Sku',
    'Qty':'件数',
    'StationNo':'操作台编号'
},inplace = True )
processBin['箱子位置'] = '在操作台处理'
processBin['位置编号'] = np.nan
processBin['机器人编号'] = np.nan
stationInfo.rename(columns = {'操作台物理位置x':'箱子物理位置x','操作台物理位置y':'箱子物理位置y'},inplace = True)
processBin = pd.merge(processBin,stationInfo,how = 'left',on = ['操作台编号'])


processBin = processBin[columns]



KubotLoadBinFromConveyor = pd.read_csv('longDis/loaded_from_station.csv',names = ['Time','BinID','KubotNo','StaitonNo'])
KubotLoadBinFromConveyor.rename(columns = {
    'Time':'时间',
    'BinID':'箱子编号',
    'KubotNo':'机器人编号',
    'StaitonNo':'操作台编号'
},inplace = True)
KubotLoadBinFromConveyor['箱子位置'] = '正要离开操作台'
KubotLoadBinFromConveyor['位置编号'] = np.nan
KubotLoadBinFromConveyor['Sku'] = np.nan
KubotLoadBinFromConveyor['件数'] = np.nan

stationInfo

KubotLoadBinFromConveyor = pd.merge(KubotLoadBinFromConveyor,stationInfo,how = 'left',on = ['操作台编号'])
KubotLoadBinFromConveyor = KubotLoadBinFromConveyor[columns]


BinUnloadToShelf = pd.read_csv('longDis/unloaded_to_shelf.csv',names = ['Time','BinID','KubotNo','Loc'])
BinUnloadToShelf.rename(columns = {
    'Time':'时间',
    'BinID':'箱子编号',
    'KubotNo':'机器人编号',
    'Loc':'位置编号'
},inplace = True)
BinUnloadToShelf['箱子位置'] = '机器人(unload)'

BinUnloadToShelf = pd.merge(BinUnloadToShelf,location_df,how = 'left',on = ['位置编号'])
BinUnloadToShelf.rename(columns = {'x':'箱子物理位置x','y':'箱子物理位置y'},inplace = True)

BinUnloadToShelf['Sku'] = np.nan
BinUnloadToShelf['操作台编号'] = np.nan
BinUnloadToShelf['件数']  = np.nan

BinUnloadToShelf = BinUnloadToShelf[columns]


Bin_snapshot = pd.concat([ReserveStock,BinLoadedFromShel,processBin,KubotLoadBinFromConveyor,BinUnloadToShelf])
Bin_snapshot.sort_values(by = '时间',inplace = True)
Bin_snapshot.drop_duplicates(inplace = True)


Bin_snapshot.sort_values(by = ['箱子编号','时间'],inplace = True)
Bin_snapshot.机器人编号.fillna(method = 'ffill',inplace = True)



#机器人接到搬箱子的命令
kubotAcceptMission = pd.read_csv('longDis/kubotAcceptMission.csv',names = ['Time','KubotNo','BinID','StationID'])
kubotAcceptMission['箱子位置'] = '机器人接到搬箱子的命令'

#机器人接到前往操作台的命令
kubotToStation = pd.read_csv('longDis/kubotToStation.csv',names = ['Time','KubotNo','StationID'])
kubotToStation['BinID'] = np.nan
kubotAcceptMission['箱子位置'] = '机器人接到前往操作台的命令'

pd.set_option('display.max_columns', None)
kubotBin = pd.concat([kubotAcceptMission,kubotToStation]).sort_values(by = ['KubotNo','Time'])
kubotBin.StationID.fillna(method = 'bfill',inplace = True)
kubotBin.dropna(subset = ['BinID'],inplace = True)

kubotBin.rename(columns = {'Time':'时间','KubotNo':'机器人编号','BinID':'箱子编号','StationID':'映射点'},inplace=True)  

#操作台编号要调整一下
stationmap = pd.read_csv('station.csv')
del stationmap['操作台物理位置x']
del stationmap['操作台物理位置y']
kubotBin = pd.merge(kubotBin,stationmap,how = 'left',on = ['映射点'])
del kubotBin['映射点']

#获得上述机器人接到命令时候的位置
#原始坐标点
ori_state_points = pd.read_csv('file/ori_state_points.csv',names = ['state_id','theta','x','y'])

#机器人每时每刻的位置
kubotLoc = pd.read_csv('file/state_id.csv',names = ['时间','机器人编号','state_id','target_state_id'])
del kubotLoc['target_state_id']

kubotLoc.drop_duplicates(inplace = True)

kubotLoc = pd.merge(kubotLoc,ori_state_points,how = 'left', on = ['state_id'])
del kubotLoc['theta']
del kubotLoc['state_id']

kubotLoc.rename(columns = {'x':'机器人位置x','y':'机器人位置y'},inplace = True)


kubotLoc['时间'] = kubotLoc['时间'].apply(lambda x :re.findall(r"\d{1,}?\.\d{1}", str(x)) )
kubotLoc['时间1'] = kubotLoc['时间'].apply(lambda x : x[0])
del kubotLoc['时间']
kubotLoc.rename(columns = {'时间1':'时间'},inplace = True)
kubotLoc['时间'] = kubotLoc['时间'].apply(lambda x : float(x))
kubotLoc.to_csv('kubotLoc.csv',index = False)



Bin_snapshot['机器人编号'].fillna(0,inplace = True)
Bin_snapshot['机器人编号'] = Bin_snapshot['机器人编号'].apply(lambda x : int(x))

Bin_snapshot = pd.merge(Bin_snapshot,kubotLoc , how = 'left', on=['时间','机器人编号'])



Bin_snapshot.to_csv('Bin_snapshot.csv',index = False)

print("Yeah!")