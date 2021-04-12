#!/bin/bash
rm -f normal_order_dispatch.csv
rm -f recombine_order_dispatch.csv
rm -f normal_order_finish.csv
rm -f recombine_order_finish.csv
rm -f order_progress_fitst.csv
rm -f order_progress_sh.csv
rm -f kubot_progress_sh.csv
rm -f bin_progress_sh.csv
rm -f station_map.csv
rm -f state_id.csv
rm -f ori_state_id.csv
rm -f ori_theta.csv
rm -f rm -f ori_x.csv
rm -f rm -f rm -f ori_y.csv
rm -f ori_state_points.csv
rm -f station_state_id.csv
rm -f rest_zoom_x.csv
rm -f rest_zoom_y.csv
rm -f rest_zoom.csv
rm -f alley.csv
rm -f qr_code.csv
rm -f station_id.csv
rm -f play_back_speed.csv
rm -f pp.csv
rm -f station.csv
rm -f info.log
rm -f debug.log
rm -f project_name.csv
rm -f station_config.csv
rm -f DB_project_name.csv
rm -f unique_key.csv

echo "Input target project data file name: "
read input_project_name
project_name=`echo $input_project_name`
echo $project_name > project_name.csv

echo "Input the entire config yaml  name (including suffix): "
read input_config_file



cat ../../log/simulator/${project_name}_INFO_* > info.log
cat ../../log/simulator/${project_name}_DEBUG_* > debug.log
info_file=info.log
debug_file=debug.log
state_points_json=.../../data/${project_name}/state_points.json
station_config_json=.../../data/${project_name}/station_config.json
rest_stations_json=.../../data/${project_name}/rest_stations.json
adjacency_list_in=.../../data/${project_name}/adjacency_list.in
order_file=.../../data/${project_name}/order.csv
locations_file=.../../data/${project_name}/locations.csv
config_file=../config/${input_config_file}


mysql -h 172.20.8.10 -uroot -proot FastSimulation_log -e "SELECT * from dwd_project_name" > DB_project_name.csv

grep $project_name DB_project_name.csv >> /dev/null

#if [ $? -ne 0 ]; then
#table name : project_name
#table field : project_name
script_project_name="
import pandas as pd
import numpy as np
import pymysql
from sqlalchemy import create_engine
project_namefile  = pd.read_csv('project_name.csv',names = ['project_name'])
engine = create_engine('mysql+pymysql://root:root@172.20.8.10:3306/FastSimulation_log',encoding='utf8')
project_namefile.to_sql(name='dwd_project_name', con=engine,if_exists='append', chunksize=1000,index=None)
"
python -c """$script_project_name"""

simulation_time=$(date "+%Y-%m-%d %H:%M:%S")

function git.branch {
  br=`git branch | grep "*"`
  echo ${br/* /}
}
simulation_main_branch=`git.branch`

current_git_branch_latest_id=`git rev-parse HEAD`
commit_id=$current_git_branch_latest_id



### mc
cd ../../src/models/mission_control/MissionControlAlgorithm
function git.branch_mc {
  br=`git branch | grep "*"`
  echo ${br/* /}
}
simulation_main_branch_mc=`git.branch_mc`


current_git_branch_latest_id_mc=`git rev-parse HEAD`
commit_id_mc=$current_git_branch_latest_id_mc



###OP
cd ../../order_processor/OrderProcessorAlgorithm
function git.branch_op {
  br=`git branch | grep "*"`
  echo ${br/* /}
}
simulation_main_branch_op=`git.branch_op`

current_git_branch_latest_id_op=`git rev-parse HEAD`
commit_id_op=$current_git_branch_latest_id_op


cd ../../../../statistic/script
echo "$project_name,$simulation_time,$simulation_main_branch,$commit_id,$simulation_main_branch_mc,$commit_id_mc,$simulation_main_branch_op,$commit_id_op" > unique_key.csv

#table name : uniquek_key_map
#table field : project_name create_time main_branch_name main_branch_commitID mc_branch_name mc_branch_commitID op_branch_name op_branch_commitID
script_uniquek_key_map="
import pandas as pd
import numpy as np
import pymysql
import warnings
from sqlalchemy import create_engine
warnings.filterwarnings('ignore')
unique_key = pd.read_csv('unique_key.csv',names = ['project_name','create_time','main_branch_name','main_branch_commitID','mc_branch_name','mc_branch_commitID','op_branch_name','op_branch_commitID'])

engine = create_engine('mysql+pymysql://root:root@172.20.8.10:3306/FastSimulation_log',encoding='utf8')
unique_key.to_sql(name='dwd_unique_key_map', con=engine,if_exists='append', chunksize=1000,index=None)
print('Get unique_key_map')
"
python -c """$script_uniquek_key_map"""

#get project key
mysql -h 172.20.8.10 -uroot -proot FastSimulation_log -e "SELECT project_key from dwd_unique_key_map order by create_time desc limit 1" > project_key.csv

echo "start to clean log"

#table name : order_progress_sh
#table field : time order_no station_id sku qty bin_id order_status pick_bin_time
cat $info_file | grep "dispatch to Station" | grep -v 'recombined' | awk -F '[ ,]' '{print $8","$11","$15",,,,dispatch"}' > normal_order_dispatch.csv
cat $info_file | grep "dispatch to Station" | grep 'recombined' | awk -F '[ ,]' '{print $8","$12","",,,,dispatch"}' > recombine_order_dispatch.csv
cat $info_file | grep "completed!" | grep -v "recombined" | awk -F '[ ,]' '{print $8","$11",,,,,finish"}' > normal_order_finish.csv
cat $info_file | grep "completed!" | grep  "recombined" | awk -F '[ ,]' '{print $8","$12",,,,,finish"}' > recombine_order_finish.csv
cat $info_file | grep ', Amount ' | awk -F '[ ,]' '{print $8","$14","$11","$17","$23","$20",progress"}' > order_progress_fitst.csv

#auxiliary pick_bin_time
#time station_id bin_id pick_bin_time
cat $info_file | grep "Send Picking Action" | awk -F '[ ,]' '{print $8","$16","$19","$20 }' > pick_bin_time.csv
cat normal_order_dispatch.csv recombine_order_dispatch.csv normal_order_finish.csv recombine_order_finish.csv order_progress_fitst.csv >  order_progress_sh.csv

script_order_progress="
import pandas as pd
import numpy as np
import warnings
import pymysql
from sqlalchemy import create_engine
warnings.filterwarnings('ignore')


order_progress_sh = pd.read_csv('order_progress_sh.csv',
                               names = ['time','order_no','station_id','sku','qty','bin_id','order_status'])
order_progress_sh.sort_values(by = ['order_no','time','station_id'],inplace= True)
project_key = pd.read_csv('project_key.csv')


order_progress_sh['station_id'] = np.where(
    ((order_progress_sh['station_id'].isnull() == True) & (order_progress_sh['order_status'] == 'dispatch')),
    order_progress_sh['station_id'].fillna(method = 'bfill'),
    order_progress_sh['station_id'] )

order_progress_sh['station_id'] = np.where(
    ((order_progress_sh['station_id'].isnull() == True) & (order_progress_sh['order_status'] == 'finish')),
    order_progress_sh['station_id'].fillna(method = 'ffill'),
    order_progress_sh['station_id'] )

order_progress_sh_bin_progress = order_progress_sh[order_progress_sh['order_status'] == 'progress']
del order_progress_sh_bin_progress['order_no']
del order_progress_sh_bin_progress['sku']
del order_progress_sh_bin_progress['qty']
order_progress_sh_bin_progress.drop_duplicates(inplace = True)
order_progress_sh_bin_progress.sort_values(by = ['station_id','time'] , inplace = True)

order_progress_sh_bin_progress.station_id =  order_progress_sh_bin_progress.station_id.apply(lambda x :int(x))
order_progress_sh_bin_progress.bin_id =  order_progress_sh_bin_progress.bin_id.apply(lambda x :int(x))

pick_bin_time = pd.read_csv('pick_bin_time.csv',names = ['time','station_id','bin_id','pick_bin_time'])
pick_bin_time.sort_values(by = ['station_id','time'] , inplace = True)
del pick_bin_time['time']

order_progress_sh_bin_progress = pd.merge(order_progress_sh_bin_progress,pick_bin_time,
                                         how = 'left',
                                         on = ['station_id','bin_id'])

order_progress_sh = pd.merge(order_progress_sh,order_progress_sh_bin_progress,
                            how = 'left',
                            on = ['time','station_id','bin_id','order_status'])


order_progress_sh = pd.concat([order_progress_sh,project_key], axis = 1)

order_progress_sh = order_progress_sh[['project_key','time','order_no','station_id','sku','qty','bin_id','order_status','pick_bin_time']]

order_progress_sh.fillna(method = 'ffill',inplace = True)
engine = create_engine('mysql+pymysql://root:root@172.20.8.10:3306/FastSimulation_log',encoding='utf8')
order_progress_sh.to_sql(name='dwd_order_progress', con=engine,if_exists='append', chunksize=1000,index=None)
print('Get order_progress')
"
python -c """$script_order_progress"""
#
#
#
##table name : bin_progress_sh
##table field : time station_id kubot_no bin_id sku qty location bin_status
#cat $info_file |  awk -F '[ ,]' '{switch($0) {case/Reserve Stock/: print $8",,"","$13","$15","$16","$14",reserve_stock" ; break ; \
#                                        case /Bin Loaded From Shelf/: print $8","","$15","$14",,,"$16",bin _loaded _from _shelf" ; break ; \
#                                        case /, Amount /: print $8","$11",,"$20","$17","$23",,processed_by_station" ; \
#                                        case/Kubot Load Bin From Conveyor/: print $8","$18","$17","$16",,,,kubot _load _bin _from_conveyor" ; break ; \
#                                        case/Bin Unload To Shelf/: print $8",,"$15","$14",,,"$16",bin_unload _to_shelf" }}' > bin_progress_sh.csv
#
#
#
#script_bin_progress="
#import pandas as pd
#import numpy as np
#import pymysql
#from sqlalchemy import create_engine
#import warnings
#warnings.filterwarnings('ignore')
#
#bin_progress_sh = pd.read_csv('bin_progress_sh.csv',
#                               names = ['time','station_id','kubot_no','bin_id','sku','qty','location','bin_status'])
#bin_progress_sh.sort_values(by = ['bin_id','time'],inplace = True)
#bin_progress_sh.sku.fillna(method = 'ffill',inplace = True)
#bin_progress_sh.qty.fillna(method = 'ffill',inplace = True)
#bin_progress_sh.kubot_no.fillna(method = 'bfill',inplace = True)
#bin_progress_sh['station_id'] = np.where(
#    ((bin_progress_sh['station_id'].isnull() == True) & (bin_progress_sh['bin_status'].isin(['reserve_stock','bin _loaded _from _shelf']))),
#    bin_progress_sh['station_id'].fillna(method = 'bfill'),
#    bin_progress_sh['station_id'] )
#
#bin_progress_sh['station_id'] = np.where(
#    ((bin_progress_sh['station_id'].isnull() == True) & (bin_progress_sh['bin_status']== 'bin_unload _to_shelf')),
#    bin_progress_sh['station_id'].fillna(method = 'ffill'),
#    bin_progress_sh['station_id'] )
#
#engine = create_engine('mysql+pymysql://root:root@172.20.8.10:3306/FastSimulation_log',encoding='utf8')
#bin_progress_sh.to_sql(name='dwd_bin_progress', con=engine,if_exists='append', chunksize=1000,index=None)
#print('Get bin_progress')
#"
#
#python -c """$script_bin_progress"""
#
#
#
#
#
##table name : kubot_progress_sh
##table field : time kubot_no  sation_id bin_id kubot_status battery
#cat $info_file | awk -F '[ ,]' '{switch($0){case/to load bin/: print $8","$11",,"$15",accept_mission_to_load_bin," ; break ;
#                                        case/to destination/: print $8","$11","$14",,to_destination," ; break ;
#                                        case/Bin Loaded From Shelf/: print $8","$15",,"$14",bin_loaded_from_shelf," ; break ;
#                                        case/bin; Kubot/: print $8","$11",,,arrive_at_station_first_unload_port," ; break ;
#                                        case/Kubot Load Bin From Conveyor/: print $8","$17","$18","$16",kubot_load_bin_from_conveyor," ; break ;
#                                        case/to unload bin/: print $8","$11",,"$15",accept_mission_to_unload_bin," ; break ;
#                                        case/Bin Unload To Shelf/: print $8","$15",,"$14",bin_unload_to_shelf," ; break ; \
#                                        case/begin charging/: print $8","$11",,,begin_charging,"$16 ; break;
#                                        case/finish charging/: print $8","$11",,,begin_charging,"$16; break ;
#                                        case/is working/: print $8","$11",,,kubot_is_working," ; break ;
#                                        case/is idle/: print $8","$11",,,kubot_is_idle," }}' > kubot_progress_sh.csv
#
##auxiliary table : station_map
## real_station_id  station_id
#cat $info_file | grep "destination id" | awk -F "[ , ']" '{print $7","$11}' > station_map.csv
#
#script_kubot_progress="
#import pandas as pd
#import numpy as np
#import pymysql
#from sqlalchemy import create_engine
#import warnings
#warnings.filterwarnings('ignore')
#
#kubot_progress_sh = pd.read_csv('kubot_progress_sh.csv',
#                               names = ['time','kubot_no','station_id','bin_id','kubot_status','battery'])
#kubot_progress_sh.sort_values(by = ['kubot_no','time'],inplace = True)
#spliteAcceptMcAndToDest = kubot_progress_sh[kubot_progress_sh['kubot_status'].isin(['accept_mission_to_load_bin','to_destination'])]
#spliteNOT_AcceptMcAndToDest = kubot_progress_sh[~kubot_progress_sh['kubot_status'].isin(['accept_mission_to_load_bin','to_destination'])]
#spliteAcceptMcAndToDest.station_id.fillna(method = 'bfill',inplace = True)
#kubot_progress_sh = pd.concat([spliteNOT_AcceptMcAndToDest,spliteAcceptMcAndToDest])
#kubot_progress_sh.sort_values(by = ['kubot_no','time'],inplace = True)
#kubot_progress_sh.station_id.fillna(method = 'ffill', inplace = True)
#
##有没有station的映射表
#station_map = pd.read_csv('station_map.csv',names = ['real_station_id','station_id'])
#
#kubot_progress_sh = pd.merge(kubot_progress_sh,station_map,how = 'left',on = ['station_id'])
#kubot_progress_sh.station_id = np.where(kubot_progress_sh.real_station_id.isnull() == True,
#                                        kubot_progress_sh.station_id,
#                                        kubot_progress_sh.real_station_id)
#del kubot_progress_sh['real_station_id']
#kubot_progress_sh = kubot_progress_sh[kubot_progress_sh.kubot_status != 'to_destination']
#
#engine = create_engine('mysql+pymysql://root:root@172.20.8.10:3306/FastSimulation_log',encoding='utf8')
#kubot_progress_sh.to_sql(name='dwd_kubot_progress', con=engine,if_exists='append', chunksize=1000,index=None)
#print('Get kubot_progress')
#"
#python -c """$script_kubot_progress"""
#
#
##table_name : kubot_pp
##table field : kubot_id current_state target_state
#cat $debug_file | grep "Statistic(Kubot ID" |   awk -F '[ ,]' '{print $8","$15","$17","$19}' > state_id.csv
##Aquire system state_map_table : state_id theta x y
#cat $state_points_json | grep "state id" |  awk -F '[,: ]' '{print $12}'  >  ori_state_id.csv
#cat $state_points_json | grep "theta" | awk -F '[,: ]' '{print $15}'  >  ori_theta.csv
#cat $state_points_json | grep "x" |  awk -F '[,: ]' '{print $15}'  >  ori_x.csv
#cat $state_points_json | grep "y" | awk  '{print $2}'  >  ori_y.csv
#paste ori_state_id.csv ori_theta.csv ori_x.csv ori_y.csv -d "," > ori_state_points.csv
#
##Aquire statio_map_table : stateion_id state_id x y
#cat $station_config_json | grep "station id" | awk -F '[,:]' '{print $2}' >  station_id_temp.csv
#sed 's/ //g' station_id_temp.csv |  grep -v '^\s*$' > station_id.csv
#rm -f station_id_temp.csv
#cat $station_config_json |  grep '"unload state id":' | awk -F '[, ]' '{print $6}' |  grep -v '^\s*$'   >  station_state_id_temp.csv
#cat $station_config_json | grep "target state id" | awk  '{print $4}' |  grep -v '^\s*$'   >>  station_state_id_temp.csv
#sed 's/ //g' station_state_id_temp.csv > station_state_id.csv
#rm -f station_state_id_temp.csv
#paste -d "," station_state_id.csv station_id.csv |  grep -v '^\s*$'  > station.csv
#
##Aquire charging zone point location(x,y)
#cat $rest_stations_json | grep "x" | awk -F '[,:]' '{print $2}'  >  rest_zoom_x.csv
#cat $rest_stations_json | grep "y" | awk -F '[,:]' '{print $2}'  >  rest_zoom_y.csv
#paste rest_zoom_x.csv rest_zoom_y.csv  -d "," > rest_zoom.csv
#echo "1" >> play_back_speed.csv
#
##Aquire every point and its alley id
#cat $adjacency_list_in | awk -F '[ , : ; ]' '{print $1","$2","$3}' > alley.csv
#
##Aquire qr_code
#cat $adjacency_list_in | awk -F '[ , : ; ]' '{print $1","$2}' > qr_code.csv
#
#script_kubot_pp="
#import pandas as pd
#import numpy as np
#import pymysql
#from sqlalchemy import create_engine
#
##读取数据
#state_file = 'state_id.csv'
#state_data = pd.read_csv(state_file,sep = ',',names = ['time','kubot','state_id','target_state_id'])
#state_data['kubot'] = state_data['kubot'].apply(lambda x :int(x))
#
#ori_state_points = pd.read_csv('ori_state_points.csv',names = ['state_id','theta','x','y'])
#
#### 1.每个机器人的每个时刻当前位置和目标位置总表:  state_data
#### 2.整个地图 state id和x、y的对应点  : kubotlocation_file_map
#kubotlocation_file_map = pd.read_csv('ori_state_points.csv',names=['state_id','theta','x','y'])
## del kubotlocation_file_map['theta']
#kubotlocation_file_map.drop_duplicates(inplace = True)
#
### 3.得到机器人每时每刻的x、y  : kubotlocation_df
#kubotlocation_df = pd.merge(state_data,kubotlocation_file_map,how = 'left', on = ['state_id'])
#
### 4.获取巷道信息 ：alleyEachPoint
##### -1表示主干道，其余表示货架区巷道编号
#alleyEachPoint = pd.read_csv('alley.csv',names = ['x','y','alleyId'])
#alleyEachPoint.head()
#alleyEachPoint_target = alleyEachPoint.copy()
#alleyEachPoint_target.rename(columns = {'alleyId':'alleyId_target'},inplace = True)
#
### 5. 给机器人总表匹配巷道信息 ： kubotlocation_df
##### 6.添加一列判断巷道是否为主干道
#kubotlocation_df = pd.merge(kubotlocation_df,alleyEachPoint, how = 'left' , on = ['x','y'])
#kubotlocation_df['location'] = np.where(kubotlocation_df.alleyId == -1 , '主干道','货架区')
#kubotlocation_df.head()
#
### 7.获取操作台state_id 及 station_id
#station_relative_file = 'station.csv'
#station_relative_data = pd.read_csv(station_relative_file,sep = ',',names = ['state_id','station_id'])
#station_relative_data_target = station_relative_data.copy()
#station_relative_data_target.rename(columns = {'state_id':'target_state_id'},inplace = True)
#
### 8.匹配当前机器人的操作台位置
#kubotlocation_df = pd.merge(kubotlocation_df,station_relative_data,how = 'left',on = 'state_id')
#kubotlocation_df['location'] = np.where( (kubotlocation_df.station_id.isnull() == True),
#                                         kubotlocation_df.location,
#                                         '操作台')
#
### 9.匹配机器人休息区
#rest_zoom_file = 'rest_zoom.csv'
#rest_zoom_data = pd.read_csv(rest_zoom_file,sep = ',',names = ['x','y'])
#rest_zoom_data = rest_zoom_data.sort_values(by = 'y')
#rest_zoom_data = rest_zoom_data.reset_index(drop=True)
#rest_zoom_data['rest_id'] = list(range(1,len(rest_zoom_data)+1))
#rest_zoom_data_target = rest_zoom_data.copy()
#kubotlocation_df = pd.merge(kubotlocation_df,rest_zoom_data,how = 'left',on = ['x','y'])
#kubotlocation_df['location'] = np.where( (kubotlocation_df.rest_id.isnull() == True),
#                                         kubotlocation_df.location,
#                                         '充电桩')
#
#kubotlocation_df.rename(columns = {'location':'kubot_present_location',
#                                   'station_id':'precent_station_id',
#                                   'rest_id':'present_rest_id',
#                                   'x':'present_x',
#                                   'y':'present_y'},inplace = True)
#
### 匹配目标点操作台
#kubotlocation_df = pd.merge(kubotlocation_df,station_relative_data_target,how = 'left',on = 'target_state_id')
#kubotlocation_df['kubot_target_location'] = np.where( (kubotlocation_df.station_id.isnull() == True),'other','操作台')
#
### 匹配目标点休息区
##每个休息区应该也对应一个state_id
#rest_zoom_data_target = pd.merge(rest_zoom_data_target,kubotlocation_file_map,how = 'left',on = ['x','y'])
#del rest_zoom_data_target['x']
#del rest_zoom_data_target['y']
#rest_zoom_data_target.rename(columns = {'state_id':'target_state_id'},inplace = True)
#kubotlocation_df = pd.merge(kubotlocation_df,rest_zoom_data_target,how = 'left', on ='target_state_id')
#
#kubotlocation_df['kubot_target_location'] = np.where( (kubotlocation_df.rest_id.isnull() == True),
#                                                      kubotlocation_df['kubot_target_location'],
#                                                      '充电桩')
#
#kubotlocation_df['kubot_target_location'] = np.where(
#    (kubotlocation_df['kubot_target_location']) == 'other',
#    '货架区',
#    kubotlocation_df['kubot_target_location'])
#
### 需要对每个机器人来划分任务
##对每个机器人按照时间升序
#kubotlocation_df.sort_values(['kubot','time'],ascending=[True,True],inplace = True)
#
##基于上面的基础，当前位置是操作台 并且 目标位置是操作台，则认为这个时刻，mc_point ： 到达操作台
#kubotlocation_df['MC_Point'] = np.where(
#    (kubotlocation_df.kubot_present_location == '操作台') & \
#    (kubotlocation_df.kubot_target_location == '操作台'),
#    '到达操作台','other')
#
#kubotlocation_df['MC_Point'] = np.where(
#    (kubotlocation_df.kubot_present_location == '充电桩') & \
#    (kubotlocation_df.kubot_target_location == '充电桩'),
#    '到达充电桩',kubotlocation_df.MC_Point)
#kubotlocation_df['MC_Point_shift'] = kubotlocation_df.MC_Point.shift()
#kubotlocation_df['mc_id'] = np.where(kubotlocation_df.MC_Point_shift == '到达操作台',
#                                     kubotlocation_df.time + kubotlocation_df.kubot,
#                                     np.nan)
#kubotlocation_df['mc_id'] = np.where(kubotlocation_df.time == 0 ,
#                                     kubotlocation_df.time,
#                                     kubotlocation_df.mc_id)
#kubotlocation_df.mc_id.fillna(method = 'ffill',inplace = True)
#del kubotlocation_df['MC_Point_shift']
#del kubotlocation_df['rest_id']
#
#kubotlocation_df.reset_index(inplace = True)
#
### 操作台前三个位置是排队点
#arrive_station_index = pd.DataFrame(kubotlocation_df[kubotlocation_df['MC_Point'] == '到达操作台'].index.tolist(),
#                                    columns = ['arrive_station_index'])
#arrive_station_index['approchaing_1'] = arrive_station_index.arrive_station_index - 1
#arrive_station_index['approchaing_2'] = arrive_station_index.arrive_station_index - 2
#arrive_station_index['approchaing_3'] = arrive_station_index.arrive_station_index - 3
#
#
#approchaing_list_1 = list(arrive_station_index.approchaing_1)
#approchaing_list_2 = list(arrive_station_index.approchaing_2)
#approchaing_list_3 = list(arrive_station_index.approchaing_3)
#approchaing_list = approchaing_list_1 + approchaing_list_2 + approchaing_list_3
#
#kubotlocation_df['index'] = kubotlocation_df.index.tolist()
#kubotlocation_df['kubot_present_location']=np.where(
#    kubotlocation_df.index.isin(approchaing_list) == True,
#    '排队点',
#    kubotlocation_df.kubot_present_location
#)
#
#
#
#del kubotlocation_df['precent_station_id']
#del kubotlocation_df['present_rest_id']
#del kubotlocation_df['station_id']
#
### 统计每个机器人每一趟的任务总耗时
#kubotlocation_df.MC_Point = np.where (
#    kubotlocation_df.kubot_present_location == '排队点',
#    '到达排队点',
#    kubotlocation_df.MC_Point
#)
#kubotlocation_df.MC_Point = np.where (
#    ((kubotlocation_df.MC_Point == 'other')&(kubotlocation_df.alleyId == -1)),
#    '在主干道',
#    kubotlocation_df.MC_Point
#)
#
#kubotlocation_df.MC_Point = np.where (
#    kubotlocation_df.MC_Point == 'other',
#    '在货架区',
#    kubotlocation_df.MC_Point
#)
#
#del kubotlocation_df['theta_y']
#del kubotlocation_df['index']
#kubotlocation_df.rename(columns = {'theta_x':'present_theta'},inplace = True)
#
#engine = create_engine('mysql+pymysql://root:root@172.20.8.10:3306/FastSimulation_log',encoding='utf8')
#kubotlocation_df.to_sql(name='dwd_kubot_pp', con=engine,if_exists='append', chunksize=1000,index=None)
#ori_state_points.to_sql(name='dwd_state_points', con=engine,if_exists='append', chunksize=1000,index=None)
#alleyEachPoint.to_sql(name='dwd_alley', con=engine,if_exists='append', chunksize=1000,index=None)
#print('Get kubot_PP')
#rest_zoom_data.to_sql(name='dwd_rest_zoom', con=engine,if_exists='append', chunksize=1000,index=None)
#print('Get rest_zoom')
#"
#python -c """$script_kubot_pp"""
#
#
##table_name : kubot_station_progress
## time kubot_no station_id bin_id port_id kubot_status station_status 7
#cat $info_file | awk -F '[ ,]' '{switch($0) {case/bin; Kubot/: print $8","$11",,,,机器人到达操作台," ; break ;
#                                            case/WMS send unload prepare bin/: print $8","$18",,"$15",,wms发送指令让机器人准备卸箱子," ; break ;
#                                            case/WMS send request unload bin/: print $8","$18","$21","$15","$23",wms发送具体卸箱子指令给机器人,"; break ;
#                                            case/permit Kubot/: print $8","$14","$11",,,输送线允许机器人_"$15","; break;
#                                            case/WMS send unload bin/: print $8","$17",,"$14",,wms发送指令让机器人卸箱子,"; break ;
#                                            case/completed at conveyor./: print $8","$11",,"$14",,机器人已经把箱子卸到输送线上,"; break;
#                                            case/ready to process/: print $8",,"$11","$16",,,输送线让wms准备处理箱子"; break ;
#                                            case/start picking/: print $8",,"$11",,,,操作台开始处理箱子"; break ;
#                                            case/pick finished/: print $8",,"$18","$13",,,操作台结束处理这个箱子"; break;
#                                            case/move to conveyor/: print $8","$11",,,,机器人已经移动至取箱口,"; break ;
#                                            case/request load bin at Conveyor/: print $8","$13","$19",,"$21",wms要求让机器人在输送线取箱子,"; break ;
#                                            case/WMS send load bin/: print $8","$17",,"$14",,wms告诉机器可以取箱子了,"; break ;
#                                            case/WMS send load prepare for Kubot/: print $8","$16",,,,wms告诉机器人准备取箱子,"; break ;
#                                            case/load complete at conveyor/: print $8","$13","$18",,,wms告诉机器人该箱子取完成了," ; break ;
#                                            case/Set Free/: print $8","$13",,,,机器人可以离开操作台了," }}' > kubot_station_progress_sh.csv
#
#script_kubot_station_progress="
#import pandas as pd
#import numpy as np
#import pymysql
#from sqlalchemy import create_engine
#kubot_station_progress_sh= pd.read_csv('kubot_station_progress_sh.csv',names = ['time','kubot_no','station_id','bin_id','port_id','kubot_status','station_status'])
#
#kubot_station_progress_sh.station_id = np.where(
#            kubot_station_progress_sh.kubot_status == '机器人可以离开操作台了',
#    kubot_station_progress_sh.station_id.fillna(method = 'ffill'),
#    kubot_station_progress_sh.station_id
#)
#
#kubot_station_progress_sh.bin_id = np.where(
#            kubot_station_progress_sh.kubot_status == '机器人可以离开操作台了',
#    kubot_station_progress_sh.bin_id.fillna(method = 'ffill'),
#    kubot_station_progress_sh.bin_id
#)
#kubot_station_progress_sh.station_id.fillna(method = 'bfill',inplace = True)
#kubot_station_progress_sh.bin_id.fillna(method = 'bfill',inplace = True)
#
#
#kubot_station_progress_sh['progress_status'] = np.where(
#    kubot_station_progress_sh.kubot_status.isnull() == True,
#    kubot_station_progress_sh.station_status,
#    kubot_station_progress_sh.kubot_status
#)
#
#del kubot_station_progress_sh['kubot_status']
#del kubot_station_progress_sh['station_status']
#
#engine = create_engine('mysql+pymysql://root:root@172.20.8.10:3306/FastSimulation_log',encoding='utf8')
#kubot_station_progress_sh.to_sql(name='dwd_kubot_station_progress', con=engine,if_exists='append', chunksize=1000,index=None)
#print('Get kubot_station_progress')
#"
#python -c """$script_kubot_station_progress"""
#
#
#
#
#
#
#
#
##table_name : simulation_order
##table field :
#
## 把订单之前8列用不到的信息剔除，保留SKU及件數信息
#cut -d "," -f 8- $order_file > order_sku.csv
#
## 保存偶數列，也就是件數
#cat order_sku.csv | cut -d, -f$(seq -s, 2 2 999) > qty.csv
#
##保存奇數列，也就是sku列
#cat order_sku.csv | cut -d, -f$(seq -s, 1 2 999) > sku.csv
#
##qty和sku都整合到同一行
#sed -i ':a;N;$!ba;s/\n/ /g' qty.csv
#sed -i ':a;N;$!ba;s/\n/ /g' sku.csv
#
#
##要把逗號都替換為空格
#cat sku.csv | awk '{gsub(","," "); print $0 }' > sku_1.csv
#cat qty.csv | awk '{gsub(","," "); print $0 }' > qty_1.csv
#
#
##然後行轉列
#awk '{i=1;while(i <= NF){col[i]=col[i] $i " ";i=i+1}} END {i=1;while(i<=NF){print col[i];i=i+1}}' qty_1.csv | sed 's/[ \t]*$//g' > qty_2.csv
#awk '{i=1;while(i <= NF){col[i]=col[i] $i " ";i=i+1}} END {i=1;while(i<=NF){print col[i];i=i+1}}' sku_1.csv | sed 's/[ \t]*$//g' > sku_2.csv
#
##把sku和件數按照逗號為分隔符，合併
#paste -d "," sku_2.csv qty_2.csv | grep -v '^\s*$' > order_sku_1.csv
#
#rm -f sku_2.csv
#rm -f qty_2.csv
#rm -f sku.csv
#rm -f qty.csv
#rm -f sku_1.csv
#rm -f qty_1.csv
#rm -f order_sku.csv
#
##處理訂單及sku
##把订单之前6列用不到的信息剔除，保留SKU及件數信息
##只保留前兩列
#cut -d "," -f 6- $order_file  >  order_temp.csv
#cat order_temp.csv |  awk -F '[,]' '{print $1","$2}' > order_1.csv
#
##保留订单下发的第N个5分钟，及sku数
#cat $order_file | awk -F '[,]' '{print $2","$7}' > order_dispatch_temp.csv
#
#rm -f order_temp.csv
#
#script="
#import numpy as np
#import pandas as pd
#file = 'order_1.csv'
#data = pd.read_csv(file,sep=',', names = ['order_num','show_times'])
#ll = np.repeat(data.order_num,data.show_times)
#ll = pd.DataFrame(ll)
#ll.to_csv('order_sh.csv',index = False,header=False)
#
#file_sku = 'order_dispatch_temp.csv'
#data2 = pd.read_csv(file_sku,sep=',', names = ['dispatch_no','show_times'])
#ll2 = np.repeat(data2.dispatch_no,data.show_times)
#ll2 = pd.DataFrame(ll2)
#ll2.to_csv('order_dispatch.csv',index = False,header=False)
#"
#python -c """$script"""
#
#sed -i 's/\n//g' order_sh.csv
#
###合併訂單、sku、件數
#paste  -d ','  order_sh.csv  order_dispatch.csv  order_sku_1.csv   > simulation_order_temp.csv
#
#cat simulation_order_temp.csv | awk -F '[ , \r]' '{print $1","$3","$4","$2}' > simulation_order.csv
#rm -f simulation_order_temp.csv
#
##在首行插入列表名
#sed -i "1i order_num,sku,qty,dispatch_id" simulation_order.csv
#
#script_store_simulation_order="
#import numpy as np
#import pandas as pd
#import pymysql
#from sqlalchemy import create_engine
#project_namefile  = pd.read_csv('project_name.csv')
#project_name = project_namefile.columns.values.tolist()[0]
#
#simulation_order = pd.read_csv('simulation_order.csv',sep=',')
#simulation_inv = pd.read_csv('.../../data/'+project_name+'/inv.csv', names = ['bin_id','sku','qty'])
#
#engine = create_engine('mysql+pymysql://root:root@172.20.8.10:3306/FastSimulation_log',encoding='utf8')
#simulation_order.to_sql(name='dwd_simulation_order', con=engine,if_exists='append', chunksize=1000,index=None)
#print('Get simulation_order')
#simulation_inv.to_sql(name='dwd_simulation_inv', con=engine,if_exists='append', chunksize=1000,index=None)
#print('Get simulation_inv')
#"
#python -c """$script_store_simulation_order"""
#
#
##table name: location
##table field : location_id x y location_height location_type
#cat $locations_file | awk -F '[ ,]' '{print $1","$2","$3","$4","$8}' > location_sh.csv
#script_location="
#import pandas as pd
#import numpy as np
#import pymysql
#from sqlalchemy import create_engine
#
#location = pd.read_csv('location_sh.csv', names = ['location_id','x','y','location_height','location_type'])
#engine = create_engine('mysql+pymysql://root:root@172.20.8.10:3306/FastSimulation_log',encoding='utf8')
#location.to_sql(name='dwd_location', con=engine,if_exists='append', chunksize=1000,index=None)
#print('Get location')
#"
#
#
##table name: simulation_config
##table field : key value
#cat $config_file | awk -F '[:]' '{print $1","$2}' > config_sh.csv
#script_simulation_config="
#import pandas as pd
#import numpy as np
#import pymysql
#from sqlalchemy import create_engine
#
#simulation_config = pd.read_csv('config_sh.csv', names = ['key','value'])
#engine = create_engine('mysql+pymysql://root:root@172.20.8.10:3306/FastSimulation_log',encoding='utf8')
#simulation_config.to_sql(name='dwd_simulation_config', con=engine,if_exists='append', chunksize=1000,index=None)
#print('Get simulation_config')
#"
#python -c """$script_simulation_config"""
#
##table naem : station_config
##table field :
#
#script_station_config="
#import re
#import csv
#import json
#import pandas as  pd
#import numpy as np
#import pymysql
#from sqlalchemy import create_engine
#project_namefile  = pd.read_csv('project_name.csv')
#project_name = project_namefile.columns.values.tolist()[0]
#
#data = json.load(open('.../../data/'+project_name+'/station_config.json', 'r', encoding='utf-8'))
## print(data)
#
## data=json.load(obj)
## #2.遍历data，打印列表中的每一个字典
## station_id = None
## type = None
## x_ = None
## y_ = None
## order_slot_number = None
##
## single_item_order_slot_limit = None
## single_item_order_only = None
## unload_state_id = None
## unload_state_id_2 = None
## load_state_id = None
## load_state_id_2 = None
## conveyor_height = None
## conveyor_theta = None
#
## 1. 创建文件对象
#f = open('station_config.csv', 'w', newline ='',encoding='utf-8')
#
## 2. 基于文件对象构建 csv写入对象
#csv_writer = csv.writer(f)
## 3. 构建列表头
#csv_writer.writerow(
#    ['station_id', 'type', 'x_', 'y_', 'order_slot_number', 'single_item_order_slot_limit', 'single_item_order_only',
#     'unload_state_id', 'unload_state_id_2', 'load_state_id', 'load_state_id_2', 'target state id', 'conveyor_height',
#     'conveyor_theta'])
#
#for dict_data in data:
#    # print(len(dict_data))
#
#    station_id = str(dict_data['station id'])
#    type = str(dict_data['type'])
#
#    load = json.loads(json.dumps(dict_data['location']))
#    x_ = str(load['x'])
#    y_ = load['y']
#
#    line_Conveyer_JD = 'Conveyer_JD'
#    line_Normal = 'Normal'
#
#    # ret = jsonpath.jsonpath(data, '$..*')
#    # ret1 = jsonpath.jsonpath(data, '$.unload_state_id_2')
#    # print(ret1)
#
#    if len(dict_data) == 12:
#        # print('re打印',str(re_match))
#        order_slot_number = str(dict_data['order slot number'])
#        single_item_order_slot_limit = str(dict_data['station id'])
#        single_item_order_only = str(dict_data['single-item order only'])
#        unload_state_id = str(dict_data['unload state id'])
#        unload_state_id_2 = str(dict_data['unload state id 2'])
#        load_state_id = str(dict_data['load state id'])
#        load_state_id_2 = str(dict_data['load state id 2'])
#        conveyor_height = str(dict_data['conveyor height'])
#        conveyor_theta = str(dict_data['conveyor theta'])
#        # 4. 写入csv文件内容
#        csv_writer.writerow(
#            [station_id, type, x_, y_, order_slot_number, single_item_order_slot_limit, single_item_order_only,
#             unload_state_id, unload_state_id_2, load_state_id, load_state_id_2, None,conveyor_height, conveyor_theta])
#
#        # print(station_id, type, x_, y_, order_slot_number, single_item_order_slot_limit, single_item_order_only,
#        #       unload_state_id, load_state_id, conveyor_height, conveyor_theta)
#    if len(dict_data) == 10:
#
#        order_slot_number = str(dict_data['order slot number'])
#        single_item_order_slot_limit = str(dict_data['station id'])
#        single_item_order_only = str(dict_data['single-item order only'])
#        unload_state_id = str(dict_data['unload state id'])
#        load_state_id = str(dict_data['load state id'])
#        conveyor_height = str(dict_data['conveyor height'])
#        conveyor_theta = str(dict_data['conveyor theta'])
#        # 4. 写入csv文件内容
#        csv_writer.writerow(
#            [station_id, type, x_, y_, order_slot_number, single_item_order_slot_limit, single_item_order_only,
#             unload_state_id, None, load_state_id, None, None,conveyor_height, conveyor_theta])
#
#    elif re.match(type, line_Normal):
#
#        order_slot_number = str(dict_data['order slot number'])
#        single_item_order_slot_limit = str(dict_data['station id'])
#        single_item_order_only = str(dict_data['single-item order only'])
#        target_state_id = str(dict_data['target state id'])
#        # 4. 写入csv文件内容
#        csv_writer.writerow(
#            [station_id, type, x_, y_, order_slot_number, single_item_order_slot_limit, single_item_order_only,
#             None, None, None, None,target_state_id, None, None])
#
#        # print('station id=' + str(dict_data['station id']), 'type=' + str(dict_data['type']))
#        # print('station id=' + str(dict_data['station id']), 'type=' + str(dict_data['type']))
#        # print(station_id, type, x_, y_, order_slot_number, single_item_order_slot_limit, single_item_order_only,
#        #       target_state_id)
## 5. 关闭文件
#f.close()
#station_config = pd.read_csv('station_config.csv')
#engine = create_engine('mysql+pymysql://root:root@172.20.8.10:3306/FastSimulation_log',encoding='utf8')
#station_config.to_sql(name='dwd_station_config', con=engine,if_exists='append', chunksize=1000,index=None)
#
## print('最后', station_id, type, x_, y_, order_slot_number, single_item_order_slot_limit, single_item_order_only,
##       unload_state_id, load_state_id, conveyor_height, conveyor_theta)
#
#"
#
#python -c """$script_station_config"""
#else
#echo "project data exit aleady !"
#fi
#
#
#
#
#
#rm -f normal_order_dispatch.csv
#rm -f recombine_order_dispatch.csv
#rm -f normal_order_finish.csv
#rm -f recombine_order_finish.csv
#rm -f order_progress_fitst.csv
#rm -f order_progress_sh.csv
#rm -f kubot_progress_sh.csv
#rm -f bin_progress_sh.csv
#rm -f station_map.csv
#rm -f state_id.csv
#rm -f ori_state_id.csv
#rm -f ori_theta.csv
#rm -f rm -f ori_x.csv
#rm -f rm -f rm -f ori_y.csv
#rm -f ori_state_points.csv
#rm -f station_state_id.csv
#rm -f rest_zoom_x.csv
#rm -f rest_zoom_y.csv
#rm -f rest_zoom.csv
#rm -f alley.csv
#rm -f qr_code.csv
#rm -f station_id.csv
#rm -f play_back_speed.csv
#rm -f pp.csv
#rm -f station.csv
#rm -f kubot_station_progress_sh.csv
#rm -f config_sh.csv
#rm -f config_sh.csv
#rm -f location_sh.csv
#rm -f simulation_order.csv
#rm -f order_1.csv
#rm -f order_dispatch_temp.csv
#rm -f order_sku_1.csv
#rm -f pick_bin_time.csv
#rm -f order_dispatch.csv
#rm -f order_sh.csv
#rm -f order_dispatch.csv
#rm -f order_sku_1.csv
#rm -f order_dispatch_temp.csv
#rm -f order_1.csv
#rm -f simulation_order.csv
#rm -f info.log
#rm -f debug.log
#rm -f station_config.csv
#rm -f DB_project_name
#rm -f unique_key.csv