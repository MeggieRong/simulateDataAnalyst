#!/bin/bash

rm -rf longDis
mkdir longDis

file=info.log
#当箱子被机器人预约后
#KubotID;BinID;BinLoc;SKUID;Amount;DestID;ReqID
cat $file | grep "Reserve Stock" | awk -F '[ , ;]' '{print $8","$18","$19","$20","$21","$22","$23","$24}' > longDis/kubot_reserve_tock.csv




#当箱子被机器人成功load之后
#BinID;KubotID;Loc
cat $file | grep "Bin Loaded From Shel" | awk -F '[ ,;]' '{print $8","$16","$17","$18}' > longDis/bin_loaded_from_shelf.csv

#当箱子开始被处理之后
#[时间]  [箱子编号]  [sku]  [件数]  [操作台编号
cat $file | grep "OrderID" | grep "Amount" | awk -F '[ ,]' '{print $8","$20","$17","$23","$11}' > longDis/bin_start_process.csv

#当箱子被机器人从操作台load之后，开始归还，或者前往另一个操作台(如果启动跨操作台) 
# [时间]  [箱子编号]  [机器人编号 操作台编号
cat $file | grep "Kubot Load Bin From Conveyor" | awk -F '[ ,;]' '{print $8","$18","$19","$20}' > longDis/loaded_from_station.csv


#当箱子被机器人unload到货架上
#[时间]  [箱子编号]  [机器人编号]  [机器人的位置]
cat $file | grep "Bin Unload To Shelf" | awk -F '[ ,;]' '{print $8","$16","$17","$18}' > longDis/unloaded_to_shelf.csv


#机器人接到搬箱命令
#time kubotNo binID stationID
cat $file | grep "to load bin" |  awk -F '[ ,]' '{print $8","$11","$15,""}' > longDis/kubotAcceptMission.csv

#机器人接到前往的操作台的命令
#time kubot stationID
cat $file | grep "to destination"  |  awk -F '[ ,]' '{print $8","$11","$14}' > longDis/kubotToStation.csv







python3 LongDis.py