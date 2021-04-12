#!/bin/bash


#project_name=$1
project_name=New_DaLingShan-JD-20210308


if [ ! -d "output" ]; then
  mkdir output
fi
if [ ! -d "log_collection" ]; then
  mkdir log_collection
fi

echo 'project_name is '+$project_name

rm -rf output/${project_name}
rm -rf output/${project_name}/test
rm -rf output/${project_name}/other_detail_data
rm -rf output/${project_name}/order_graph
mkdir output/${project_name}
mkdir output/${project_name}/other_detail_data
mkdir output/${project_name}/test
mkdir output/${project_name}/order_detail
mkdir output/${project_name}/order_graph

cat ../log/${project_name}/simulator/*INFO*.log > log_collection/${project_name}_info.log

file_name=log_collection/${project_name}_info.log

file=../demo/${project_name}/order.csv

# time kubotNo missionNo
# mission begin
cat $file_name | awk '{switch($0) {case /MC->SS, Mission: kubot/: print  $8$12$14 ; break ; case /SolutionManager.cpp:/ : print  $8$10",SolutionManager" }}' > output/${project_name}/other_detail_data/kubotMissionBegin_temp.csv

#mission finish
#time kubotNo missionNo
cat $file_name | grep  "SS->MC" | grep "mission" | grep "completed" |  awk '{print $8$11","$13}' > output/${project_name}/other_detail_data/kubotMissionFinish.csv


cat output/${project_name}/other_detail_data/kubotMissionBegin_temp.csv | grep -v "allocation" |  awk -F '[ , :]' '{print $1","$2","$3}'  > output/${project_name}/other_detail_data/kubotMissionBegin.csv


rm -f output/${project_name}/other_detail_data/kubotMissionBegin_temp.csv

# 把订单之前8列用不到的信息剔除，保留SKU及件數信息
cut -d "," -f 8- $file > output/${project_name}/test/order_sku.csv

# 保存偶數列，也就是件數
cat output/${project_name}/test/order_sku.csv | cut -d, -f$(seq -s, 2 2 999) > output/${project_name}/test/qty.csv

#保存奇數列，也就是sku列
cat output/${project_name}/test/order_sku.csv | cut -d, -f$(seq -s, 1 2 999) > output/${project_name}/test/sku.csv

#qty和sku都整合到同一行
sed -i ':a;N;$!ba;s/\n/ /g' output/${project_name}/test/qty.csv
sed -i ':a;N;$!ba;s/\n/ /g' output/${project_name}/test/sku.csv


#要把逗號都替換為空格
cat output/${project_name}/test/sku.csv | awk '{gsub(","," "); print $0 }' > output/${project_name}/test/sku_1.csv
cat output/${project_name}/test/qty.csv | awk '{gsub(","," "); print $0 }' > output/${project_name}/test/qty_1.csv


#然後行轉列
awk '{i=1;while(i <= NF){col[i]=col[i] $i " ";i=i+1}} END {i=1;while(i<=NF){print col[i];i=i+1}}' output/${project_name}/test/qty_1.csv | sed 's/[ \t]*$//g' > output/${project_name}/test/qty_2.csv
awk '{i=1;while(i <= NF){col[i]=col[i] $i " ";i=i+1}} END {i=1;while(i<=NF){print col[i];i=i+1}}' output/${project_name}/test/sku_1.csv | sed 's/[ \t]*$//g' > output/${project_name}/test/sku_2.csv

#把sku和件數按照逗號為分隔符，合併
paste -d "," output/${project_name}/test/sku_2.csv output/${project_name}/test/qty_2.csv | grep -v '^\s*$' > output/${project_name}/test/order_sku_1.csv


rm -f output/${project_name}/test/sku_2.csv
rm -f output/${project_name}/test/qty_2.csv
rm -f output/${project_name}/test/sku.csv
rm -f output/${project_name}/test/qty.csv
rm -f output/${project_name}/test/sku_1.csv
rm -f output/${project_name}/test/qty_1.csv
rm -f output/${project_name}/test/order_sku.csv


#處理訂單及sku
#把订单之前6列用不到的信息剔除，保留SKU及件數信息
#只保留前兩列
cut -d "," -f 6- $file  >  output/${project_name}/test/order_temp.csv
cat output/${project_name}/test/order_temp.csv |  awk -F '[,]' '{print $1","$2}' > output/${project_name}/test/order_1.csv

#保留订单下发的第N个5分钟，及sku数
cat $file | awk -F '[,]' '{print $2","$7}' > output/${project_name}/test/order_dispatch_temp.csv

rm -f output/${project_name}/test/order_temp.csv

script="
import numpy as np
import pandas as pd
file = 'output/${project_name}/test/order_1.csv'
data = pd.read_csv(file,sep=',', names = ['order_num','show_times'])
ll = np.repeat(data.order_num,data.show_times)
ll = pd.DataFrame(ll)
ll.to_csv('output/${project_name}/test/order.csv',index = False,header=False)

file_sku = 'output/${project_name}/test/order_dispatch_temp.csv'
data2 = pd.read_csv(file_sku,sep=',', names = ['dispatch_no','show_times'])
ll2 = np.repeat(data2.dispatch_no,data.show_times)
ll2 = pd.DataFrame(ll2)
ll2.to_csv('output/${project_name}/test/order_dispatch.csv',index = False,header=False)
"
python3 -c """$script"""


sed -i 's/\n//g' output/${project_name}/test/order.csv


##合併訂單、sku、件數
paste  -d ','  output/${project_name}/test/order.csv  output/${project_name}/test/order_dispatch.csv  output/${project_name}/test/order_sku_1.csv   > output/${project_name}/simulation_order_temp.csv

cat output/${project_name}/simulation_order_temp.csv | awk -F '[ , \r]' '{print $1","$3","$4","$2}' > output/${project_name}/simulation_order.csv
rm -f output/${project_name}/simulation_order_temp.csv

#在首行插入列表名
sed -i "1i order_num,sku,qty,dispatch_id" output/${project_name}/simulation_order.csv


start=`cat $file_name  | grep "t =" | head -n 1 | awk '{print $8 }' | awk -F ',' '{print int($1)}' `
end=`cat $file_name  | grep "t =" | tail -n 1 | awk '{print $8 }' | awk -F ',' '{print int($1)}' `
total_second=$(($end-$start))


##倍速还原(快速模拟器的日志对于统计来说都是1倍速)
total_hour=`echo |  awk  "{print  ( $total_second / 3600)}"`



#計算總件數
total_qty=`cat $file_name | grep "OrderID" | grep ", Amount" | awk -F '[, ]' '{sum+= $23}END{print sum}'`


#计算总搬箱数
total_bins=`cat $file_name  | grep "carry" |  awk -F '[ ,]' '{print $13}' | grep -v 'carry' | awk '{sum += $0}END{print sum}'`

#入库总箱数
inbound_total_bins=`cat $file_name | grep "Kubot Load Bin From Conveyor" | awk -F '[ ,]' '{print $16}' | wc -w`


#操作台数量
station_num=`cat $file_name |  grep 'WMS->Station, Send Picking Action: Station' | awk -F '[ , ]' '{print $16}' | sort -u | wc -l`

#机器人数量
kubot_num=`cat $file_name   | grep "to destination"| awk '{print $10}' | sort -u | wc -l`


#系统每小时处理件数
qty_per_hour=$(printf "%.f\n" `echo "scale=6; $total_qty / $total_hour" | bc`)


#系统每小时入库箱子数
inbound_bin_num_each_hour=$(printf "%.f\n" `echo "scale=6; $inbound_total_bins / $total_hour" | bc`)


#每小时单机器人入库箱子数
inbound_bin_num_each_hour_each_kubot=$(printf "%.f\n" `echo "scale=6; $inbound_bin_num_each_hour / $kubot_num" | bc`)


#系统每小时出库箱子数
outbound_bin_num_each_hour=$(printf "%.f\n" `echo "scale=6; $total_bins / $total_hour" | bc`)


#每小时单机器人出库箱子数
outbound_bin_num_each_hour_each_kubot=$(printf "%.f\n" `echo "scale=6; $outbound_bin_num_each_hour / $kubot_num" | bc`)



#每小时每操作台处理件数
station_qty_per_hour=$(printf "%.f\n" `echo "scale=6; $qty_per_hour / $station_num" | bc`)



#出入库总箱子数
total_bins_all=`expr $total_bins + $inbound_total_bins`

# 系统每小时搬箱数
sys_bin_carry=$(printf "%.f\n" `echo "scale=6; $total_bins_all / $total_hour" | bc`)


#每小时每个机器人搬箱数
kubot_bin_carry=$(printf "%.f\n" `echo "scale=6; $sys_bin_carry / $kubot_num" | bc`)


#整体单箱命中率
mean_bin_fulfill_amount=$(printf "%.2f\n" `echo "scale=6; $total_qty / $total_bins" | bc`)


#整体单箱命中行数
Bin_mean_pay_order_numberTemp=`cat output/${project_name}/simulation_order.csv   | wc -l`
Bin_mean_pay_order_number=$(printf "%.2f\n" `echo "scale=6; ($Bin_mean_pay_order_numberTemp - 1)/$total_bins "| bc`)

#订单delay
delayAmount=`cat $file_name | grep "delay" | grep "Amount" | awk -F '[ ,]' '{sum += $23}END{print sum}'`

if [ ! $delayAmount  ];
then
  delayAmount=0
else
  delayAmount=`cat $file_name | grep "delay" | grep "Amount" | awk -F '[ ,]' '{sum += $23}END{print sum}'`
fi


#订单delay平均耗时
delayAvgTimeTemp=`cat $file_name | grep "delay" | grep "Amount" | awk -F '[ ,]' '{sum += $26}END{print sum/NR}'`
if [ ! $delayAvgTimeTemp  ];
then
  delayAvgTimeTemp=0
else
  delayAvgTimeTemp=`cat $file_name | grep "delay" | grep "Amount" | awk -F '[ ,]' '{sum += $26}END{print sum/NR}'`
fi
delayAvgTime=$(printf "%.2f\n" `echo $delayAvgTimeTemp `)


#最长挂单耗时(s)
maxDelayTime=`cat $file_name  | grep "delay" | grep "Amount" | awk -F '[ ,]' '{print $26}' |awk 'BEGIN {max = 0} {if ($0+0 > max+0) max=$0} END {print  max}'`
if [ ! $maxDelayTime  ];
then
  maxDelayTime=0
else
  maxDelayTime=`cat $file_name  | grep "delay" | grep "Amount" | awk -F '[ ,]' '{print $26}' |awk 'BEGIN {max = 0} {if ($0+0 > max+0) max=$0} END {print  max}'`
fi



#最短挂单耗时(s)
minDelayTime=`cat $file_name | grep "delay" | grep "Amount" | awk -F '[ ,]' '{print $26}' |awk 'BEGIN {min = 65536} {if ($0+0 < min+0) min=$0} END {print min}'`
if [ ! $dminDelayTime  ];
then
  minDelayTime=0
else
  minDelayTime=`cat $file_name | grep "delay" | grep "Amount" | awk -F '[ ,]' '{print $26}' |awk 'BEGIN {min = 65536} {if ($0+0 < min+0) min=$0} END {print min}'`
fi



echo "模拟基础数据,统计值" >> output/${project_name}/基础数据.csv
echo "操作台数量,$station_num" >> output/${project_name}/基础数据.csv
echo "机器人数量,$kubot_num" >> output/${project_name}/基础数据.csv
echo "总耗时(h),${total_hour}" >> output/${project_name}/基础数据.csv
echo "总出库件数,$total_qty" >> output/${project_name}/基础数据.csv
echo "出库件数/时,$qty_per_hour" >> output/${project_name}/基础数据.csv
echo "件数/时/操作台,$station_qty_per_hour" >> output/${project_name}/基础数据.csv
echo "总出库搬箱数,$total_bins" >> output/${project_name}/基础数据.csv
#echo "总入库搬箱数,$inbound_total_bins" >> output/${project_name}基础数据.csv
echo "出库箱数/时,$outbound_bin_num_each_hour" >> output/${project_name}/基础数据.csv
#echo "入库箱数/时,$inbound_bin_num_each_hour" >> output/${project_name}基础数据.csv
#echo "系统搬箱数/时,$sys_bin_carry" >> output/${project_name}基础数据.csv
echo "出库箱数/时/机器人,$outbound_bin_num_each_hour_each_kubot" >> output/${project_name}/基础数据.csv
#echo "入库箱数/时/机器人,$inbound_bin_num_each_hour_each_kubot" >> output/${project_name}基础数据.csv
#echo "系统搬箱数/时/机器人,$kubot_bin_carry" >> output/${project_name}基础数据.csv
echo "平均单箱命中件数,$mean_bin_fulfill_amount" >> output/${project_name}/基础数据.csv
echo "平均单箱命中行数,$Bin_mean_pay_order_number" >> output/${project_name}/基础数据.csv
echo "挂单总件数,$delayAmount" >> output/${project_name}/基础数据.csv
echo "平均单笔订单挂单耗时(s),$delayAvgTime" >> output/${project_name}/基础数据.csv
echo "最长挂单耗时(s),$maxDelayTime" >> output/${project_name}/基础数据.csv
echo "最短挂单耗时(s),$minDelayTime" >> output/${project_name}/基础数据.csv



#======================搬箱效率 得到的字段是 时间 搬运的箱子数量
cat $file_name | grep "bin; Kubot" |  awk  '{print $8$10","$12}' > output/${project_name}/other_detail_data/bin_carry.csv

#======================空闲率曲线 得到的字段是 时间 操作台编号 动作
cat $file_name  | awk -F '[, ]' '{switch($0)  {case/start picking/: print $8","$11","$12" "$13 ; break ; case /finish picking/: print $8","$11","$12" "$13 }}' > output/${project_name}/other_detail_data/station_work.csv


#机器人精准的工作时长
cat $file_name  | grep "is working" |  awk -F '[ ,]' '{print $8","$11",working"}' > output/${project_name}/other_detail_data/kubotWorking.csv
cat $file_name  | grep "is idle" |awk -F '[ ,]' '{print $8","$11",idle"}' > output/${project_name}/other_detail_data/kubotIDLE.csv
cat output/${project_name}/other_detail_data/kubotWorking.csv output/${project_name}/other_detail_data/kubotIDLE.csv > output/${project_name}/other_detail_data/kubotcarry.csv

cat $file_name | grep "to load bin" | awk -F '[ ,]' '{print $8","$11","$15}'  > output/${project_name}/other_detail_data/toLoadBin.csv

#机器人趟数和每一趟接到命令的间隔 时间 机器人编号  箱子id 操作台编号
cat $file_name | awk -F '[ ,]' '{switch($0) {case /to load bin/: print $8","$11","$15"," ; break ; case /to destination/: print $8","$11",,"$14}}' > output/${project_name}/other_detail_data/mcTime.csv


#======================每一笔订单的完成时刻 时间 订单编号
#多非
cat  $file_name  | grep "completed!" | grep "recombined Order" | awk '{print $8$11}' > output/${project_name}/other_detail_data/recombined_order_begin_finish.csv
#普通多合
cat  $file_name  | grep "completed!" | grep ", Order" | awk '{print $8$10}' > output/${project_name}/other_detail_data/normal_order_begin_finish.csv

#合并
cat output/${project_name}/other_detail_data/recombined_order_begin_finish.csv output/${project_name}/other_detail_data/normal_order_begin_finish.csv | awk '{print $0","""",""order_finish"}'> output/${project_name}/other_detail_data/order_finish.csv

rm -f output/${project_name}/other_detail_data/recombined_order_begin_finish.csv
rm -f output/${project_name}/other_detail_data/normal_order_begin_finish.csv


#======================每一笔订单每一个sku的完成  得到的字段是  时间  操作台  订单编号  sku  箱子编号 件数
cat $file_name   | grep "OrderID" |  awk -F '[, ]' '{print $8","$11","$14","$17","$20","$23}' > output/${project_name}/other_detail_data/order_finish_detail.csv




#======================订单下发时刻=====================
#合单订单下发时刻 字段是  时间 订单编号  操作台
cat $file_name  | grep "dispatch to Station" | grep "recombined Order" |  awk '{print $8$11","$16}' > output/${project_name}/other_detail_data/recombined_order_work_id.csv

#普通多合单订单下发时刻 字段是  时间 订单编号  操作台
cat $file_name  | grep "dispatch to Station" | grep ", Order" |  awk '{print $8$10","$14}' > output/${project_name}/other_detail_data/normal_order_work_id.csv

#多非与多合的下发合并在一起
cat output/${project_name}/other_detail_data/recombined_order_work_id.csv  output/${project_name}/other_detail_data/normal_order_work_id.csv  | awk '{print $0",""order_begin"}'> output/${project_name}/other_detail_data/order_begin.csv

rm -f output/${project_name}/other_detail_data/recombined_order_work_id.csv
rm -f output/${project_name}/other_detail_data/normal_order_work_id.csv

cat output/${project_name}/other_detail_data/order_finish.csv output/${project_name}/other_detail_data/order_begin.csv > output/${project_name}/other_detail_data/order_begin_and_finish.csv
rm -f output/${project_name}/other_detail_data/order_finish.csv
rm -f output/${project_name}/other_detail_data/order_begin.csv


#出库的件数，就是搬箱数
#时间 件数
cat $file_name  | grep ", Amount" | grep "OrderID" | awk -F '[, ]' '{print $8","$23}' > output/${project_name}/other_detail_data/outbouond.csv

#入库的时间 箱子编号
cat $file_name | grep "Kubot Load Bin From Conveyor" | awk -F '[ ,]' '{print $8","$16}' > output/${project_name}/other_detail_data/inbound.csv


#======================获取操作台个数
echo $station_num > output/${project_name}/other_detail_data/station_num.csv


#多非合单要把负数的订单编号映射到非负的订单编号，从而才能得到相应的
cat $file_name | grep "recombined Order" | awk -F '[: [ , ]' '{print $16","$19}' > output/${project_name}/other_detail_data/recombined_Order.csv


#想要知道每一个发放的订单的原始总件数需要多少件
#多非
cat $file_name  | grep "dispatch to Station" | grep "recombined Order" |  awk -F'[]: [ ]' '{print $15","$18}'  > output/${project_name}/other_detail_data/recombined_Order_list.csv
#多合
cat $file_name  | grep "dispatch to Station" | grep ", Order" | awk '{print $10}' > output/${project_name}/other_detail_data/normal_order_list.csv

#time binid kubotno stationno
cat $file_name  | grep "WMS send request" | awk '{print $8$14","$17","$20}' > output/${project_name}/other_detail_data/kubotCarryToStation.csv

#time binid kubotno stationno
cat $file_name  | grep "WMS send request" | awk '{print $8";"$14";"$17";"$20}' > output/${project_name}/other_detail_data/kubotCarryToStation2.csv


# time binID
cat $file_name | grep "used up" | awk '{print $8$10}' > output/${project_name}/other_detail_data/usedUpBin.csv

#挂单明细表 时间 操作台  订单编号 sku 箱子编号 件数 挂单时长
cat $file_name| grep "delay" | grep "Amount" | awk -F '[ ,]' '{print $8","$11","$14","$17","$20","$23","$26}' > output/${project_name}/other_detail_data/delay_detail.csv

#原始订单对应哪个波次 orderId , 波次
cat $file | awk -F '[ ,]' '{print $5","int($4/3600)}' >  output/${project_name}/other_detail_data/orderRound.csv

#获得内配大宗订单的编号
cat $file | awk -F '[ ,]' '{print $3","$5}' | grep "^0," | awk -F '[, ]' '{print $2",NeiPei"}' > output/${project_name}/other_detail_data/NeiPeiandDaZong.csv

cat $file_name | grep 'Send Picking Action' |  awk -F '[ , ]' '{print $8","$16","$19","$20}' > output/${project_name}/other_detail_data/binPick.csv

#获取每个订单的截止时间
cat $file | awk -F '[,]' '{print $5","$4}' >  output/${project_name}/other_detail_data/orderDeadline.csv

echo 'Simple Data Clean have done!'


python3 simulation_order_analysis.py $project_name



