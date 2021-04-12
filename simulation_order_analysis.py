# -*- encoding=UTF-8 -*-
import pandas as pd
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import warnings
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import time
import sys


warnings.filterwarnings("ignore")
for i in range(0, len(sys.argv)):
    project_name = sys.argv[i]

special_project_name = 'JD815-9Stations'

invFile_df = pd.read_csv('../demo/'+project_name +
                         '/inv.csv', names=['locationId', 'SKU', 'qty'])
inv_location = pd.read_csv('../demo/'+project_name+'/locations.csv', names=[
    'locationId', 'x', 'y', 'floor', 'nothing-1', 'nothing-2', 'nothing-3', 'locationType'])
order_file = 'output/'+project_name+'/simulation_order.csv'
order_df = pd.read_csv(order_file)

# ================================订单基础表===============================
# 增加订单类型（单件/多件订单）
order_df_add_type = pd.pivot_table(order_df, values=['qty'], index=[
    'order_num'], aggfunc=np.sum).reset_index()
order_df_add_type['order_type'] = np.where(
    order_df_add_type.qty == 1, 'single-order', 'multi-order')
order_df_add_type.drop(columns=['qty'], inplace=True)
order_df = pd.merge(order_df, order_df_add_type, on='order_num', how='left')


# 订单总件数
order_qty_sum = order_df['qty'].sum()

# 订单量
order_num = pd.DataFrame(
    order_df[['order_num', 'order_type']]).drop_duplicates()
order_num_count_distinct = order_num['order_num'].count()


# 订单行
order_line = len(order_df)

# sku品种数
sku_num = pd.DataFrame(order_df['sku']).drop_duplicates()
sku_num_count_distinct = sku_num['sku'].count()

# 件单比
avg_outbound_qty_per_order = round(order_qty_sum/order_num_count_distinct, 2)

# 行件比
avg_outbound_qty_per_order_line = round(order_qty_sum/order_line, 2)

# 单件订单数
order_num_single = order_num[order_num['order_type']
                             == 'single-order']['order_num'].count()

# 多件订单数
order_num_multi = order_num[order_num['order_type']
                            == 'multi-order']['order_num'].count()


# 多件件单比
avg_outbound_qty_per_order_multi = round(
    (order_qty_sum - order_num_single)/order_num_multi, 2)

# 单笔最大出库件数
order_df_max_qty = order_df.sort_values(
    by=['qty'], ascending=False).reset_index().loc[0, 'qty']

# 单笔多件订单最多包含sku品种数
order_num_multi_sku = order_df[order_df['order_type']
                               == 'multi-order'][['order_num', 'sku']]

if len(order_num_multi_sku) != 0:
    order_num_multi_sku_pivot = pd.pivot_table(order_num_multi_sku, values=[
        'sku'], index=['order_num'], aggfunc='count').reset_index()
    order_num_multi_maximun_sku_num = order_num_multi_sku_pivot.sort_values(
        by=['sku'], ascending=False).reset_index().loc[0, 'sku']
    # 多件订单平均每笔包含sku品种数
    order_num_multi_avg_sku_num = round(
        order_num_multi_sku_pivot['sku'].mean(), 2)
    # 保存表格就好
    order_df_multi_sku = order_df[order_df['order_type']
                                  == 'multi-order'][['order_num', 'sku']]
    order_df_multi_sku_distribution_order = pd.pivot_table(order_df_multi_sku, values=[
        'sku'], index=['order_num'], aggfunc='count').reset_index()
    order_df_multi_sku_distribution = pd.pivot_table(order_df_multi_sku_distribution_order, values=[
        'order_num'], index=['sku'], aggfunc='count').reset_index()
    order_df_multi_sku_distribution.rename(
        columns={'order_num': 'order_count'}, inplace=True)
    order_df_multi_sku_distribution_order_sum = order_df_multi_sku_distribution['order_count'].sum(
    )
    order_df_multi_sku_distribution['order_count_percentage(%)'] = round(
        order_df_multi_sku_distribution['order_count']/order_df_multi_sku_distribution_order_sum, 4)*100
else:
    order_num_multi_avg_sku_num = np.nan
    order_num_multi_maximun_sku_num = np.nan


# ==============下发到系统的订单
order_df['dispatch_hour'] = ((order_df['dispatch_id']*5)/60).apply(np.ceil)
order_dispatch = order_df[['order_num', 'dispatch_hour']]
order_dispatch = pd.pivot_table(order_dispatch, values=['order_num'], index=[
    'dispatch_hour'], aggfunc=pd.Series.nunique).reset_index()


# ==============下发到系统的件数条形图，显示标签
order_df_dispath_qty_to_sys = order_df[['dispatch_hour', 'qty']]
order_df_dispath_qtyTo_Sys = pd.pivot_table(order_df_dispath_qty_to_sys, values=[
    'qty'], index=['dispatch_hour'], aggfunc=np.sum).reset_index()


# EQ分析，根据出库件数，将订单划分范围
order_df_EQ = order_df[['order_num', 'qty']]
order_df_EQ_orig = order_df_EQ.copy()
order_df_EQ = pd.pivot_table(order_df_EQ, values=['qty'], index=[
    'order_num'], aggfunc=np.sum).reset_index()
order_df_EQ.sort_values(by=['qty'], ascending=False, inplace=True)
order_df_EQ.reset_index(drop=True, inplace=True)

# EQ画图
max_qty = max(order_df_EQ.qty)
step = 5
bin_list = list(range(0, int(max_qty)+2*step, step))
qty_list = order_df_EQ.qty.values.tolist()
order_EQ = pd.cut(qty_list, bins=bin_list, right=True)
order_EQ_df = pd.DataFrame(order_EQ.value_counts()).reset_index()
order_EQ_df.rename(columns={'index': 'Scope', 0: 'Order_Count'}, inplace=True)
order_EQ_df['Scope'] = order_EQ_df['Scope'].astype('str')
order_EQ_df['fre'] = range(1, len(order_EQ_df)+1)
order_EQ_df['fre'] = round(order_EQ_df['fre']/order_EQ_df['fre'].max(), 2)

# 只保留
order_EQ_df = order_EQ_df[order_EQ_df['fre'] <= 0.4]


order_df_EQ_order_line = pd.pivot_table(order_df[['order_num', 'qty']], values=[
    'qty'], index=['order_num'], aggfunc='count').reset_index()
order_df_EQ_order_line.sort_values(by=['qty'], ascending=False, inplace=True)
order_df_EQ_order_line.rename(columns={'qty': 'order_line'}, inplace=True)


# EQ-order_line画图
max_qty = max(order_df_EQ_order_line.order_line)
step = 1
bin_list = list(range(0, max_qty+2*step, step))
qty_list = order_df_EQ_order_line.order_line.values.tolist()
order_EQ_order_line = pd.cut(qty_list, bins=bin_list, right=True)
order_EQ_order_line = pd.DataFrame(
    order_EQ_order_line.value_counts()).reset_index()
order_EQ_order_line.rename(
    columns={'index': 'Scope', 0: 'Order_Count'}, inplace=True)
order_EQ_order_line['Scope'] = order_EQ_order_line['Scope'].astype('str')
order_EQ_order_line['fre'] = range(1, len(order_EQ_order_line)+1)
#order_EQ_order_line['fre'] = round(order_EQ_order_line['fre']/len(order_EQ_order_line),4)


# 保留前20的范围
order_EQ_order_line = order_EQ_order_line[order_EQ_order_line['fre'] <= 20]


# EIQ分析-IQ (Item Quantiry)
#SKU & qty
order_df_IQ = order_df[['sku', 'qty']]
order_df_IQ_orig = order_df_IQ.copy()
order_df_IQ = pd.pivot_table(order_df_IQ, values=['qty'], index=[
    'sku'], aggfunc=np.sum).reset_index()
order_df_IQ.sort_values(by=['qty'], ascending=False, inplace=True)
order_df_IQ.reset_index(drop=True, inplace=True)
order_df_IQ['sku_no'] = range(1, len(order_df_IQ)+1)
order_df_IQ['sku_cumsum_%'] = (
        round(order_df_IQ.sku_no/max(order_df_IQ.sku_no), 4)*100).astype('str')


order_df_IQ_order_line = order_df[['sku', 'qty']]
order_df_IQ_order_line = pd.pivot_table(order_df_IQ_order_line, values=[
    'qty'], index=['sku'], aggfunc='count').reset_index()
order_df_IQ_order_line.sort_values(by=['qty'], ascending=False, inplace=True)
order_df_IQ_order_line.reset_index(drop=True, inplace=True)
order_df_IQ_order_line['sku_no'] = range(1, len(order_df_IQ_order_line)+1)
order_df_IQ_order_line['sku_cumsum_%'] = (round(
    order_df_IQ_order_line.sku_no/max(order_df_IQ_order_line.sku_no), 4)*100).astype('str')
order_df_IQ_order_line.rename(
    columns={'qty': 'order_line_count'}, inplace=True)


print("Order Analysis Completed!")

# OP Analysis
orderFinishDetailFile = 'output/'+project_name + \
                        '/other_detail_data/order_finish_detail.csv'

#
# judge = input('Do u use haiport ?(Y/N)')
judge = 'Y'
if judge == 'Y' or judge == 'y':
    import ast
    file = pd.read_csv('output/'+project_name+'/other_detail_data/kubotCarryToStation2.csv',
                       names=['Time', 'BinId', 'KubotNo', 'StationNo'], sep=';')
    file['Time'] = file['Time'].apply(lambda x: x.replace(',', ''))
    file['BinId'] = file['BinId'].apply(
        lambda x: (x.replace('{', '[').replace('}', ']')))
    from re import search
    file_ = pd.DataFrame()
    file['repeatTime'] = np.nan
    for i in range(0, len(file)):
        if '[' in file['BinId'].iloc[i]:
            file['repeatTime'].iloc[i] = len(
                ast.literal_eval(file['BinId'].iloc[i]))
        else:
            file['repeatTime'].iloc[i] = 1

    Time = pd.DataFrame(np.repeat(file.Time, file.repeatTime))
    Time.reset_index(drop=True, inplace=True)
    Time = Time.apply(lambda x: x.astype('float64'))
    KubotNo = pd.DataFrame(np.repeat(file.KubotNo, file.repeatTime))
    KubotNo.reset_index(drop=True, inplace=True)
    StationNo = pd.DataFrame(np.repeat(file.StationNo, file.repeatTime))
    StationNo.reset_index(drop=True, inplace=True)

    BinId = pd.DataFrame()
    for i in range(0, len(file)):
        convertDf = pd.DataFrame(ast.literal_eval(file['BinId'].iloc[i]))
        BinId = BinId.append(convertDf)
    BinId.rename(columns={0: 'BinId'}, inplace=True)
    BinId.reset_index(drop=True, inplace=True)
    kubotCarryToStation = pd.concat([Time, BinId, KubotNo, StationNo], axis=1)

    kubot_action_file = kubotCarryToStation.copy()
    kubot_action_file_df = kubot_action_file.copy()

else:
    kubot_action_file = 'output/'+project_name + \
                        '/other_detail_data/kubotCarryToStation.csv'
    kubot_action_file_df = pd.read_csv(kubot_action_file, names=[
        'Time', 'BinId', 'KubotNo', 'StationNo'])


orderFinishDetailF_df = pd.read_csv(orderFinishDetailFile, names=[
    'Time', 'StationNo', 'OrderID', 'Sku', 'BinId', 'Qty'])


outbound = orderFinishDetailF_df[['Time', 'StationNo', 'Qty']]
outbound['hour'] = outbound['Time'].apply(lambda x: math.ceil(x/3600))

del outbound['Time']
outboundDf = pd.pivot_table(outbound, index=['hour'], values=['Qty', 'StationNo'], aggfunc={'StationNo': pd.Series.nunique, 'Qty': np.sum
                                                                                            }).reset_index()


outboundDf['qty'] = outboundDf['Qty'].copy()
outboundDf['qty_per_station_per_hour'] = outboundDf['qty'] / \
                                         outboundDf['StationNo']
outboundDf['qty_per_station_per_hour'] = outboundDf['qty_per_station_per_hour'].apply(
    lambda x: math.ceil(x))
outboundDf_fig = px.bar(outboundDf, x='hour', y='qty_per_station_per_hour',
                        width=800, height=500, text='qty_per_station_per_hour')
outboundDf_fig.update(
    layout=dict(
        xaxis=dict(title="模拟小时", tickangle=-0, showline=True, nticks=30),
        yaxis=dict(title="出库件数", showline=True),
        title='系统每小时每操作台出库件数'
    ))
outboundDf_fig.update_traces(textposition='outside')
pd.set_option('display.max_columns', None)


kubot_action_file_df = kubot_action_file_df.rename(
    columns={'time': 'Time', 'binNum': 'BinId', 'kubotNo': 'KubotNo', 'stationNo': 'StationNo'})

kubot_action_file_df['Qty'] = np.nan
kubot_action_file_df['Sku'] = np.nan
kubot_action_file_df['OrderID'] = np.nan
orderFinishDetailF_df['KubotNo'] = np.nan

cols = ['Time', 'StationNo', 'OrderID', 'Sku', 'BinId', 'Qty', 'KubotNo']
kubot_action_file_df = kubot_action_file_df[cols]
union_df1 = pd.concat([kubot_action_file_df, orderFinishDetailF_df], axis=0)


def station_bin_sort(df):
    df.sort_values(by=['BinId', 'Qty'], ascending=True, inplace=True)


StationNo = pd.DataFrame(union_df1['StationNo'].drop_duplicates(), columns=[
    'StationNo']).reset_index(drop=True)
StationNo.dropna(how='any', inplace=True)

# 平均每小时单箱命中率
bin_carry_for_hit_rate = kubot_action_file_df.copy()
bin_carry_for_hit_rate['hour'] = bin_carry_for_hit_rate.Time.apply(
    lambda x: math.ceil(x/3600))
bin_carry_for_hit_rate_df = pd.pivot_table(bin_carry_for_hit_rate, index=[
    'hour'], values=['BinId'], aggfunc='count').reset_index()

bin_carry_for_hit_rate_df2 = outboundDf[['hour', 'Qty']]
bin_carry_for_hit_rate_df = pd.merge(
    bin_carry_for_hit_rate_df2, bin_carry_for_hit_rate_df, how='left', on='hour')
bin_carry_for_hit_rate_df['avg_hit_rate'] = bin_carry_for_hit_rate_df['Qty'] / \
                                            bin_carry_for_hit_rate_df['BinId']
bin_carry_for_hit_rate_df['avg_hit_rate'] = bin_carry_for_hit_rate_df['avg_hit_rate'].apply(
    lambda x: round(x, 2))
bin_carry_for_hit_rate_df = bin_carry_for_hit_rate_df[['hour', 'avg_hit_rate']]
bin_carry_for_hit_rate_df_fig = px.line(bin_carry_for_hit_rate_df, x='hour',
                                        y='avg_hit_rate', text='avg_hit_rate',
                                        width=800, height=500)
bin_carry_for_hit_rate_df_fig.update(layout=dict(
    xaxis=dict(title="模拟小时", tickangle=-0, showline=True, nticks=30),
    yaxis=dict(title="平均单箱命中件数", showline=True),
    title='每小时平均单箱命中件数'))

# bin_carry_for_hit_rate_df_fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')


station_fill = pd.DataFrame()
for i in range(0, len(StationNo)):
    union_single_station = union_df1[union_df1['StationNo']
                                     == StationNo.loc[i, 'StationNo']]
    station_bin_sort(union_single_station)
    union_single_station['KubotNo'].fillna(method='bfill', inplace=True)
    union_single_station.dropna(subset=['Qty'], inplace=True)
    station_fill = station_fill.append(union_single_station)


# 每个操作台每小时接收到订单数、件数、sku品种数
stationOverview = orderFinishDetailF_df.copy()
stationOverview['hour'] = stationOverview['Time'].apply(
    lambda x: math.ceil(x/3600))
stationOverview_df = pd.pivot_table(stationOverview, index=['hour', 'StationNo'], values=['OrderID', 'Sku', 'Qty'], aggfunc={
    'OrderID': pd.Series.nunique, 'Sku': pd.Series.nunique, 'Qty': np.sum}).reset_index()
# 操作台订单一览表
stationOverview_df.rename(
    columns={'OrderID': 'orderCount', 'Sku': 'skuCount', 'Qty': 'sumQty'}, inplace=True)

# 平均操作台每小时处理订单数
avgEachStationProcessedOrderCount = round(
    stationOverview_df.orderCount.mean(), 0)

# 平均每个操作台每小时出库件数
avgEachStationProcessedsumQty = round(stationOverview_df.sumQty.mean(), 0)

# 平均每个操作台每小时下发sku品种数
avgEachStationProcessedSkuCouont = round(stationOverview_df.skuCount.mean(), 0)

# 操作台空闲率
stationWork = pd.read_csv('output/'+project_name+'/other_detail_data/binPick.csv',
                          names=['Time', 'StationNo', 'BinNo', 'BinProcessTime'])
stationWork['hour'] = stationWork['Time'].apply(lambda x: math.ceil(x/3600))

# 操作台空闲率总表
binProcessTimeEachStation = pd.pivot_table(stationWork, index=[
    'hour', 'StationNo'], values=['BinProcessTime'], aggfunc=np.sum).reset_index()
binProcessTimeEachStation['Idle_Rate'] = binProcessTimeEachStation['BinProcessTime'].apply(
    lambda x: round(1-(x/3600), 2))

# 平均每个操作台空闲率
max_hour = binProcessTimeEachStation.hour.max()
avgIdelRatePerStation = round(
    binProcessTimeEachStation[binProcessTimeEachStation.hour != max_hour].Idle_Rate.mean(), 2)


# 准备插入订单开始和结束时间
orderBeginFinishFile = 'output/'+project_name + \
                       '/other_detail_data/order_begin_and_finish.csv'
orderBeginFinish_df = pd.read_csv(orderBeginFinishFile, names=[
    'Time', 'OrderID', 'StationNo', 'OrderStatus'])


orderBegin_df = orderBeginFinish_df[orderBeginFinish_df['OrderStatus'] == 'order_begin'][[
    'Time', 'OrderID']].rename(columns={'Time': 'DispacthToSlotTime'})
orderFinish_df = orderBeginFinish_df[orderBeginFinish_df['OrderStatus'] == 'order_finish'][[
    'Time', 'OrderID']].rename(columns={'Time': 'FinishedTime'})


station_fill = pd.merge(station_fill, orderBegin_df,
                        how='left', on=['OrderID'])
station_fill = pd.merge(station_fill, orderFinish_df,
                        how='left', on=['OrderID'])


# 每个工作站出库的订单量/订单行数/出库件数
station_order_process = pd.pivot_table(station_fill, index=['StationNo'], values=[
    'OrderID', 'Qty'], aggfunc={'OrderID': [pd.Series.nunique, 'count'], 'Qty': np.sum})
station_order_process.columns = station_order_process.columns.droplevel(0)
station_order_process.reset_index(inplace=True)
station_order_process.rename(columns={
    'count': 'OrderCount', 'nunique': 'OrderLineCount', 'sum': 'QtySum'}, inplace=True)


stationSku = []
station = []
for i in range(0, len(StationNo)):
    station_sku = station_fill[station_fill['StationNo']
                               == StationNo.loc[i, 'StationNo']]
    SKU = pd.pivot_table(station_sku, index=['Sku'], values=[
        'OrderID'], aggfunc='count').reset_index().rename(columns={'OrderID': 'OrderLine'})
    sku_show_times = pd.DataFrame(station_sku['Sku'])
    sku_show_times['Sku_temp'] = sku_show_times['Sku'].shift()
    sku_show_times = sku_show_times[sku_show_times['Sku']
                                    != sku_show_times['Sku_temp']]
    sku_show_times_cal = pd.pivot_table(sku_show_times, index='Sku', values=[
        'Sku_temp'], aggfunc='count').reset_index()
    sku_show_times_cal.rename(
        columns={'Sku_temp': 'SkuShowTimes'}, inplace=True)
    SKU = pd.merge(SKU, sku_show_times_cal, how='left', on='Sku')
    SKU = SKU[SKU['SkuShowTimes'] != 0]
    SKU['平均单操作台单个sku每次命中订单数'] = round(SKU['OrderLine']/SKU['SkuShowTimes'], 2)
    AvgHitOrderline = round(SKU['平均单操作台单个sku每次命中订单数'].mean(), 2)
    stationSku.append(AvgHitOrderline)
    station.append(StationNo.loc[i, 'StationNo'])

SkuHitOrderLineEachStationEachTime = pd.DataFrame(
    {'StationNo': station, '平均单操作台单个sku每次命中订单数': stationSku})
SkuHitOrderLineEachStationEachTime_mean = round(
    SkuHitOrderLineEachStationEachTime['平均单操作台单个sku每次命中订单数'].mean(), 2)


# 京东重组优化映射
recombined_Order_list = pd.read_csv(
    'output/'+project_name+'/other_detail_data/recombined_Order_list.csv', sep='，', names=['temp_col'])
if len(recombined_Order_list) != 0:
    recombined_Order_list = pd.concat(
        [recombined_Order_list, recombined_Order_list['temp_col'].str.split(',', expand=True)], axis=1)
    recombined_Order_list = recombined_Order_list = recombined_Order_list.drop(columns=[
        'temp_col'])
    recombined_Order_list = recombined_Order_list.rename(
        columns={0: 're_order_id'})
    recombined_Order_list = recombined_Order_list.set_index(['re_order_id'])
    recombined_Order_list = recombined_Order_list.stack()
    pd.set_option('display.max_rows', None)
    recombined_Order_list.drop_duplicates(inplace=True)
    recombined_Order_list = recombined_Order_list.unstack()
    recombined_Order_list = recombined_Order_list.stack()
    recombined_Order_list = pd.DataFrame(recombined_Order_list)
    recombined_Order_list = recombined_Order_list.reset_index()
    recombined_Order_list = recombined_Order_list.rename(
        columns={0: 'order_id'})
    recombined_Order_list = recombined_Order_list[['re_order_id', 'order_id']]

    recombined_Order_list.to_csv(
        'output/'+project_name+'/other_detail_data/重组优化映射.csv', index=False, header=False)
else:
    recombined_Order_list = pd.DataFrame()
    recombined_Order_list.to_csv(
        'output/'+project_name+'/other_detail_data/重组优化映射.csv', index=False, header=False)

jd_judge = os.path.exists('output/'+project_name +
                          '/other_detail_data/重组优化映射.csv')

if jd_judge == True:
    recombinedOrder = pd.read_csv(
        'output/'+project_name+'/other_detail_data/重组优化映射.csv', names=['OrderID', 'OrigOrderID'])
    recombinedOrder.dropna(subset=['OrigOrderID'], inplace=True)
    station_fill = pd.merge(station_fill, recombinedOrder,
                            how='left', on=['OrderID'])
    station_fill['OrigOrderID'] = np.where(station_fill['OrigOrderID'].isnull(
    ) == True, station_fill['OrderID'], station_fill['OrigOrderID'])
else:
    station_fill['OrigOrderID'] = station_fill['OrderID'].copy()


# 原始订单下发到系统
OrigSimulationOrder = pd.read_csv(
    'output/'+project_name+'/simulation_order.csv')
OrigSimulationOrder.rename(columns={
    'order_num': 'OrigOrderID', 'dispatch_id': 'DispatchToSysTime'}, inplace=True)
OrigSimulationOrder.drop(columns=['sku', 'qty'], inplace=True)
OrigSimulationOrder['OrigOrderID'] = OrigSimulationOrder['OrigOrderID'].astype(
    str).apply(lambda x: x.split("C", 1)[1]).astype('float64')

OrigSimulationOrder['DispatchToSysTime'] = OrigSimulationOrder['DispatchToSysTime'].apply(
    lambda x: x*5*60)


station_fill = pd.merge(station_fill, OrigSimulationOrder,
                        how='left', on=['OrigOrderID'])

station_fill['下发到系统和下发到槽位的时间间隔(s)'] = round(
    station_fill.DispacthToSlotTime - station_fill.DispatchToSysTime, 0)
dispatchToSysAndSlot = round(station_fill['下发到系统和下发到槽位的时间间隔(s)'].mean(), 2)


# 平均每个订单下发到槽位至开始配订单行的时间间隔，那就是最小的配单Time
OrderDispatchProcess = pd.pivot_table(station_fill, index=[
    'OrderID', 'DispacthToSlotTime'], values=['Time'], aggfunc=min).reset_index()
OrderDispatchProcess['下发到槽位至开始配订单行的时间间隔(s)'] = (
                                                       OrderDispatchProcess.Time - OrderDispatchProcess.DispacthToSlotTime)/60

DispathToSlotAndStartProcess = round(
    OrderDispatchProcess['下发到槽位至开始配订单行的时间间隔(s)'].mean(), 2)


# 每个订单下发到槽位至订单完成的耗时
OrderDispatchFinished = station_fill[[
    'StationNo', 'OrderID', 'DispacthToSlotTime', 'FinishedTime']].drop_duplicates()
OrderDispatchFinished['平均每个订单下发到槽位至订单完成的耗时(min)'] = round(
    (OrderDispatchFinished.FinishedTime - OrderDispatchFinished.DispacthToSlotTime)/60, 0)

OrderStartAndOrderFinish = round(
    OrderDispatchFinished['平均每个订单下发到槽位至订单完成的耗时(min)'].mean(), 2)


# 每个订单完成平均耗时
orderBeginFinishFile_df = orderBeginFinish_df.sort_values(
    by=['OrderID', 'Time'])
orderBeginFinishFile_df.StationNo.fillna(method='ffill', inplace=True)

orderBeginFinishFile_df['Time_shift'] = orderBeginFinishFile_df.Time.shift()
orderBeginFinishFile_df['TimeDiff(min)'] = round(
    (orderBeginFinishFile_df.Time - orderBeginFinishFile_df.Time_shift)/60, 2)
orderBeginFinishFileSingleOrderTime = orderBeginFinishFile_df[
    orderBeginFinishFile_df['OrderStatus'] == 'order_finish']


# 整体平均每个订单完成的耗时min
sysOrderProcessTotalTime = round(
    orderBeginFinishFileSingleOrderTime['TimeDiff(min)'].mean(), 2)


orderBeginFinishFileSingleOrderTime['hour'] = orderBeginFinishFileSingleOrderTime['Time'].apply(
    lambda x: math.ceil(x/3600))

orderBeginFinishFileSingleOrderTime_bar = pd.pivot_table(orderBeginFinishFileSingleOrderTime[['hour', 'TimeDiff(min)']],
                                                         index=['hour'], values=['TimeDiff(min)'], aggfunc=np.mean).reset_index()

orderBeginFinishFileSingleOrderTime_bar['TimeDiff(min)'] = orderBeginFinishFileSingleOrderTime_bar['TimeDiff(min)'].apply(
    lambda x: round(x, 2))


# 平均每个箱子的处理耗时
bin_process = pd.read_csv('output/'+project_name+'/other_detail_data/station_work.csv',
                          names=['Time', 'StationNo', 'BinAction'])
bin_process.sort_values(by=['StationNo', 'Time'], ascending=True, inplace=True)
bin_process['TimeShift'] = bin_process['Time'].shift().fillna(value=0)
bin_process['BinProcessTime'] = bin_process.Time - bin_process.TimeShift
bin_process = bin_process[bin_process['BinAction'] == 'finish picking'].drop(
    columns=['BinAction', 'TimeShift'])

# 把这个箱子的处理耗时，添加到总表里面
bin_process.drop(columns=['StationNo'], inplace=True)

# right
bin_processBar_ = stationWork.copy()
bin_processBar_ = bin_processBar_[bin_processBar_['BinProcessTime'] <= 30]
bin_processBar_['BinProcessTimeCount'] = bin_processBar_[
    'BinProcessTime'].copy()
bin_processBar_fig1 = pd.pivot_table(bin_processBar_, index=['BinProcessTime'], values=[
    'BinProcessTimeCount'], aggfunc='count').reset_index()

bin_process_Bar = px.bar(bin_processBar_fig1, x='BinProcessTime',
                         y='BinProcessTimeCount', width=800, height=500, text='BinProcessTimeCount')

bin_process_Bar.update(
    layout=dict(
        xaxis=dict(title="单箱操作耗时(s)", tickangle=-0, showline=True, nticks=30),
        yaxis=dict(title="计数", showline=True),
        title='单箱操作分布图(剔除超过30s的数据)'
    ))
bin_process_Bar.update_traces(textposition='outside')


station_fill = pd.merge(station_fill, bin_process, how='left', on=['Time'])
bin_processTime = round(station_fill['BinProcessTime'].mode(), 2)
bin_processTime = bin_processTime[0]
# print(bin_processTime[0])

station_fill['BinID'] = station_fill['BinId']
QtyAndOrderLineProcess = pd.pivot_table(station_fill, index=['Time', 'StationNo', 'BinId'], values=['BinID', 'Qty', 'BinProcessTime'],
                                        aggfunc={'BinID': 'count', 'Qty': np.sum, 'BinProcessTime': np.mean}).reset_index()
QtyAndOrderLineProcess.rename(columns={'BinID': 'OrderLine'}, inplace=True)
QtyAndOrderLineProcess['平均处理1个订单行耗时(s)'] = round(
    QtyAndOrderLineProcess.BinProcessTime/QtyAndOrderLineProcess.OrderLine, 2)
QtyAndOrderLineProcess['平均拣1件要多少时间(s)'] = round(
    QtyAndOrderLineProcess.BinProcessTime/QtyAndOrderLineProcess.Qty, 2)

QtyAndOrderLineProcessTime = round(
    QtyAndOrderLineProcess['平均处理1个订单行耗时(s)'].mean(), 2)
QtyAndOrderLineProcessQtyTime = round(
    QtyAndOrderLineProcess['平均拣1件要多少时间(s)'].mean(), 2)


# 下发到槽位的件数-小时序图
dispatchToSlot_bar_df = station_fill[[
    'StationNo', 'Qty', 'DispacthToSlotTime']]
dispatchToSlot_bar_df['hour'] = dispatchToSlot_bar_df['DispacthToSlotTime'].apply(
    lambda x: math.ceil(x/3600))
dispatchToSlot_bar = pd.pivot_table(dispatchToSlot_bar_df, index=[
    'hour'], values=['Qty'], aggfunc=np.sum).reset_index()


# 下发到各操作台槽位的件数-时序图
dispatchToSlotEachStation_bar_df = station_fill[[
    'StationNo', 'Qty', 'DispacthToSlotTime']]
dispatchToSlotEachStation_bar_df['hour'] = dispatchToSlotEachStation_bar_df['DispacthToSlotTime'].apply(
    lambda x: math.ceil(x/3600))
dispatchToSlotEachStation_bar = pd.pivot_table(dispatchToSlotEachStation_bar_df, index=[
    'StationNo', 'hour'], values=['Qty'], aggfunc=np.sum).reset_index()


# 下发到槽位的订单数-时序图
dispatchOrderCountToSlot_bar_df = station_fill[[
    'StationNo', 'OrderID', 'DispacthToSlotTime']]
dispatchOrderCountToSlot_bar_df['hour'] = dispatchOrderCountToSlot_bar_df['DispacthToSlotTime'].apply(
    lambda x: math.ceil(x/3600))
dispatchOrderCountToSlot_bar = pd.pivot_table(dispatchOrderCountToSlot_bar_df, index=[
    'hour'], values=['OrderID'], aggfunc=pd.Series.nunique).reset_index()
#dispatchOrderCountToSlot_bar.to_csv('下发到槽位的订单数-时序图.csv',index = False)

# 下发到各操作台槽位的订单数-时序图
dispatchOrderCountToSlotEachStation_bar_df = station_fill[[
    'StationNo', 'OrderID', 'DispacthToSlotTime']]
dispatchOrderCountToSlotEachStation_bar_df['hour'] = dispatchOrderCountToSlotEachStation_bar_df['DispacthToSlotTime'].apply(
    lambda x: math.ceil(x/3600))
dispatchOrderCountToSlotEachStation_bar = pd.pivot_table(dispatchOrderCountToSlotEachStation_bar_df, index=[
    'StationNo', 'hour'], values=['OrderID'], aggfunc=pd.Series.nunique).reset_index()


# 下发到槽位的sku品种数-时序图
dispatchSkuCountToSlot_bar_df = station_fill[[
    'StationNo', 'Sku', 'DispacthToSlotTime']]
dispatchSkuCountToSlot_bar_df['hour'] = dispatchSkuCountToSlot_bar_df['DispacthToSlotTime'].apply(
    lambda x: math.ceil(x/3600))
dispatchSkuCountToSlot_bar = pd.pivot_table(dispatchSkuCountToSlot_bar_df, index=[
    'hour'], values=['Sku'], aggfunc=pd.Series.nunique).reset_index()


# 下发到各操作台槽位的sku品种数-时序图
dispatchSkuCountToSlotEachStation_bar_df = station_fill[[
    'StationNo', 'Sku', 'DispacthToSlotTime']]
dispatchSkuCountToSlotEachStation_bar_df['hour'] = dispatchSkuCountToSlotEachStation_bar_df['DispacthToSlotTime'].apply(
    lambda x: math.ceil(x/3600))
dispatchSkuCountToSlotEachStation_bar = pd.pivot_table(dispatchSkuCountToSlotEachStation_bar_df, index=[
    'StationNo', 'hour'], values=['Sku'], aggfunc=pd.Series.nunique).reset_index()


print("OP Analysis Completed!")
# MC Analysis

# 单纯搬箱图
# 入库小时搬箱
inbound_hour_df = pd.read_csv(
    'output/'+project_name+'/other_detail_data/inbound.csv', names=['Time', 'binId'])

# 出库小时搬箱
outbound_hour_df = pd.read_csv(
    'output/'+project_name+'/other_detail_data/bin_carry.csv', names=['Time', 'KubotNo', 'BinNum'])
outbound_hour_df['hour'] = outbound_hour_df.Time.apply(
    lambda x: math.ceil(x/3600))

# 每小时出库搬箱
outboundBinEachHour = pd.pivot_table(outbound_hour_df, index=['hour'], values=[
    'BinNum'], aggfunc=np.sum).reset_index()
outboundBinEachHour.rename(columns={'BinNum': 'outboundBinNum'}, inplace=True)


# 每小时工作机器人数
working_kubot = outbound_hour_df.copy()
working_kubot['hour'] = working_kubot['Time'].apply(
    lambda x: math.ceil(x/3600))
working_kubotDf = pd.pivot_table(working_kubot, index=['hour'], values=[
    'KubotNo'], aggfunc=pd.Series.nunique).reset_index()


working_kubotDf_fig = px.bar(
    working_kubotDf, x='hour', y='KubotNo', width=800, height=500, text='KubotNo')
working_kubotDf_fig.update(
    layout=dict(
        xaxis=dict(title="模拟小时", tickangle=-0, showline=True, nticks=30),
        yaxis=dict(title="工作机器人数", showline=True),
        title='每小时工作机器人数'
    ))
working_kubotDf_fig.update_traces(textposition='outside')

# 每小时平均每个机器人每趟搬箱数
roundworkingKubotBincarry = outbound_hour_df.copy()
roundworkingKubotBincarry['hour'] = roundworkingKubotBincarry['Time'].apply(
    lambda x: math.ceil(x/3600))
roundworkingKubotBincarry_df = pd.pivot_table(roundworkingKubotBincarry, index=['hour'], values=['KubotNo', 'BinNum'],
                                              aggfunc={'KubotNo': 'count', 'BinNum': np.sum}).reset_index()

roundworkingKubotBincarry_df['carryBinEachRound'] = round(
    roundworkingKubotBincarry_df.BinNum / roundworkingKubotBincarry_df.KubotNo, 0)


roundworkingKubotBincarry_df_fig = px.bar(
    roundworkingKubotBincarry_df, x='hour', y='carryBinEachRound', width=800, height=500, text='carryBinEachRound')
roundworkingKubotBincarry_df_fig.update(
    layout=dict(
        xaxis=dict(title="模拟小时", tickangle=-0, showline=True, nticks=30),
        yaxis=dict(title="平均每趟搬箱数", showline=True),
        title='机器人平均每趟搬箱数'
    ))
roundworkingKubotBincarry_df_fig.update_traces(textposition='outside')

if len(inbound_hour_df) != 0:
    inbound_hour_df['hour'] = inbound_hour_df.Time.apply(
        lambda x: math.ceil(x/3600))
    # 每小时入库箱子数
    inboundBinEachHour = pd.pivot_table(inbound_hour_df, index=['hour'], values=[
        'binId'], aggfunc='count').reset_index()
    inboundBinEachHour.rename(columns={'binId': 'inboundBinNum'}, inplace=True)

    # 系统搬箱数
    sys_bin_hour_df = pd.merge(
        inboundBinEachHour, outboundBinEachHour, how='outer', on=['hour'])
    sys_bin_hour_df = sys_bin_hour_df.fillna(0)
    sys_bin_hour_df['sys_total_binNum'] = sys_bin_hour_df.inboundBinNum + \
                                          sys_bin_hour_df.outboundBinNum

    # 系统每小时搬箱数
    sys_bin_hour_df_bar = sys_bin_hour_df[['hour', 'sys_total_binNum']]
else:
    sys_bin_hour_df = outboundBinEachHour.copy()
    sys_bin_hour_df = sys_bin_hour_df.fillna(0)
    sys_bin_hour_df['sys_total_binNum'] = sys_bin_hour_df['outboundBinNum']
    inboundBinEachHour = pd.DataFrame()
    sys_bin_hour_df_bar = pd.DataFrame()


station_fill_temp = station_fill[[
    'Time', 'StationNo', 'BinId']].drop_duplicates()
BinPickSkuEachStationEachTime = pd.pivot_table(
    station_fill, index=['Time'], values=['Sku'], aggfunc='count').reset_index()
BinPickSkuEachStationEachTime = pd.merge(
    BinPickSkuEachStationEachTime, station_fill_temp, how='left', on=['Time'])

BinPickSkuEachStationEachTime_df = pd.pivot_table(BinPickSkuEachStationEachTime, index=[
    'StationNo', 'BinId'], values=['Sku'], aggfunc=pd.Series.nunique).reset_index()
BinPickSkuEachStationEachTimeDf = pd.pivot_table(BinPickSkuEachStationEachTime_df, index=[
    'StationNo'], values=['Sku'], aggfunc=np.mean).reset_index()
BinPickSkuEachStationEachTimeDf.rename(
    columns={'Sku': '平均每个工作站单个料箱命中sku品种数'}, inplace=True)

BinPickSkuEachStationEachTime_num = round(
    BinPickSkuEachStationEachTimeDf['平均每个工作站单个料箱命中sku品种数'].mean(), 2)


# 平均每个料箱命中几个工作站
# BinId StationNo 先 distinct 后 groupby
BinHitStaton = station_fill[['BinId', 'StationNo']].drop_duplicates()
BinHitStaton_df = pd.pivot_table(BinHitStaton, index=['BinId'], values=[
    'StationNo'], aggfunc='count').reset_index()
BinHitStaton_df['StationNo'].mean()

avgBinHitStaton = round(BinHitStaton_df['StationNo'].mean(), 2)


# #带有箱子空与否状态的总表
# #station_fill
# 单操作台箱子重取率
# 该操作台搬运的箱子中，有多少是搬运2次及以上的
# 排序用station和time来拍

BinRepeatEachStation = station_fill[[
    'BinId', 'StationNo', 'Time']].sort_values(by=['StationNo', 'Time'])

BinRepeatEachStation['Time_shift'] = BinRepeatEachStation['Time'].shift()
BinRepeatEachStation = BinRepeatEachStation[BinRepeatEachStation['Time']
                                            != BinRepeatEachStation['Time_shift']]
BinRepeatEachStation.drop(columns=['Time_shift'], inplace=True)
BinRepeatEachStation['BinID'] = BinRepeatEachStation['BinId'].copy()
BinRepeatEachStation_df = pd.pivot_table(BinRepeatEachStation, index=[
    'StationNo', 'BinId'], values=['BinID'], aggfunc='count').reset_index()
BinRepeatEachStation_df.rename(
    columns={'BinID': 'BinCarried_SameStation'}, inplace=True)
BinRepeatEachStation_df['BinType'] = np.where(
    BinRepeatEachStation_df['BinCarried_SameStation'] >= 2, 'repeatBin', 'OneTimeBin')


# 同操作台重取总表
# BinRepeatEachStation_df


StationTotalBinCarryDf = station_fill[[
    'BinId', 'StationNo', 'Time']].sort_values(by=['Time'])
StationTotalBinCarryDf.drop_duplicates(inplace=True)


StationTotalBinCarry = StationTotalBinCarryDf.groupby(['StationNo'])[
    'BinId'].count()


StationTotalRepeatBinCarry = BinRepeatEachStation_df[BinRepeatEachStation_df['BinType'] == 'repeatBin'].groupby([
    'StationNo'])['BinId'].count()
StationTotalRepeatBinCarry = StationTotalRepeatBinCarry.to_frame(
).reset_index().rename(columns={'BinId': 'RepeatBinCount'})

StationTotalBinCarry = StationTotalBinCarry.to_frame(
).reset_index().rename(columns={'BinId': 'TotalBinCount'})


StationBin_RepeatRate = pd.merge(
    StationTotalBinCarry, StationTotalRepeatBinCarry, how='left', on=['StationNo'])
StationBin_RepeatRate['BinRepeatRate(%)'] = round(
    (StationBin_RepeatRate['RepeatBinCount']/StationBin_RepeatRate['TotalBinCount'])*100, 2)


StationBin_RepeatRateAvg = round(
    StationBin_RepeatRate['BinRepeatRate(%)'].mean(), 0)/100
StationBinPicked = math.ceil(StationBin_RepeatRate['TotalBinCount'].mean())


# 哪个箱子在什么时候变成了空箱
emptyBinFile = 'output/'+project_name+'/other_detail_data/usedUpBin.csv'
emptyBin_df = pd.read_csv(emptyBinFile, names=['Time', 'BinId'])


# 模拟没有空箱取出
if len(emptyBin_df) != 0:
    emptyBin_df['BinStatus'] = 'Empty'
    station_fill = pd.merge(station_fill, emptyBin_df,
                            how='left', on=['Time', 'BinId'])
    station_fill['BinStatus'].fillna(value='NotEmpty', inplace=True)
    stationEmptyBinCount = pd.pivot_table(station_fill[station_fill['BinStatus'] == 'Empty'], index=[
        'StationNo'], values=['BinId'], aggfunc=pd.Series.nunique).reset_index()
    stationEmptyBinCount.rename(
        columns={'BinId': 'EmptyBinCount'}, inplace=True)
    # 空箱表
    EmptyStationTable = pd.merge(
        StationTotalBinCarry, stationEmptyBinCount, how='left', on=['StationNo'])
    EmptyStationTable['空箱取出占比(%)'] = round(
        (EmptyStationTable.EmptyBinCount/EmptyStationTable.TotalBinCount)*100, 2)
    emptyBin_df['hour'] = emptyBin_df['Time'].apply(
        lambda x: math.ceil(x/3600))
    emptyBin_df_hourCount = pd.pivot_table(emptyBin_df, index=['hour'], values=[
        'BinId'], aggfunc='count').reset_index()

    emptyBinCount = emptyBin_df_hourCount.BinId.sum()
else:
    emptyBinCount = 0
    emptyBin_df_hourCount = pd.DataFrame()
    EmptyStationTable = pd.DataFrame()


# 同操作台重取分布，就是已经是重取的箱子，重取次数分布
BinRepeatEachStation_df_repeat = BinRepeatEachStation_df[
    BinRepeatEachStation_df['BinType'] == 'repeatBin']
BinRepeatEachStation_df_repeat_Distribu = pd.pivot_table(BinRepeatEachStation_df_repeat, index=['StationNo', 'BinCarried_SameStation'],
                                                         values=['BinId'], aggfunc='count')


# 同操作台重取分布
BinRepeatEachStation_df_repeat_Distribu.reset_index(inplace=True)


BinRepeatEachStation_overOneStation = BinRepeatEachStation.drop(columns=[
    'BinID', 'Time'])
BinRepeatEachStation_overOneStation_df = pd.pivot_table(BinRepeatEachStation_overOneStation, index=[
    'BinId'], values=['StationNo'], aggfunc=pd.Series.nunique).reset_index()
BinRepeatEachStation_overOneStation_df.rename(
    columns={'StationNo': '跨操作台个数'}, inplace=True)

# 跨操作台总表
# BinRepeatEachStation_overOneStation_df

# 整体的跨操作台重取率 有跨操作台行为的箱子个数(去重)/总箱子个数(去重)
AllCarriedBin = BinRepeatEachStation_overOneStation_df['BinId'].count()
OverStation_Bin = BinRepeatEachStation_overOneStation_df[
    BinRepeatEachStation_overOneStation_df['跨操作台个数'] >= 2]['BinId'].count()


# 整体的跨操作台重取率
OverStation_BinCarried_Rate = round(OverStation_Bin/AllCarriedBin, 2)

# 多个操作台重取分布
BinRepeatEachStation_overOneStation_df_distribu = pd.pivot_table(BinRepeatEachStation_overOneStation_df, index=[
    '跨操作台个数'], values=['BinId'], aggfunc=pd.Series.nunique).reset_index()


BinRepeatEachStation_overOneStation_df_distribu['BinId'] = BinRepeatEachStation_overOneStation_df_distribu['BinId'].apply(
    lambda x: x + 1)
BinRepeatEachStation_overOneStation_df_distribu.rename(
    columns={'BinId': '箱子个数'}, inplace=True)
BinRepeatEachStation_overOneStation_df_distribu['箱子占比(%)'] = round(
    BinRepeatEachStation_overOneStation_df_distribu['箱子个数']/AllCarriedBin, 2)*100

# 每个操作台每小时接受的箱子个数
StationBinAcceptEachHour = station_fill[[
    'Time', 'StationNo', 'BinId']].drop_duplicates()
StationBinAcceptEachHour['hour'] = StationBinAcceptEachHour['Time'].apply(
    lambda x: math.ceil(x/3600))
StationBinAcceptEachHour_df = pd.pivot_table(StationBinAcceptEachHour, index=[
    'StationNo', 'hour'], values=['Time'], aggfunc='count').reset_index()
StationBinAcceptEachHour_df.rename(columns={'Time': 'BinCount'}, inplace=True)

StationBinAcceptEachHourAvg = math.ceil(
    StationBinAcceptEachHour_df['BinCount'].mean())


# 机器人搬箱表
bin_carry_df = pd.read_csv(
    'output/'+project_name+'/other_detail_data/bin_carry.csv', names=['Time', 'KubotNo', 'BinNum'])
bin_carry_df['hour'] = bin_carry_df['Time'].apply(lambda x: math.ceil(x/3600))


bin_carry_df['hour'] = bin_carry_df['Time'].apply(lambda x: math.ceil(x/3600))

# 机器人精确工作时长
kubot_mc_time = pd.read_csv(
    'output/'+project_name+'/other_detail_data/kubotcarry.csv', names=['Time', 'KubotNo', 'action'])
kubot_mc_time.sort_values(by=['KubotNo', 'Time'], inplace=True)
kubot_mc_time.reset_index(drop=True, inplace=True)
# 机器人数量
kubotNum = len(kubot_mc_time.groupby('KubotNo').count().index)

KUBOTWORKtIME = pd.DataFrame()
KUBOTWORKtIMEEachHour = pd.DataFrame()
# working 到 idle 算一组，里面可能会包括很多趟任务
if project_name == special_project_name:
    stationRest_dict = {'Time': [7200, 21600, 43200, 46800, 68400, 72000], 'action': [
        'reset', 'reset', 'reset', 'reset', 'reset', 'reset']}
else:
    stationRest_dict = {'Time': [0, 0, 0, 0, 0, 0], 'action': [
        'reset', 'reset', 'reset', 'reset', 'reset', 'reset']}


stationRest = pd.DataFrame(stationRest_dict)
for i in range(1, kubotNum+1):
    kubot_mc_timecopy = kubot_mc_time[kubot_mc_time['KubotNo'] == i]

    StationRestDf = stationRest.copy()
    StationRestDf['KubotNo'] = np.nan

    kubot_mc_timecopy = kubot_mc_timecopy.append(StationRestDf)
    kubot_mc_timecopy.sort_values(by=['Time'], inplace=True)
    if project_name != special_project_name:
        kubot_mc_timecopy['rest_If_Not'] = 'no'
    else:
        kubot_mc_timecopy['rest_If_Not'] = np.where(
            ((kubot_mc_timecopy['Time'] >= 7200) & (kubot_mc_timecopy['Time'] <= 21600)) |
            ((kubot_mc_timecopy['Time'] >= 43200) & (kubot_mc_timecopy['Time'] <= 46800)) |
            ((kubot_mc_timecopy['Time'] >= 68400) & (kubot_mc_timecopy['Time'] <= 72000)), 'yes', 'no')
    kubot_mc_timecopy['action_temp'] = kubot_mc_timecopy['action'].shift()

    kubot_mc_timecopy['process'] = kubot_mc_timecopy['action_temp'] + \
                                   " to " + kubot_mc_timecopy['action']

    kubot_mc_timecopy['time_delta'] = kubot_mc_timecopy.Time - \
                                      kubot_mc_timecopy.Time.shift()
    del kubot_mc_timecopy['action_temp']

    kubot_mc_timecopy['keep_or_not'] = np.where(
        ((kubot_mc_timecopy['action'] == 'working') & (kubot_mc_timecopy['process'] == 'working to reset') & (kubot_mc_timecopy['rest_If_Not'] == 'no')) |
        ((kubot_mc_timecopy['action'] == 'idle') & (kubot_mc_timecopy['process'] == 'reset to idle') & (kubot_mc_timecopy['rest_If_Not'] == 'no')) |
        (kubot_mc_timecopy['process'] == 'working to idle'), 'keep', 'drop')
    kubot_mc_timecopy = kubot_mc_timecopy[kubot_mc_timecopy['keep_or_not'] == 'keep']

    # 每小时每个机器人总工作耗时
    kubot_mc_timecopy['hour'] = kubot_mc_timecopy['Time'].apply(
        lambda x: math.ceil(x/3600))
    kubotWorkingTimeEachHour = pd.pivot_table(kubot_mc_timecopy, index=[
        'hour', 'KubotNo'], values=['time_delta'], aggfunc=np.sum).reset_index()
    KUBOTWORKtIMEEachHour = KUBOTWORKtIMEEachHour.append(
        kubotWorkingTimeEachHour)

    # 每个机器人总工作耗时
    kubotWOrkingTimeTotal = pd.pivot_table(kubot_mc_timecopy, index=['KubotNo'], values=[
        'time_delta'], aggfunc=np.sum).reset_index()
    KUBOTWORKtIME = KUBOTWORKtIME.append(kubotWOrkingTimeTotal)

# 每小时每个机器人总工作耗时
KUBOTWORKtIMEEachHour['工作时长(不含充电)'] = KUBOTWORKtIMEEachHour['time_delta'].apply(
    lambda x: x/3600)
KUBOTWORKtIMEEachHour.rename(
    columns={'hour': '模拟小时', 'KubotNo': '机器人编号'}, inplace=True)
del KUBOTWORKtIMEEachHour['time_delta']


# 各机器人总耗时
KUBOTWORKtIME['工作时长(不含充电)'] = KUBOTWORKtIME['time_delta'].apply(
    lambda x: x/3600)
del KUBOTWORKtIME['time_delta']
# KUBOTWORKtIME.rename(columns = {'KubotNo':'机器人编号'},inplace = True)


# 机器人搬箱
kubotLoadBin = pd.read_csv(
    'output/'+project_name+'/other_detail_data/toLoadBin.csv', names=['Time', 'KubotNo', 'binNo'])
if project_name != special_project_name:
    kubotLoadBin['rest_If_Not'] = 'no'
else:
    kubotLoadBin['rest_If_Not'] = np.where(
        ((kubotLoadBin['Time'] >= 7200) & (kubotLoadBin['Time'] <= 21600)) |
        ((kubotLoadBin['Time'] >= 43200) & (kubotLoadBin['Time'] <= 46800)) |
        ((kubotLoadBin['Time'] >= 68400) & (kubotLoadBin['Time'] <= 72000)), 'yes', 'no')
kubotLoadBin = kubotLoadBin[kubotLoadBin['rest_If_Not'] == 'no']

# 各机器人总搬箱
binCarryEachKubot = pd.pivot_table(kubotLoadBin, index=['KubotNo'], values=[
    'binNo'], aggfunc='count').reset_index()


# 各机器人搬箱及工作耗时
KUBOTWORKtIME = pd.merge(
    KUBOTWORKtIME, binCarryEachKubot, how='left', on=['KubotNo'])
KUBOTWORKtIME.rename(
    columns={'KubotNo': '机器人编号', 'binNo': '搬箱数'}, inplace=True)
KUBOTWORKtIME['单车效率'] = KUBOTWORKtIME['搬箱数']/KUBOTWORKtIME['工作时长(不含充电)']


# 每小时每机器人总搬箱数
kubotLoadBin['hour'] = kubotLoadBin['Time'].apply(lambda x: math.ceil(x/3600))
eachHourBinCarry = pd.pivot_table(kubotLoadBin, index=['hour', 'KubotNo'],
                                  values=['binNo'], aggfunc='count').reset_index()
eachHourBinCarry.rename(
    columns={'hour': '模拟小时', 'KubotNo': '机器人编号', 'binNo': '搬箱数'}, inplace=True)


KUBOTWORKtIMEEachHour = pd.merge(eachHourBinCarry, KUBOTWORKtIMEEachHour, how='left',
                                 on=['模拟小时', '机器人编号'])


KUBOTWORKtIMEEachHour['单车效率'] = KUBOTWORKtIMEEachHour['搬箱数'] / \
                                KUBOTWORKtIMEEachHour['工作时长(不含充电)']

KUBOTWORKtIMEEachHour_Df = pd.pivot_table(KUBOTWORKtIMEEachHour, index=['模拟小时'],
                                          values=['单车效率'],
                                          aggfunc=np.mean).reset_index()
medianKubotBin = KUBOTWORKtIMEEachHour_Df['单车效率'].median()
KUBOTWORKtIMEEachHour_Df['单车效率'] = KUBOTWORKtIMEEachHour_Df['单车效率'].apply(
    lambda x:  medianKubotBin if x >= 40 else x)
KUBOTWORKtIMEEachHour_Df.rename(columns={'单车效率': '平均单车效率'}, inplace=True)


# 每小时平均搬箱效率(不含充电)
KUBOTWORKtIMEEachHour_Df['平均单车效率'] = KUBOTWORKtIMEEachHour_Df['平均单车效率'].apply(
    lambda x: math.ceil(x))

KUBOTWORKtIMEEachHour_Df_Fig = px.bar(KUBOTWORKtIMEEachHour_Df, x=[
    '模拟小时'], y='平均单车效率', width=800, height=500, text='平均单车效率')
KUBOTWORKtIMEEachHour_Df_Fig.update(
    layout=dict(
        xaxis=dict(title="模拟小时", tickangle=-0, showline=True, nticks=30),
        yaxis=dict(title="平均单车效率(不含充电)", showline=True),
        title='每小时平均搬箱效率(不含充电)'
    ))
KUBOTWORKtIMEEachHour_Df_Fig.update_traces(textposition='outside')


# 系统平均每小时每机器人搬箱数(不含充电)
sysBinCarryWithoutCharge = round(KUBOTWORKtIME['单车效率'].mean(), 0)


# 机器人每次接到命令之间的间隔时长
mcTime = pd.read_csv('output/'+project_name+'/other_detail_data/mcTime.csv',
                     names=['Time', 'KubotNo', 'BinID', 'StationNo'])
mcTime.StationNo.fillna(method='bfill', inplace=True)
mcTime_add_groupID = mcTime[mcTime['BinID'].isnull() == True]
mcTime_add_groupID['groupID'] = list(range(1, len(mcTime_add_groupID)+1))


mcTime = pd.merge(mcTime, mcTime_add_groupID, how='left', on=[
    'Time', 'KubotNo', 'BinID', 'StationNo'])
mcTime['groupID'].fillna(method='bfill', inplace=True)
mcTime.dropna(subset=['BinID'], inplace=True)
mcTime.reset_index(drop=True)

# 每一轮的间隔时长
mcTimeEachKubot = mcTime[['Time', 'KubotNo', 'groupID']]
mcTimeEachKubot.drop_duplicates(inplace=True)

# 统计每个机器人跑了多少趟 趟数
kubotRound = pd.pivot_table(mcTimeEachKubot, index=['KubotNo'], values=[
    'groupID'], aggfunc=pd.Series.nunique).reset_index()

# 每趟中间耗时多少
mcTime_df = pd.DataFrame()
for i in range(0, len(mcTimeEachKubot)):
    mcTimeEachKubot_df = mcTimeEachKubot[mcTimeEachKubot['KubotNo'] == i]
    mcTimeEachKubot_df['time_delta'] = mcTimeEachKubot_df['Time'].shift(
        -1) - mcTimeEachKubot_df['Time']
    mcTimeEachKubot_df.dropna(subset=['time_delta'], inplace=True)
    mcTimeEachKubotDf = mcTimeEachKubot_df[['KubotNo', 'time_delta']]
    mcTime_df = mcTime_df.append(mcTimeEachKubotDf)


mcTime_df_fig = px.histogram(mcTime_df, x="time_delta", nbins=1000,
                             width=800, height=500)

mcTime_df_fig.update(
    layout=dict(
        xaxis=dict(title="耗时间隔(s)", tickangle=-0, showline=True, nticks=30),
        yaxis=dict(title="计数", showline=True),
        title='机器人任务耗时-分布图(间隔时间 = 该任务的执行耗时 + 任务之间的间隔耗时)'
    ))


# #机器人总搬箱数
# totalBinCarry =bin_carry_df['BinNum'].sum()
#
# #平均每小时每机器人搬箱数(不含充电)
# bincarryEachKubot = math.ceil(totalBinCarry/(kubotWorkingTimeInTotal/kubotNum))


# 机器人每小时行走趟数列表
KubotExecuteEveryHour = pd.pivot_table(bin_carry_df, index=[
    'hour', 'KubotNo'], values=['Time'], aggfunc='count').reset_index()
KubotExecuteEveryHour.rename(columns={'Time': 'Execute'}, inplace=True)

# 机器人平均每小时行走趟数
avgKubotExecuteEveryHour = round(KubotExecuteEveryHour.Execute.mean(), 0)

# 机器人每小时搬箱数列表
KubotBinCarryEveryHour = pd.pivot_table(bin_carry_df, index=[
    'hour', 'KubotNo'], values=['BinNum'], aggfunc=np.sum).reset_index()


# 机器人平均每小时搬箱数量
avgKubotBinCarryEveryHour = round(KubotBinCarryEveryHour.BinNum.mean(), 0)

# 机器人平均每趟搬箱数
avgBinCarryEachExecutePerKubot = round(
    bin_carry_df.BinNum.sum()/bin_carry_df.Time.count(), 0)

# 机器人每趟搬箱数占比
binCarryPercentage = pd.pivot_table(bin_carry_df, index=['BinNum'], values=[
    'Time'], aggfunc='count').reset_index()
binCarryPercentage.rename(columns={'Time': 'times'}, inplace=True)
binCarryPercentage['Percentage(%)'] = round(
    (binCarryPercentage.times / binCarryPercentage.times.sum())*100, 2)


# 想要知道机器人每个时刻是否有任务
kubotMissionFile = pd.read_csv(
    'output/'+project_name+'/other_detail_data/kubotMissionBegin.csv', names=['Time', 'kubotNo', 'missionNo'])
# 每一个Time去重之后，如果只有一个missionNo，就不是missionNo
kubotMissionFile = kubotMissionFile[['Time', 'missionNo']].drop_duplicates()
kubotMissionFile_check = pd.pivot_table(kubotMissionFile, index=['Time'], values=[
    'missionNo'], aggfunc='count').reset_index()
missionbeginTime = pd.DataFrame(
    kubotMissionFile_check[kubotMissionFile_check['missionNo'] == 2]['Time'])
missionbeginTime = pd.merge(
    missionbeginTime, kubotMissionFile, how='left', on=['Time'])
missionbeginTime = missionbeginTime[missionbeginTime['missionNo']
                                    != 'SolutionManager']
missionbeginTime.rename(columns={'Time': 'beginTime'}, inplace=True)

kubotMissionFinishFile = pd.read_csv(
    'output/'+project_name+'/other_detail_data/kubotMissionFinish.csv', names=['Time', 'kubotNo', 'missionNo'])
kubotMissionFinishFile.rename(columns={'Time': 'finishTime'}, inplace=True)
kubotMissionFinishFile['missionNo'] = kubotMissionFinishFile['missionNo'].apply(
    lambda x: str(x))
kubotMissionDf = pd.merge(
    missionbeginTime, kubotMissionFinishFile, how='left', on=['missionNo'])
kubotMissionDf.dropna(subset=['kubotNo'], how='any', inplace=True)
# kubotMissionDf


print("MC Analysis Completed!")
# 库存分析
# 基础库存宽表准备  invFile_df
inv_location.drop(columns=['nothing-1', 'nothing-2'], inplace=True)
inv_location['locationId'] = inv_location['locationId'] .apply(
    lambda x: int(x))
invFile_df = pd.merge(invFile_df, inv_location, how='left', on=['locationId'])

# 库存中，哪些sku是被订单所使用的
orderSkuList = pd.DataFrame(orderFinishDetailF_df['Sku'].drop_duplicates())
orderSkuList['SkuType'] = 'orderUsed'
orderSkuList.rename(columns={'Sku': 'SKU'}, inplace=True)
invFile_df = pd.merge(invFile_df, orderSkuList, how='left', on=['SKU'])

# 库位数量
totalLocationNum = len(inv_location['locationId'].drop_duplicates())
# 被使用库位的数量
totalInvUsedNum = len(invFile_df['locationId'].drop_duplicates())
# 库位使用率
locationUsedRate = round(totalInvUsedNum/totalLocationNum, 2)
# 存储sku品种数
skuInvCount = pd.Series.nunique(invFile_df.SKU)
# 存储件数
invTotalQty = invFile_df.qty.sum()

# sku深度 ： 平均每个sku存储的件数
avgSkuQty = round(invTotalQty/skuInvCount, 2)

# 每个箱子装载的sku品种数及件数 skuCountEachBinDetail
skuCountEachBinDetail = pd.pivot_table(invFile_df[['locationId', 'SKU', 'qty']], index=['locationId'], values=[
    'SKU', 'qty'], aggfunc={'SKU': pd.Series.nunique, 'qty': np.sum}).reset_index()
skuCountEachBinDetail.rename(
    columns={'SKU': 'skuCouont', 'qty': 'qtySum'}, inplace=True)
skuCountEachBinDetail['avgSkuQty'] = round(
    skuCountEachBinDetail.qtySum/skuCountEachBinDetail.skuCouont, 2)

# 平均每个箱子装载sku品种数及件数
avgSkuCountEachBin = round(skuCountEachBinDetail.skuCouont.mean(), 2)
avgQtyEachBin = math.ceil(skuCountEachBinDetail.qtySum.mean())

# 装载不同数量sku品种的箱子分布及平均存储件数
binDisBySkuCount = pd.pivot_table(skuCountEachBinDetail, index=['skuCouont'], values=['locationId', 'qtySum', 'avgSkuQty'], aggfunc={
    'locationId': 'count', 'qtySum': np.mean, 'avgSkuQty': np.mean}).reset_index()
binDisBySkuCount['qtySum'] = binDisBySkuCount['qtySum'].apply(
    lambda x: round(x, 2))
binDisBySkuCount['avgSkuQty'] = binDisBySkuCount['avgSkuQty'].apply(
    lambda x: round(x, 2))
binDisBySkuCount.rename(columns={'locationId': 'binCount',
                                 'qtySum': 'avgQtyEachBin', 'avgSkuQty': 'avgBinSkuQty'}, inplace=True)
binDisBySkuCount = binDisBySkuCount[[
    'skuCouont', 'binCount', 'avgQtyEachBin', 'avgBinSkuQty']]

# 被订单使用的sku品种数占据所有库存品种数的百分比
usedSkuPercent = round(pd.Series.nunique(
    invFile_df[invFile_df['SkuType'] == 'orderUsed']['SKU'].drop_duplicates())/skuInvCount, 2)


# 被订单使用的sku存储的箱子数
usedSkuBinCount = pd.Series.nunique(
    invFile_df[invFile_df['SkuType'] == 'orderUsed']['locationId'])

# 被订单使用的sku存储的箱子数占据总库存箱子的百分比
usedSkuBinCountPercentage = round(usedSkuBinCount/totalInvUsedNum, 2)

# 被订单使用的箱子每个箱子装载sku品种数和件数明细 usedBinSkuCountAndQty
usedBinSkuCountAndQty = pd.pivot_table(invFile_df[invFile_df['SkuType'] == 'orderUsed'][['locationId', 'SKU', 'qty']], index=['locationId'],
                                       values=['SKU', 'qty'], aggfunc={'SKU': pd.Series.nunique, 'qty': np.sum}).reset_index()
usedBinSkuCountAndQty.rename(
    columns={'SKU': 'skuCount', 'qty': 'qtySum'}, inplace=True)

# 被订单使用的箱子平均每个箱子装载sku品种数和件数
avgUsedBinSkuCount = round(usedBinSkuCountAndQty.skuCount.mean(), 1)
avgUsedBinQty = math.ceil(usedBinSkuCountAndQty.qtySum.mean())

# 库存中混箱个数的占比
mixBinCountPercentage = round(
    binDisBySkuCount[binDisBySkuCount['skuCouont'] == 2].binCount / totalInvUsedNum, 2).mean()


# 混箱存储件数占比
mixBinQtyPercentage = round((binDisBySkuCount[binDisBySkuCount['skuCouont'] == 2]['avgQtyEachBin']
                             * binDisBySkuCount[binDisBySkuCount['skuCouont'] == 2]['binCount'])/invTotalQty, 2)
mixBinQtyPercentage = mixBinQtyPercentage.mean()

# 区分出冷热sku
# 根据sku的出库件数降序，累加每一个sku出库件数占比总出库件数的百分比，当sku累加件数占比占据总出库件数60%，则任务是hot sku
SkuOutputQty = pd.pivot_table(orderFinishDetailF_df[['Sku', 'Qty']], index=[
    'Sku'], values=['Qty'], aggfunc=np.sum).reset_index()
SkuOutputQty.sort_values(by=['Qty'], ascending=False, inplace=True)
SkuOutputQty['Qty_cumsum_percentage'] = (
                                            SkuOutputQty['Qty'].cumsum())/SkuOutputQty.Qty.sum()
SkuOutputQty['sku_count_percentage'] = range(1, len(SkuOutputQty)+1)
SkuOutputQty['sku_count_percentage'] = SkuOutputQty['sku_count_percentage'].apply(
    lambda x: x/len(SkuOutputQty))
SkuOutputQty['hot_sku_or_not'] = np.where(
    SkuOutputQty['Qty_cumsum_percentage'] <= 0.6, 'hot-sku', 'cool-sku')

# 热sku概况
hotSkuCount = math.ceil(pd.Series.nunique(
    SkuOutputQty[SkuOutputQty['hot_sku_or_not'] == 'hot-sku']['Sku']))
hotSkuQty = math.ceil(
    SkuOutputQty[SkuOutputQty['hot_sku_or_not'] == 'hot-sku']['Qty'].sum())


# 热sku出库件数占总出库件数百分比
hotSKuQtyPercentage = round(hotSkuQty/orderFinishDetailF_df.Qty.sum(), 2)

# 热sku占所有sku品种数占比
hotSkuCountPercentage = round(
    hotSkuCount/pd.Series.nunique(SkuOutputQty['Sku']), 2)

SkuOutputQty.rename(columns={'Sku': 'SKU'}, inplace=True)

invFile_df = pd.merge(invFile_df, SkuOutputQty, how='left', on='SKU')
invFile_df.hot_sku_or_not.fillna(value='cool-sku', inplace=True)
invFile_df.drop(columns=['Qty_cumsum_percentage',
                         'sku_count_percentage'], inplace=True)

print("Inv Analysis Completed!")


# Delay Analysis
delayDetailFile = pd.read_csv('output/'+project_name+'/other_detail_data/delay_detail.csv',
                              names=['Time', 'StationId', 'OrderId', 'Sku', 'BinId', 'Qty', 'DelayTime'])

if len(delayDetailFile) != 0:

    # 订单重组信息
    reconbine_detail = pd.read_csv(
        'output/'+project_name+'/other_detail_data/重组优化映射.csv', names=['OrderId', 'Orig_OrderId'])

    # 获得原始订单的波次
    orderRound = pd.read_csv(
        'output/'+project_name+'/other_detail_data/orderRound.csv', names=['Orig_OrderId', 'Round'])

    # 获取有挂单现象的重组订单编号
    delay_recombine_orderNo = list(set(delayDetailFile.OrderId.to_list()))

    # 获取有挂单现象的子订单号
    reconbine_detail = reconbine_detail[reconbine_detail.OrderId.isin(
        delay_recombine_orderNo)]
    reconbine_detail.dropna(how='any', inplace=True)

    # 匹配波次信息
    delay_df = pd.merge(reconbine_detail, orderRound,
                        how='left', on=['Orig_OrderId'])

    # 匹配截单时间
    orderDeadline = pd.read_csv('output/'+project_name+'/other_detail_data/orderDeadline.csv', names=[
        'Orig_OrderId', 'Deadline'])

    delay_df = pd.merge(delay_df, orderDeadline, how='left', on='Orig_OrderId')

    # 匹配组合单最大完成耗时
    order_begin_and_finish = pd.read_csv('output/'+project_name+'/other_detail_data/order_finish_detail.csv', names=[
        'Time', 'StationNo', 'OrderId', 'Sku', 'BinId', 'Qty'])

    orderMaxTime = pd.DataFrame(
        order_begin_and_finish.groupby('OrderId').Time.max())
    orderMaxTime.reset_index(inplace=True)

    delay_df = pd.merge(delay_df, orderMaxTime, how='left', on='OrderId')

    delay_df['deadlineTime'] = delay_df.Time - delay_df.Deadline

    # 匹配件数
    orderqty = pd.read_csv(
        'output/'+project_name+'/simulation_order.csv', usecols=['order_num', 'qty'])
    orderqty.order_num = orderqty.order_num.apply(lambda x: x.split('C')[1])
    orderqty_df = pd.DataFrame(orderqty.groupby('order_num').qty.sum())
    orderqty_df.reset_index(inplace=True)
    orderqty_df.rename(columns={'order_num': 'Orig_OrderId'}, inplace=True)
    orderqty_df.Orig_OrderId = orderqty_df.Orig_OrderId.astype('float64')

    delay_df = pd.merge(delay_df, orderqty_df, how='left', on='Orig_OrderId')

    # 过滤子订单中不挂单的明细
    delay_df = delay_df[delay_df.deadlineTime > 0]

    # 统计挂单总件数
    totalDelayQty = delay_df.qty.sum()

    # 统计平均单笔订单挂单耗时(s)
    avgDelayEachOrder = delay_df.deadlineTime.mean()

    # 统计最长挂单耗时(s)
    maxDelayTime = delay_df.deadlineTime.max()

    # 统计最短挂单耗时(s)
    minDelayTime = delay_df.deadlineTime.min()

    # 统计每个波次的挂单总件数和平均挂单耗时
    bociDelay = delay_df.groupby('Round').agg(
        {'qty': np.sum, 'deadlineTime': np.mean})

    # print(totalDelayQty)
    # print(avgDelayEachOrder)
    # print(maxDelayTime)
    # print(minDelayTime)
    # print(bociDelay)
    delayOverview_T = ff.create_table(bociDelay)
    delayOverview_T.update_layout(
        autosize=False,
        title_text='每个波次的挂单总件数和平均挂单耗时',
        margin={'t': 35, 'l': 30, 'r': 30, 'b': 30},
        title_font_family="Times New Roman",
        title_font_color="blue")

else:
    totalDelayQty = 0
    avgDelayEachOrder = 0
    maxDelayTime = 0
    minDelayTime = 0
    bociDelay = pd.DataFrame()


# 下发到系统和下发到槽位的时间间隔(s)-分布图
dispatchToSysAndSlot_bar = station_fill[['Time', '下发到系统和下发到槽位的时间间隔(s)']]
dispatchToSysAndSlot_bar['下发到系统和下发到槽位的时间间隔(s)'].fillna(0, inplace=True)
dispatchToSysAndSlot_bar['下发到系统和下发到槽位的时间间隔(s)'] = dispatchToSysAndSlot_bar['下发到系统和下发到槽位的时间间隔(s)'].apply(
    lambda x: math.ceil(x))
dispatchToSysAndSlot_bar['计数'] = dispatchToSysAndSlot_bar['下发到系统和下发到槽位的时间间隔(s)'].copy(
)

dispatchToSysAndSlot_bar_fig = pd.pivot_table(dispatchToSysAndSlot_bar, index=['下发到系统和下发到槽位的时间间隔(s)'],
                                              values=['计数'],
                                              aggfunc='count').reset_index()


station_order_process.to_csv(
    'output/'+project_name+'/order_graph/1-每个工作站出库的订单量订单行数出库件数.csv', index=False)
station_fill['下发到系统和下发到槽位的时间间隔(s)'].describe().reset_index().to_csv(
    'output/'+project_name+'/order_graph/2-下发到系统和下发到槽位的时间间隔(s).csv', index=False)
OrderDispatchProcess['下发到槽位至开始配订单行的时间间隔(s)'].describe().reset_index().to_csv(
    'output/'+project_name+'/order_graph/3-平均每个订单下发到槽位至开始配订单行的时间间隔.csv', index=False)
OrderDispatchFinished['平均每个订单下发到槽位至订单完成的耗时(min)'].describe().reset_index().to_csv(
    'output/'+project_name+'/order_graph/4-平均每个订单下发到槽位至订单完成的耗时(min).csv', index=False)
bin_process['BinProcessTime'].describe().reset_index().to_csv(
    'output/'+project_name+'/order_graph/5-平均每个箱子的处理耗时(s).csv', index=False)
dispatchToSlot_bar.to_csv('output/'+project_name +
                          '/order_graph/6-下发到槽位的件数-小时序图.csv', index=False)
dispatchToSlotEachStation_bar.to_csv(
    'output/'+project_name+'/order_graph/7-下发到各操作台槽位的件数-时序图.csv', index=False)
dispatchOrderCountToSlotEachStation_bar.to_csv(
    'output/'+project_name+'/order_graph/8-下发到各操作台槽位的订单数-时序图.csv', index=False)
dispatchSkuCountToSlot_bar.to_csv(
    'output/'+project_name+'/order_graph/9-下发到槽位的sku品种数-时序图.csv', index=False)
dispatchSkuCountToSlotEachStation_bar.to_csv(
    'output/'+project_name+'/order_graph/10-下发到各操作台槽位的sku品种数-时序图.csv', index=False)
StationBin_RepeatRate.to_csv(
    'output/'+project_name+'/order_graph/11-单操作台箱子重取率.csv', index=False)
emptyBin_df_hourCount.to_csv(
    'output/'+project_name+'/order_graph/12-空箱取出的小时曲线.csv', index=False)
BinRepeatEachStation_overOneStation_df_distribu.to_csv(
    'output/'+project_name+'/order_graph/13-多个操作台重取分布.csv', index=False)
BinRepeatEachStation_df_repeat_Distribu.to_csv(
    'output/'+project_name+'/order_graph/14-同操作台重取分布.csv', index=False)
EmptyStationTable.to_csv('output/'+project_name +
                         '/order_graph/15-各操作台接受箱子数及空箱取出数量.csv', index=False)
StationTotalBinCarry.to_csv(
    'output/'+project_name+'/order_graph/16-每个操作台的搬箱数.csv', index=False)
StationBinAcceptEachHour_df.to_csv(
    'output/'+project_name+'/order_graph/17-每个操作台每小时接受的箱子个数.csv', index=False)
orderBeginFinishFileSingleOrderTime_bar.to_csv(
    'output/'+project_name+'/order_graph/18-平均每小时每个订单完成耗时(min)分布.csv', index=False)
KubotExecuteEveryHour.to_csv(
    'output/'+project_name+'/order_graph/19-机器人每小时行走趟数列表.csv', index=False)
KubotBinCarryEveryHour.to_csv(
    'output/'+project_name+'/order_graph/20-机器人每小时搬箱数列表.csv', index=False)
binCarryPercentage.to_csv('output/'+project_name +
                          '/order_graph/21-机器人每趟搬箱数占比.csv', index=False)
binProcessTimeEachStation.to_csv(
    'output/'+project_name+'/order_graph/22-操作台空闲率总表.csv', index=False)
inboundBinEachHour.to_csv('output/'+project_name +
                          '/order_graph/23-入库的搬箱数曲线.csv', index=False)
outboundBinEachHour.to_csv('output/'+project_name +
                           '/order_graph/24-出库的搬箱数曲线.csv', index=False)
sys_bin_hour_df_bar.to_csv('output/'+project_name +
                           '/order_graph/25-系统每小时搬箱数.csv', index=False)
order_df_IQ_order_line.to_csv(
    'output/'+project_name+'/order_graph/26-sku出库订单行数分析.csv', index=False)
order_df_IQ.to_csv('output/'+project_name +
                   '/order_graph/26-sku出库件数分析.csv', index=False)
order_EQ_order_line.to_csv('output/'+project_name +
                           '/order_graph/27-不同出库订单行范围内的订单数分布.csv', index=False)
order_EQ_df.to_csv('output/'+project_name +
                   '/order_graph/28-不同出库件数范围内的订单数分布.csv', index=False)
order_dispatch.to_csv('output/'+project_name +
                      '/order_graph/29-下发到系统的订单数-时序图.csv', index=False)
order_df_dispath_qtyTo_Sys.to_csv(
    'output/'+project_name+'/order_graph/30-下发到系统的件数数-时序图.csv', index=False)


# 订单基础信息
basic_order_list = ['订单总件数',
                    '订单量',
                    '订单行',
                    'sku品种数',
                    '件单比',
                    '件行比',
                    '单件订单数',
                    '多件订单数',
                    '多件件单比',
                    '单笔最大出库件数',
                    '单笔多件订单最多包含sku品种数',
                    '多件订单平均每笔包含sku品种数']

basic_order_table = pd.DataFrame(basic_order_list, columns=['订单基础信息'])
basic_order_table['统计值'] = basic_order_table['订单基础信息'].apply(lambda x:
                                                             order_qty_sum if x == '订单总件数'
                                                             else(order_num_count_distinct if x == '订单量'
                                                                  else(order_line if x == '订单行'
                                                                       else(sku_num_count_distinct if x == 'sku品种数'
                                                                            else(avg_outbound_qty_per_order if x == '件单比'
                                                                                 else(avg_outbound_qty_per_order_line if x == '件行比'
                                                                                      else(order_num_single if x == '单件订单数'
                                                                                           else(order_num_multi if x == '多件订单数'
                                                                                                else(avg_outbound_qty_per_order_multi if x == '多件件单比'
                                                                                                     else(order_df_max_qty if x == '单笔最大出库件数'
                                                                                                          else(order_num_multi_maximun_sku_num if x == '单笔多件订单最多包含sku品种数'
                                                                                                               else(order_num_multi_avg_sku_num if x == '多件订单平均每笔包含sku品种数'
                                                                                                                    else 'ERROR'))))))))))))


basic_order_table.to_csv('output/'+project_name+'/订单基础信息.csv', index=False)


# OP基础信息
basic_op_list = ['平均下发到系统和下发到槽位的时间间隔(s)',
                 '下发到槽位至开始配订单行的时间间隔(s)',
                 '平均每个订单下发到槽位至订单完成的耗时(min)',
                 '平均每个箱子的处理耗时(s)',
                 '平均处理1个订单行耗时(s)',
                 '平均拣1件要多少时间(s)',
                 '整体平均每个订单完成的耗时(min)',
                 '平均单操作台单个sku每次命中订单数',
                 '平均操作台每小时下发订单数',
                 '平均每个操作台每小时出库件数',
                 '平均每个操作台每小时下发sku品种数',
                 '平均每个操作台空闲率'
                 ]

basic_op_table = pd.DataFrame(basic_op_list, columns=['OP基础信息'])
basic_op_table['统计值'] = basic_op_table['OP基础信息'].apply(lambda x:
                                                       dispatchToSysAndSlot if x == '平均下发到系统和下发到槽位的时间间隔(s)'
                                                       else(DispathToSlotAndStartProcess if x == '下发到槽位至开始配订单行的时间间隔(s)'
                                                            else(OrderStartAndOrderFinish if x == '平均每个订单下发到槽位至订单完成的耗时(min)'
                                                                 else(bin_processTime if x == '平均每个箱子的处理耗时(s)'
                                                                      else(QtyAndOrderLineProcessTime if x == '平均处理1个订单行耗时(s)'
                                                                           else(QtyAndOrderLineProcessQtyTime if x == '平均拣1件要多少时间(s)'
                                                                                else(SkuHitOrderLineEachStationEachTime_mean if x == '平均单操作台单个sku每次命中订单数'
                                                                                     else(sysOrderProcessTotalTime if x == '整体平均每个订单完成的耗时(min)'
                                                                                          else(avgEachStationProcessedOrderCount if x == '平均操作台每小时下发订单数'
                                                                                               else(avgEachStationProcessedsumQty if x == '平均每个操作台每小时出库件数'
                                                                                                    else(avgEachStationProcessedSkuCouont if x == '平均每个操作台每小时下发sku品种数'
                                                                                                         else(avgIdelRatePerStation if x == '平均每个操作台空闲率'
                                                                                                              else 'ERROR'))))))))))))


basic_op_table.to_csv('output/'+project_name+'/OP基础信息.csv', index=False)


# MC基础信息
basic_mc_list = ['平均每个工作站单个料箱命中sku品种数',
                 '平均每个料箱命中工作站数量',
                 '平均每个操作台接收箱子个数',
                 '平均单操作台箱子重取率(同操作台重取)',
                 '空箱个数',
                 '整体跨操作台重取率',
                 '平均每小时每操作台接受箱子个数',
                 '机器人平均每小时行走趟数',
                 '机器人平均每小时搬箱数量',
                 '机器人平均每趟搬箱数'
                 ]

basic_mc_table = pd.DataFrame(basic_mc_list, columns=['MC基础信息'])
basic_mc_table['统计值'] = basic_mc_table['MC基础信息'].apply(lambda x:
                                                       BinPickSkuEachStationEachTime_num if x == '平均每个工作站单个料箱命中sku品种数'
                                                       else(avgBinHitStaton if x == '平均每个料箱命中工作站数量'
                                                            else(StationBinPicked if x == '平均每个操作台接收箱子个数'
                                                                 else(StationBin_RepeatRateAvg if x == '平均单操作台箱子重取率(同操作台重取)'
                                                                      else(emptyBinCount if x == '空箱个数'
                                                                           else(OverStation_BinCarried_Rate if x == '整体跨操作台重取率'
                                                                                else(StationBinAcceptEachHourAvg if x == '平均每小时每操作台接受箱子个数'
                                                                                     else(avgKubotExecuteEveryHour if x == '机器人平均每小时行走趟数'
                                                                                          else(avgKubotBinCarryEveryHour if x == '机器人平均每小时搬箱数量'
                                                                                               else(avgBinCarryEachExecutePerKubot if x == '机器人平均每趟搬箱数'
                                                                                                    else 'ERROR'))))))))))


basic_mc_table.to_csv('output/'+project_name+'/MC基础信息.csv', index=False)


# 库存基础信息
basic_inv_list = ['库位数量',
                  '被使用库位的数量',
                  '库位使用率',
                  '存储sku品种数',
                  '存储件数',
                  '平均每个sku存储的件数',
                  '被订单使用的sku品种数占据所有库存品种数的百分比',
                  '被订单使用的sku存储的箱子数',
                  '被订单使用的sku存储的箱子数占据总库存箱子的百分比',
                  '被订单使用的箱子平均每个箱子装载sku品种数',
                  '被订单使用的箱子平均每个箱子装载件数',
                  '库存中混箱个数的占比',
                  '混箱存储件数占比',
                  '热sku品种数',
                  '热sku库存件数',
                  '热sku出库件数占总出库件数百分比',
                  '热sku占所有sku品种数占比'
                  ]

basic_inv_table = pd.DataFrame(basic_inv_list, columns=['库存基础信息'])
basic_inv_table['统计值'] = basic_inv_table['库存基础信息'].apply(lambda x:
                                                         totalLocationNum if x == '库位数量'
                                                         else(totalInvUsedNum if x == '被使用库位的数量'
                                                              else(locationUsedRate if x == '库位使用率'
                                                                   else(skuInvCount if x == '存储sku品种数'
                                                                        else(invTotalQty if x == '存储件数'
                                                                             else(avgSkuQty if x == '平均每个sku存储的件数'
                                                                                  else(usedSkuPercent if x == '被订单使用的sku品种数占据所有库存品种数的百分比'
                                                                                       else(usedSkuBinCount if x == '被订单使用的sku存储的箱子数'
                                                                                            else(usedSkuBinCountPercentage if x == '被订单使用的sku存储的箱子数占据总库存箱子的百分比'
                                                                                                 else(avgUsedBinSkuCount if x == '被订单使用的箱子平均每个箱子装载sku品种数'
                                                                                                      else(avgUsedBinQty if x == '被订单使用的箱子平均每个箱子装载件数'
                                                                                                           else(mixBinCountPercentage if x == '库存中混箱个数的占比'
                                                                                                                else(mixBinQtyPercentage if x == '混箱存储件数占比'
                                                                                                                     else(hotSkuCount if x == '热sku品种数'
                                                                                                                          else(hotSkuQty if x == '热sku库存件数'
                                                                                                                               else(hotSKuQtyPercentage if x == '热sku出库件数占总出库件数百分比'
                                                                                                                                    else(hotSkuCountPercentage if x == '热sku占所有sku品种数占比'
                                                                                                                                         else 'ERROR')))))))))))))))))

basic_inv_table.to_csv('output/'+project_name+'/库存基础信息.csv', index=False)


# 可视化
print("Begin to draw graph")
tableBasicOverview = pd.read_csv('output/'+project_name+'/基础数据.csv')
add_row = {'模拟基础数据': '系统平均每小时每机器人搬箱数(不含充电)', '统计值': sysBinCarryWithoutCharge}
add_delayqty = {'模拟基础数据': '挂单总件数', '统计值': totalDelayQty}
add_avgDelayEachOrder = {'模拟基础数据': '平均单笔订单挂单耗时(s)', '统计值': avgDelayEachOrder}
add_maxDelayTime = {'模拟基础数据': '最长挂单耗时(s)', '统计值': maxDelayTime}
add_minDelayTime = {'模拟基础数据': '最短挂单耗时(s)', '统计值': minDelayTime}
tableBasicOverview = tableBasicOverview.append(add_row, ignore_index=True)
tableBasicOverview = tableBasicOverview.append(add_delayqty, ignore_index=True)
tableBasicOverview = tableBasicOverview.append(
    add_avgDelayEachOrder, ignore_index=True)
tableBasicOverview = tableBasicOverview.append(
    add_maxDelayTime, ignore_index=True)
tableBasicOverview = tableBasicOverview.append(
    add_minDelayTime, ignore_index=True)

tableOrder = pd.read_csv('output/'+project_name+'/订单基础信息.csv')
tableInv = pd.read_csv('output/'+project_name+'/库存基础信息.csv')
tableOP = pd.read_csv('output/'+project_name+'/OP基础信息.csv')
tableMC = pd.read_csv('output/'+project_name+'/MC基础信息.csv')
# tablePP = pd.read_csv('output/'+project_name+'/PP基础信息.csv')


def convertIntoTable(df):
    table = ff.create_table(df)
    table.update_layout(
        autosize=False,
        width=700,
        #         title_text = '基础数据',
        margin={'t': 35, 'l': 30, 'r': 30, 'b': 30},
        title_font_family="Times New Roman",
        title_font_color="blue")
    return table


tableBasicOverview_T = convertIntoTable(tableBasicOverview)
tableOrder_T = convertIntoTable(tableOrder)
tableInv_T = convertIntoTable(tableInv)
tableOP_T = convertIntoTable(tableOP)
tableMC_T = convertIntoTable(tableMC)
# tablePP_T = convertIntoTable(tablePP)


# 一级标题
def add_firstTitle(firstTitle):
    title = pd.DataFrame()
    test = ff.create_table(title, height_constant=10)
    test.update_layout(margin=dict(l=30, r=0, t=1500, b=30),
                       paper_bgcolor="white",
                       title_font_family="Times New Roman",
                       title_font_color="blue")
    test.add_annotation(dict(font=dict(color='blue', size=50),
                             x=0,
                             y=0,
                             showarrow=False,
                             text=firstTitle,
                             textangle=0,
                             xanchor='left',
                             xref="paper",
                             yref="paper"))
    return test

# 二级标题


def add_secTitle(secTitle):
    title = pd.DataFrame()
    test_2 = ff.create_table(title, height_constant=10)
    test_2.update_layout(margin=dict(l=30, r=0, t=80, b=30), paper_bgcolor="white",
                         title_font_family="Times New Roman",
                         title_font_color="blue")
    test_2.add_annotation(dict(font=dict(color='blue', size=30),
                               x=0,
                               y=0,
                               showarrow=False,
                               text=secTitle,
                               textangle=0,
                               xanchor='left',
                               xref="paper",
                               yref="paper"))
    return test_2

# 三级标题


def add_thirTitle(thirTitle):
    title = pd.DataFrame()
    test_3 = ff.create_table(title, height_constant=10)
    test_3.update_layout(margin=dict(l=30, r=0, t=80, b=30), paper_bgcolor="white",
                         title_font_family="Times New Roman",
                         title_font_color="blue")
    test_3.add_annotation(dict(font=dict(color='blue', size=20),
                               x=0,
                               y=0,
                               showarrow=False,
                               text=thirTitle,
                               textangle=0,
                               xanchor='left',
                               xref="paper",
                               yref="paper"))
    return test_3


simulationDataAnalisysReport = add_firstTitle(
    "数据分析模拟报告 " + time.strftime("%Y/%m/%d"))
UpsysOrderDispatch = add_secTitle("上游下发订单数据概览")
opAnalysis = add_secTitle("订单分配数据概览")
mcAnalysis = add_secTitle("任务管理数据概览")
op_dispatch = add_thirTitle("订单分配-订单下发概览")
op_finish = add_thirTitle("订单分配-订单完成概览")
mc_kubotBinCarry = add_thirTitle("任务管理-机器人搬箱")
mc_kubotBinProcess = add_thirTitle("任务管理-操作台箱子处理")
orderDelay = add_secTitle("挂单数据概览")

# 30-下发到系统的件数数-时序图.csv
UpSysDispatchOrderQty = pd.read_csv(
    'output/'+project_name+'/order_graph/30-下发到系统的件数数-时序图.csv')
UpSysDispatchOrderQty_fig = px.bar(
    UpSysDispatchOrderQty, x='dispatch_hour', y='qty', width=800, height=500, text='qty')
UpSysDispatchOrderQty_fig.update(
    layout=dict(
        xaxis=dict(title="模拟小时", tickangle=-0, showline=True, nticks=30),
        yaxis=dict(title="下发件数", showline=True),
        title='下发到系统的件数-时序图'
    ))
UpSysDispatchOrderQty_fig.update_traces(textposition='outside')


# 29-下发到系统的订单数-时序图
UpSysDispatchOrderCount = pd.read_csv(
    'output/'+project_name+'/order_graph/29-下发到系统的订单数-时序图.csv')
UpSysDispatchOrderCount_fig = px.bar(UpSysDispatchOrderCount, x='dispatch_hour', y='order_num',
                                     width=800, height=500, text='order_num')
UpSysDispatchOrderCount_fig.update(
    layout=dict(
        xaxis=dict(title="模拟小时", tickangle=-0, showline=True, nticks=30),
        yaxis=dict(title="下发订单数", showline=True),
        title='下发到系统的订单数-时序图'
    ))

UpSysDispatchOrderCount_fig.update_traces(textposition='outside')


# 6-下发到槽位的件数-小时序图.csv
OpDispatchOrderQty = pd.read_csv(
    'output/'+project_name+'/order_graph/6-下发到槽位的件数-小时序图.csv')
OpDispatchOrderQty_fig = px.bar(OpDispatchOrderQty, x='hour', y='Qty',
                                width=800, height=500, text='Qty')
OpDispatchOrderQty_fig.update(
    layout=dict(
        xaxis=dict(title="模拟小时", tickangle=-0, showline=True, nticks=30),
        yaxis=dict(title="下发件数", showline=True),
        title='下发到槽位的件数-时序图'
    ))
OpDispatchOrderQty_fig.update_traces(textposition='outside')


# 9-下发到槽位的sku品种数-时序图.csv
OpDispatchOrderSkuCount = pd.read_csv(
    'output/'+project_name+'/order_graph/9-下发到槽位的sku品种数-时序图.csv')
OpDispatchOrderSkuCount_fig = px.bar(
    OpDispatchOrderSkuCount, x='hour', y='Sku', text='Sku', width=800, height=500)
OpDispatchOrderSkuCount_fig.update(
    layout=dict(
        xaxis=dict(title="模拟小时", tickangle=-0, showline=True, nticks=30),
        yaxis=dict(title="下发SKU数", showline=True),
        title='下发到槽位的sku品种数-时序图',
        uniformtext_minsize=8,
        uniformtext_mode='hide'
    ))

OpDispatchOrderSkuCount_fig.update_traces(
    texttemplate='%{text:.2s}', textposition='outside')


# 7-下发到各操作台槽位的件数-时序图.csv
OpDispatchOrderQtyEachStation = pd.read_csv(
    'output/'+project_name+'/order_graph/7-下发到各操作台槽位的件数-时序图.csv')
OpDispatchOrderQtyEachStation.StationNo = OpDispatchOrderQtyEachStation.StationNo.apply(
    lambda x: str(x))
OpDispatchOrderQtyEachStation_fig = px.bar(OpDispatchOrderQtyEachStation, x='hour', y='Qty', color='StationNo',
                                           width=800, height=500)
OpDispatchOrderQtyEachStation_fig.update(
    layout=dict(
        xaxis=dict(title="模拟小时", tickangle=-0, showline=True, nticks=30),
        yaxis=dict(title="下发件数", showline=True),
        title='下发到各操作台槽位的件数-时序图'
    ))


# 8-下发到各操作台槽位的订单数-时序图.csv
OpDispatchOrderCountEachStation = pd.read_csv(
    'output/'+project_name+'/order_graph/8-下发到各操作台槽位的订单数-时序图.csv')
OpDispatchOrderCountEachStation.StationNo = OpDispatchOrderCountEachStation.StationNo.apply(
    lambda x: str(x))
OpDispatchOrderCountEachStation_fig = px.bar(
    OpDispatchOrderCountEachStation, x='hour', y='OrderID', color='StationNo', width=800, height=500)
OpDispatchOrderCountEachStation_fig.update(
    layout=dict(
        xaxis=dict(title="模拟小时", tickangle=-0, showline=True, nticks=30),
        yaxis=dict(title="下发订单数", showline=True),
        title='下发到各操作台槽位的订单数-时序图'
    ))


# 10-下发到各操作台槽位的sku品种数-时序图.csv
OpDispatchSkuCountEachStation = pd.read_csv(
    'output/'+project_name+'/order_graph/10-下发到各操作台槽位的sku品种数-时序图.csv')
OpDispatchSkuCountEachStation.StationNo = OpDispatchSkuCountEachStation.StationNo.apply(
    lambda x: str(x))
OpDispatchSkuCountEachStation_fig = px.bar(OpDispatchSkuCountEachStation, x='hour', y='Sku', color='StationNo',
                                           width=800, height=500)
OpDispatchSkuCountEachStation_fig.update(
    layout=dict(
        xaxis=dict(title="模拟小时", tickangle=-0, showline=True, nticks=30),
        yaxis=dict(title="下发SKU数", showline=True),
        title='下发到各操作台槽位的sku品种数-时序图'
    ))


# 18-平均每小时每个订单完成耗时(min)分布.csv
OpFinishAvgOrderProcessedTime = pd.read_csv(
    'output/'+project_name+'/order_graph/18-平均每小时每个订单完成耗时(min)分布.csv')
OpFinishAvgOrderProcessedTime_fig = px.bar(
    OpFinishAvgOrderProcessedTime, x='hour', y='TimeDiff(min)', text='TimeDiff(min)', width=800, height=500)
OpFinishAvgOrderProcessedTime_fig.update(
    layout=dict(
        xaxis=dict(title="模拟小时", tickangle=-0, showline=True, nticks=30),
        yaxis=dict(title="完成耗时(min)", showline=True),
        title='平均每小时每个订单完成耗时(min)分布',
        uniformtext_minsize=8,
        uniformtext_mode='hide'
    ))
OpFinishAvgOrderProcessedTime_fig.update_traces(
    texttemplate='%{text:.2s}', textposition='outside')


# 1-每个工作站出库的订单量订单行数出库件数.csv
OpFinishEachHourOverview = pd.read_csv(
    'output/'+project_name+'/order_graph/1-每个工作站出库的订单量订单行数出库件数.csv')
OpFinishEachHourOverview.rename(columns={
    'StationNo': '操作台编号',
    'OrderCount': '操作台处理订单数',
    'OrderLineCount': '操作台处理订单行数',
    'QtySum': '操作台出库件数'
}, inplace=True)
OpFinishEachHourOverview_T = ff.create_table(OpFinishEachHourOverview)
OpFinishEachHourOverview_T.update_layout(
    autosize=False,
    title_text='订单分配-各操作台分配概览',
    margin={'t': 35, 'l': 30, 'r': 30, 'b': 30},
    title_font_family="Times New Roman",
    title_font_color="blue")


# 24-出库的搬箱数曲线.csv
McFinishOutboundBinCount = pd.read_csv(
    'output/'+project_name+'/order_graph/24-出库的搬箱数曲线.csv')
McFinishOutboundBinCount_fig = px.bar(McFinishOutboundBinCount, x='hour', y='outboundBinNum', text='outboundBinNum',
                                      width=800, height=500)
McFinishOutboundBinCount_fig.update(
    layout=dict(
        xaxis=dict(title="模拟小时", tickangle=-0, showline=True, nticks=30),
        yaxis=dict(title="出库箱子数", showline=True),
        title='出库的搬箱数曲线',
        uniformtext_minsize=1,
        uniformtext_mode='hide'
    ))
McFinishOutboundBinCount_fig.update_traces(
    texttemplate='%{text:.2s}', textposition='outside')


# 20-机器人每小时搬箱数列表.csv
mcKubotBinCarryEachHour = pd.read_csv(
    'output/'+project_name+'/order_graph/20-机器人每小时搬箱数列表.csv')

mcBinCarryEachHour = pd.pivot_table(mcKubotBinCarryEachHour, index=[
    'hour'], values=['BinNum'], aggfunc=np.mean).reset_index()
mcBinCarryEachHour['BinNum'] = mcBinCarryEachHour['BinNum'].apply(
    lambda x: math.ceil(x))
# mcKubotBinCarryEachHour.KubotNo = mcKubotBinCarryEachHour.KubotNo.apply(lambda x : str(x))
mcKubotBinCarryEachHour_fig = px.bar(mcBinCarryEachHour, x='hour', y='BinNum', text='BinNum',
                                     width=800, height=500)
mcKubotBinCarryEachHour_fig.update(
    layout=dict(
        xaxis=dict(title="模拟小时", tickangle=-0, showline=True, nticks=30),
        yaxis=dict(title="搬箱数", showline=True),
        title='机器人每小时搬箱数曲线'
    ))

mcKubotBinCarryEachHour_fig.update_traces(
    texttemplate='%{text:.2s}', textposition='outside')


# 21-机器人每趟搬箱数占比.csv
mcKubotBinCarryEachRound = pd.read_csv(
    'output/'+project_name+'/order_graph/21-机器人每趟搬箱数占比.csv')
# mcKubotBinCarryEachHour.KubotNo = mcKubotBinCarryEachHour.KubotNo.apply(lambda x : str(x))
mcKubotBinCarryEachRound_fig = px.pie(mcKubotBinCarryEachRound, values='times', names='BinNum',
                                      width=800, height=500)
mcKubotBinCarryEachRound_fig.update(
    layout=dict(
        xaxis=dict(title="机器人单趟搬运箱子数", tickangle=-0, showline=True, nticks=30),
        yaxis=dict(title="搬箱数", showline=True),
        title='机器人每趟搬箱数占比'
    ))


# 16-每个操作台的搬箱数.csv
mcStationAcceptBinNum = pd.read_csv(
    'output/'+project_name+'/order_graph/16-每个操作台的搬箱数.csv')
mcStationAcceptBinNum.StationNo = mcStationAcceptBinNum.StationNo.apply(
    lambda x: str(x))
mcStationAcceptBinNum_fig = px.bar(mcStationAcceptBinNum, x='StationNo', y='TotalBinCount',
                                   width=800, height=500, text='TotalBinCount')
mcStationAcceptBinNum_fig.update(
    layout=dict(
        xaxis=dict(title="操作台编号", tickangle=-0, showline=True, nticks=20),
        yaxis=dict(title="操作台接收箱子数", showline=True),
        title='每小时每个操作台的搬箱数'
    ))

mcStationAcceptBinNum_fig.update_traces(
    texttemplate='%{text:.2s}', textposition='outside')

# 17-每个操作台每小时接受的箱子个数.csv
mcStationAcceptBinNumEachHour = pd.read_csv(
    'output/'+project_name+'/order_graph/17-每个操作台每小时接受的箱子个数.csv')
mcStationAcceptBinNumEachHour.hour = mcStationAcceptBinNumEachHour.hour.apply(
    lambda x: str(x))
mcStationAcceptBinNumEachHour_fig = px.bar(mcStationAcceptBinNumEachHour, x='StationNo', y='BinCount', color='hour',
                                           width=800, height=500)
mcStationAcceptBinNumEachHour_fig.update(
    layout=dict(
        xaxis=dict(title="操作台编号", tickangle=-0, showline=True, nticks=20),
        yaxis=dict(title="箱子数", showline=True),
        title='每个操作台每小时接受的箱子个数'
    ))


# 19-机器人每小时行走趟数列表.csv
mcKubotExeRoundEachHour = pd.read_csv(
    'output/'+project_name+'/order_graph/19-机器人每小时行走趟数列表.csv')
mcKubotExeRoundEachHour.hour = mcKubotExeRoundEachHour.hour.apply(
    lambda x: str(x))
mcKubotExeRoundEachHour_fig = px.bar(mcKubotExeRoundEachHour, x='KubotNo', y='Execute', color='hour',
                                     width=800, height=500)
mcKubotExeRoundEachHour_fig.update(
    layout=dict(
        xaxis=dict(title="模拟小时", tickangle=-0, showline=True, nticks=60),
        yaxis=dict(title="机器人编号", showline=True),
        title='机器人每小时行走趟数'
    ))


# 22-操作台空闲率总表.csv
mcStationIdleRate = pd.read_csv(
    'output/'+project_name+'/order_graph/22-操作台空闲率总表.csv')
mcStationIdleRate.StationNo = mcStationIdleRate.StationNo.apply(
    lambda x: str(x))
mcStationIdleRate_fig = px.line(mcStationIdleRate, x='hour', y='Idle_Rate', color='StationNo',
                                width=800, height=500)
mcStationIdleRate_fig.update(
    layout=dict(
        xaxis=dict(title="模拟小时", tickangle=-0, showline=True, nticks=30),
        yaxis=dict(title="操作台空闲率", showline=True),
        title='操作台空闲率概览'
    ))


# 11-单操作台箱子重取率.csv
singleStationBinRepeat = pd.read_csv(
    'output/'+project_name+'/order_graph/11-单操作台箱子重取率.csv')
singleStationBinRepeat.rename(columns={
    'StationNo': '操作台编号',
    'TotalBinCount': '接收箱子数',
    'RepeatBinCount': '重复接收箱子数',
    'BinRepeatRate(%)': '箱子重取率(%)'
}, inplace=True)
singleStationBinRepeat_T = ff.create_table(singleStationBinRepeat)
singleStationBinRepeat_T.update_layout(
    autosize=False,
    title_text='单操作台箱子重取率',
    margin={'t': 35, 'l': 30, 'r': 30, 'b': 30},
    title_font_family="Times New Roman",
    title_font_color="blue")

# 13-跨操作台重取分布.csv
mcCrossStationRepeatBinNum = pd.read_csv(
    'output/'+project_name+'/order_graph/13-多个操作台重取分布.csv')
mcCrossStationRepeatBinNum_fig = px.bar(mcCrossStationRepeatBinNum, x='跨操作台个数', y='箱子个数', text='箱子个数',
                                        width=800, height=500)  # ,orientation='h')
mcCrossStationRepeatBinNum_fig.update(
    layout=dict(
        xaxis=dict(title="跨操作台个数"),
        yaxis=dict(title="箱子个数"),
        title='多个操作台重取分布'
    ))
mcCrossStationRepeatBinNum_fig.update_traces(
    texttemplate='%{text:.2s}', textposition='outside')

# 14-同操作台重取分布.csv
sameStationBinRepeat = pd.read_csv(
    'output/'+project_name+'/order_graph/14-同操作台重取分布.csv')
sameStationBinRepeat.rename(columns={'StationNo': '操作台编号',
                                     'BinCarried_SameStation': '单箱子重复搬运次数',
                                     'BinId': '对应的箱子数'}, inplace=True)
sameStationBinRepeat['单箱子重复搬运次数'] = sameStationBinRepeat['单箱子重复搬运次数'].apply(
    lambda x: str(x))
sameStationBinRepeat_fig = px.bar(sameStationBinRepeat, x='操作台编号', y='对应的箱子数', color='单箱子重复搬运次数',
                                  width=800, height=500)
sameStationBinRepeat_fig.update(
    layout=dict(
        xaxis=dict(title="模拟小时", tickangle=-0, showline=True, nticks=20),
        yaxis=dict(title="机器人编号", showline=True),
        title='同操作台重取分布'
    ))

# # 31-每个波次的order_graph总件数和平均order_graph耗时.csv
# if len(delayDetailFile) != 0:
#     EachHourDelay = pd.read_csv(
#         'output/'+project_name+'/order_graph/31-每个波次的挂单总件数和平均挂单耗时.csv')
#     EachHourDelay.DelayTime = EachHourDelay.DelayTime.apply(
#         lambda x: round(x, 0))
#     EachHourDelay.rename(columns={
#         'Round': '波次',
#         'DelayTime': '订单平均延误耗时(s)',
#         'Qty': '件数'
#     }, inplace=True)
#     EachHourDelay_T = ff.create_table(EachHourDelay)
#     EachHourDelay_T.update_layout(
#         autosize=False,
#         title_text='每个波次的挂单总件数和平均挂单耗时',
#         margin={'t': 35, 'l': 30, 'r': 30, 'b': 30},
#         title_font_family="Times New Roman")
#
#     # 32-order_graph订单属性占比.csv
#     delayOrderType = pd.read_csv(
#         'output/'+project_name+'/order_graph/32-挂单订单属性占比.csv')
#     delayOrderType.rename(columns={
#         'OrderType': '订单类型',
#         'Qty': '件数'
#     }, inplace=True)
#     delayOrderType_T = ff.create_table(delayOrderType)
#     delayOrderType_T.update_layout(
#         autosize=False,
#         title_text='挂单订单属性占比',
#         margin={'t': 35, 'l': 30, 'r': 30, 'b': 30},
#         title_font_family="Times New Roman")
#
#     # 35-order_graph的SKU库存概况.csv
#     delaySKuInvOverview = pd.read_csv(
#         'output/'+project_name+'/order_graph/35-挂单的SKU库存概况.csv')
#     delaySKuInvOverview.rename(columns={
#         'stockBinCount': '存储使用箱子个数',
#         'sumQty': '全库总存储件数',
#         'avgQtyEachBin': '平均单箱存储件数'
#     }, inplace=True)
#     delaySKuInvOverview_T = ff.create_table(delaySKuInvOverview)
#     delaySKuInvOverview_T.update_layout(
#         autosize=False,
#         title_text='挂单的SKU库存概况',
#         margin={'t': 35, 'l': 30, 'r': 30, 'b': 30},
#         title_font_family="Times New Roman")
#
#     # 34-order_graph的箱子库存概况.csv
#     delayBinInvOverview = pd.read_csv(
#         'output/'+project_name+'/order_graph/34-挂单的箱子库存概况.csv')
#     delayBinInvOverview.rename(columns={
#         'locationId': '箱子编号',
#         'skuCount': '储存sku品种数',
#         'sumQty': '存储总件数',
#         'stockQtyEachSku': '平均单SKU件数'
#     }, inplace=True)
#     delayBinInvOverview_T = ff.create_table(delayBinInvOverview)
#     delayBinInvOverview_T.update_layout(
#         autosize=False,
#         title_text='挂单的箱子库存概况',
#         margin={'t': 35, 'l': 30, 'r': 30, 'b': 30},
#         title_font_family="Times New Roman",
#         title_font_color="blue"
#     )
#
#     # 33-order_graph明细表.csv
#     delayOverview = pd.read_csv(
#         'output/'+project_name+'/order_graph/33-挂单明细表.csv')
#     del delayOverview['Orig_OrderId']
#     del delayOverview['Time']
#     delayOverview.sort_values(by=['DelayTime'], ascending=True)
#     delayOverview.rename(columns={
#         'StationId': '操作台编号',
#         'OrderId': '订单编号',
#         'BinId': '箱子编号',
#         'Qty': '件数',
#         'DelayTime': '延误耗时(s)',
#         'Round': '波次',
#         'OrderType': '订单类型'
#     }, inplace=True)
#     delayOverview_T = ff.create_table(delayOverview)
#     delayOverview_T.update_layout(
#         autosize=False,
#         title_text='挂单明细表',
#         margin={'t': 35, 'l': 30, 'r': 30, 'b': 30},
#         title_font_family="Times New Roman",
#         title_font_color="blue")


with open('output/'+project_name+'/数据分析模拟报告.html', 'a') as f:
    f.write(simulationDataAnalisysReport.to_html(
        full_html=False, include_plotlyjs='cdn'))  # 数据分析模拟报告
    f.write(tableBasicOverview_T.to_html(
        full_html=False, include_plotlyjs='cdn'))  # 基础数据
    f.write(tableOrder_T.to_html(full_html=False,
                                 include_plotlyjs='cdn'))  # 订单基础信息
    f.write(tableInv_T.to_html(full_html=False,
                               include_plotlyjs='cdn'))  # 订单基础信息
    f.write(opAnalysis.to_html(full_html=False,
                               include_plotlyjs='cdn'))  # 订单分配数据概览
    f.write(tableOP_T.to_html(full_html=False,
                              include_plotlyjs='cdn'))  # 订单分配数据概览
    f.write(mcAnalysis.to_html(full_html=False,
                               include_plotlyjs='cdn'))  # 任务管理数据概览
    f.write(tableMC_T.to_html(full_html=False, include_plotlyjs='cdn'))  # MC基础信息
    f.write(op_finish.to_html(full_html=False,
                              include_plotlyjs='cdn'))  # 订单分配-订单完成概览
    f.write(outboundDf_fig.to_html(full_html=False, include_plotlyjs='cdn'))
    f.write(OpFinishEachHourOverview_T.to_html(full_html=False,
                                               include_plotlyjs='cdn'))  # 每个工作站出库的订单量订单行数出库件数
    f.write(mc_kubotBinCarry.to_html(
        full_html=False, include_plotlyjs='cdn'))  # 机器人搬箱
    f.write(mcKubotBinCarryEachHour_fig.to_html(
        full_html=False, include_plotlyjs='cdn'))  # 机器人每小时搬箱数列表
    f.write(KUBOTWORKtIMEEachHour_Df_Fig.to_html(
        full_html=False, include_plotlyjs='cdn'))  # 机器人每小时搬箱数曲线
    f.write(working_kubotDf_fig.to_html(
        full_html=False, include_plotlyjs='cdn'))  # 每小时工作机器人数
    f.write(roundworkingKubotBincarry_df_fig.to_html(
        full_html=False, include_plotlyjs='cdn'))  # 机器人平均每趟搬箱数
    f.write(mcKubotBinCarryEachRound_fig.to_html(
        full_html=False, include_plotlyjs='cdn'))  # 机器人每趟搬箱数占比
    # 机器人任务耗时-分布图(间隔时间 = 该任务的执行耗时 + 任务之间的间隔耗时)
    f.write(mcTime_df_fig.to_html(full_html=False, include_plotlyjs='cdn'))
    f.write(mcStationAcceptBinNum_fig.to_html(
        full_html=False, include_plotlyjs='cdn'))  # 每个操作台的搬箱数
    f.write(mcStationIdleRate_fig.to_html(
        full_html=False, include_plotlyjs='cdn'))  # 操作台空闲率总表
    f.write(bin_carry_for_hit_rate_df_fig.to_html(
        full_html=False, include_plotlyjs='cdn'))  # 命中率
    f.write(mc_kubotBinProcess.to_html(
        full_html=False, include_plotlyjs='cdn'))  # 操作台箱子处理
    f.write(bin_process_Bar.to_html(
        full_html=False, include_plotlyjs='cdn'))  # 单箱操作分布图
    f.write(singleStationBinRepeat_T.to_html(
        full_html=False, include_plotlyjs='cdn'))  # 单操作台箱子重取率

    f.write(delayOverview_T.to_html(full_html=False, include_plotlyjs='cdn'))

print("Program Complete !")