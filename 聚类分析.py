import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from sklearn.cluster import KMeans
import folium
import numpy as np
from folium.plugins import MarkerCluster

# 读取execl数据
data = pd.read_excel('附件二：会员信息数据.xlsx')
data[['纬度', '经度']]=data['会员位置(GPS)'].str.split(" ", expand=True).dropna(axis=1, how='any').astype('float')

#使用IQR（四分位距）方法来识别和删除位置数据中的异常
Q1 = data[['纬度', '经度']].quantile(0.25)
Q3 = data[['纬度', '经度']].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR


# 筛选没有异常值的数据（根据经度和纬度）
data = data[(data['经度'] >= lower_bound['经度']) & (data['经度'] <= upper_bound['经度']) &
            (data['纬度'] >= lower_bound['纬度']) & (data['纬度'] <= upper_bound['纬度'])]

# 应用KMeans++聚类
gdf = gpd.GeoDataFrame(data, geometry=data.apply(lambda row: Point(float(row.经度), float(row.纬度)), axis=1))
kmeans = KMeans(n_clusters=3, init='k-means++')
gdf['聚类结果'] = kmeans.fit_predict(gdf[['纬度', '经度']])

# 获取聚类中心的坐标
centroids = kmeans.cluster_centers_
print("聚类中心的坐标是：")
print(centroids)

from sklearn.metrics.pairwise import euclidean_distances

# 计算每个点到聚类中心的距离
gdf['与聚类中心的距离'] = gdf.apply(lambda row: euclidean_distances([centroids[row['聚类结果']]], [[row['纬度'], row['经度']]])[0][0], axis=1)
max_dist_points = gdf.loc[gdf.groupby('聚类结果')['与聚类中心的距离'].idxmax()]
max_dist_dict = gdf.groupby('聚类结果')['与聚类中心的距离'].max().to_dict()
gdf['偏僻程度'] = gdf.apply(lambda row: row['与聚类中心的距离'] / max_dist_dict[row['聚类结果']], axis=1)

# 创建地图
m = folium.Map(location=[gdf['纬度'].mean(), gdf['经度'].mean()], zoom_start=7)

# 假设gdf['纬度'].mean()和gdf['经度'].mean()分别是纬度和经度的平均值
folium.TileLayer(tiles='http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}',
                 attr="&copy; <a href=http://ditu.amap.com/>高德地图</a>",
                 min_zoom=0,
                 max_zoom=19,
                 control=True,
                 show=True,
                 overlay=False,
                 name=1
                 ).add_to(m)
# 添加聚类点
for _, row in gdf.iterrows():
    folium.CircleMarker(
        location=[row['纬度'], row['经度']],
        radius=1,
        popup='Cluster: {}'.format(row['聚类结果']),
        color='red' if row['聚类结果'] == 0 else 'blue' if row['聚类结果'] == 1 else 'green',
        fill=True,
        fill_color='red' if row['聚类结果'] == 0 else 'blue' if row['聚类结果'] == 1 else 'green',
        fill_opacity=1
    ).add_to(m)


# 将聚类中心添加到地图上
for i, centroid in enumerate(centroids):
    folium.CircleMarker(
        location=[centroid[0], centroid[1]],
        radius=7,
        popup='Cluster Center: {}'.format(i),
        color='black',
        fill=True,
        fill_color='black'
    ).add_to(m)

# 读取附件一：已结束项目任务数据
task_data = pd.read_excel('附件一：已结束项目任务数据.xlsx')
task_gdf = gpd.GeoDataFrame(task_data, geometry=gpd.points_from_xy(task_data['任务gps经度'], task_data['任务gps 纬度']))

# 设置1.5km的缓冲半径（用经纬度单位表示）
buffer_radius_in_degrees = 1.5 / 111  # 球面上每度大约等于111km，这是一个经验值

# 计算每个任务点的1.5km缓冲区内的会员数量
def calculate_members_within_radius(task_geometry, members_gdf, buffer_radius):
    # 创建任务点的缓冲区
    task_buffer = task_geometry.buffer(buffer_radius)
    # 计算在缓冲区内的会员数量
    members_within_buffer = members_gdf[members_gdf.within(task_buffer)]
    return len(members_within_buffer)

# 计算每个任务点周围1.5km范围内的会员数量，并添加到 task_gdf 中
task_gdf['会员数量'] = task_gdf.geometry.apply(calculate_members_within_radius, args=(gdf, buffer_radius_in_degrees))

# 计算每个任务点的1.5km缓冲区内的其他任务点数量
def calculate_tasks_within_radius(task_row, task_gdf, buffer_radius):
    # 创建任务点的缓冲区
    task_buffer = task_row.geometry.buffer(buffer_radius)
    # 计算在缓冲区内的其他任务点数量（排除自身）
    tasks_within_buffer = task_gdf[task_gdf.geometry.within(task_buffer) & (task_gdf['任务号码'] != task_row['任务号码'])]
    return len(tasks_within_buffer)

# 为task_gdf中的每个任务行调用calculate_tasks_within_radius函数
task_gdf['周围任务数量'] = task_gdf.apply(calculate_tasks_within_radius, axis=1, task_gdf=task_gdf, buffer_radius=buffer_radius_in_degrees)

# 打印任务点及其相应1.5km内的会员数量和周围任务数量
print(task_gdf[['任务号码', '会员数量', '周围任务数量']])
task_gdf.to_excel("输出附件一.xlsx", index=False)
gdf.to_excel("处理附件二.xlsx", index=False)
print(gdf)
