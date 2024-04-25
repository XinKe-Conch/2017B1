import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from sklearn.cluster import KMeans
import folium
from folium.plugins import MarkerCluster

# 读取execl数据
data = pd.read_excel('附件二：会员信息数据.xlsx')
data[['经度', '纬度']]=data['会员位置(GPS)'].str.split(" ", expand=True).dropna(axis=1, how='any').astype('float')

#使用IQR（四分位距）方法来识别和删除位置数据中的异常
Q1 = data[['经度', '纬度']].quantile(0.25)
Q3 = data[['经度', '纬度']].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 筛选没有异常值的数据（根据经度和纬度）
data = data[(data['经度'] >= lower_bound['经度']) & (data['经度'] <= upper_bound['经度']) &
            (data['纬度'] >= lower_bound['纬度']) & (data['纬度'] <= upper_bound['纬度'])]

#用kmeans聚类算法分类
gdf = gpd.GeoDataFrame(data, geometry=data.apply(lambda row: Point(float(row.经度), float(row.纬度)), axis=1))
kmeans = KMeans(n_clusters=3, init='k-means++')
gdf['聚类结果'] = kmeans.fit_predict(gdf[['经度', '纬度']])

# 创建地图
m = folium.Map(location=[gdf['纬度'].mean(), gdf['经度'].mean()], zoom_start=7)  # 假设gdf['纬度'].mean()和gdf['经度'].mean()分别是纬度和经度的平均值

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
marker_cluster = MarkerCluster().add_to(m)
for _, row in gdf.iterrows():
    folium.Marker(location=[row['经度'], row['纬度']],
                  popup='Cluster: {}'.format(row['聚类结果']),
                  icon=folium.Icon(color='green' if row['聚类结果'] == 0 else 'blue' if row['聚类结果'] == 1 else 'red')).add_to(marker_cluster)

# 显示地图
m.save('map.html')  # 保存地图到HTML文件中