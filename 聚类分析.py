import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from sklearn.cluster import KMeans

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

#绘图
fig, ax = plt.subplots()
gdf.plot(column="聚类结果",categorical=True, legend=True, ax=ax,zorder=5)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.show()