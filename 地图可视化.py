import folium
import pandas as pd
centerLoc = 10, 10
m = folium.Map(
               zoom_start=5,
               control_scale=True,
               control=False,
               tiles=None
               )

folium.TileLayer(tiles='http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}',
                 attr="&copy; <a href=http://ditu.amap.com/>高德地图</a>",
                 min_zoom=0,
                 max_zoom=19,
                 control=True,
                 show=True,
                 overlay=False,
                 name=1
                 ).add_to(m)
data = pd.read_excel("附件一：已结束项目任务数据.xlsx")
for index, row in data.iterrows():
    if row["任务执行情况"] == 1:
        folium.CircleMarker([row["任务gps 纬度"], row["任务gps经度"]],
                radius=4, # define how big you want the circle markers to be
                color='green',
                fill=True,
                fill_color='green',
                fill_opacity=1
                ).add_to(m)
    if row["任务执行情况"] == 0:
        folium.CircleMarker([row["任务gps 纬度"], row["任务gps经度"]],
                radius=4, # define how big you want the circle markers to be
                color='#54534D',
                fill=True,
                fill_color='#54534D',
                fill_opacity=1).add_to(m)
m.save('已结束项目任务数据的可视化.html')