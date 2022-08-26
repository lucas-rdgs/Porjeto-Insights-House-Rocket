import geopandas
from datetime import datetime
import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster
import plotly.express as px
from st_aggrid import GridOptionsBuilder, AgGrid
from st_aggrid.shared import GridUpdateMode

pd.options.display.float_format = '{:,.2f}'.format

st.set_page_config(page_title='House Rocket Company - Sell Dashboard', layout='centered', page_icon=':house:')

def page_header(title_widths):
    t1, t2 = st.columns(title_widths)
    t1.title('')
    t1.image('home.png', use_column_width=True)
    t2.title("House Rocket Company - Sell Dashboard")
    t2.markdown('Dashboard by Lucas Rodrigues')
    return t1, t2


if __name__ == "__main__":
    title_widths = (0.15, 0.85)
    page_header(title_widths)


# @st.cache(allow_output_mutation=True)
# def get_data(path):
#     data = pd.read_csv(path)
#     return data
#
# @st.cache(allow_output_mutation=True)
# def get_geofile(url):
#     geofile = geopandas.read_file(url)
#     return geofile
#
# @st.cache(allow_output_mutation=True)
# def data_cleaning(data):
#     # Convert date type
#     data['date'] = pd.to_datetime(data['date'])
#     data['bathrooms'] = data['bathrooms'].astype(dtype='int64')
#     data['floors'] = data['floors'].astype(dtype='int64')
#
#     # Removing rows containing zeros
#     data = data[(data['bathrooms'] != 0) & (data['bedrooms'] != 0)]
#
#     # Rounding up the values of columns "bathrooms" and "floors"
#     data['bathrooms'] = data['bathrooms'].apply(np.ceil)
#     data['floors'] = data['floors'].apply(np.ceil)
#
#     # Creating columns converting square feet to square meters
#     data['m2_living'] = data['sqft_living'].apply(lambda x: x / 10.7639).astype(dtype='int64')
#     data['m2_lot'] = data['sqft_lot'].apply(lambda x: x / 10.7639).astype(dtype='int64')
#     data['m2_above'] = data['sqft_above'].apply(lambda x: x / 10.7639).astype(dtype='int64')
#     data['m2_basement'] = data['sqft_basement'].apply(lambda x: x / 10.7639).astype(dtype='int64')
#     data['m2_living15'] = data['sqft_living15'].apply(lambda x: x / 10.7639).astype(dtype='int64')
#     data['m2_lot15'] = data['sqft_lot15'].apply(lambda x: x / 10.7639).astype(dtype='int64')
#
#     # Creating a columns "condition_description" with text description for the properties' condition
#     data['condition_description'] = data['condition'].apply(lambda x: 'excellent' if x == 5 else
#                                                                       'good' if x == 4 else
#                                                                       'fair' if x == 3 else
#                                                                       'poor' if x == 2 else
#                                                                       'bad')
#
#     # # - Creating the column "season" of each property selling date
#     # # "day of year" ranges for the northern hemisphere
#     spring = range(80, 172)
#     summer = range(172, 264)
#     fall = range(264, 355)
#     # # winter for anyone else
#
#     data['season'] = data['date'].apply(lambda x: 'spring' if x.timetuple().tm_yday in spring else
#                                                   'summer' if x.timetuple().tm_yday in summer else
#                                                   'fall' if x.timetuple().tm_yday in fall else
#                                                   'winter')
#
#     return data
#
# def get_data_sell(data):
#     df_median = data[['zipcode', 'price']].groupby('zipcode').median().reset_index()
#     df_median.columns = ['zipcode', 'price_median']
#
#     data = pd.merge(data, df_median, on='zipcode', how='inner')
#
#     data_purchase = pd.DataFrame()
#     data_purchase[['id', 'zipcode', 'purchase_price', 'price_median', 'condition']] = data[
#         ['id', 'zipcode', 'price', 'price_median', 'condition']]
#
#     for i in range(len(data_purchase)):
#         if (data_purchase.loc[i, 'purchase_price'] < data_purchase.loc[i, 'price_median']) & (
#                 data_purchase.loc[i, 'condition'] >= 4):
#             data_purchase.loc[i, 'status'] = 'Buy'
#         else:
#             data_purchase.loc[i, 'status'] = 'Do not buy'
#
#     data = pd.merge(data, data_purchase[['id', 'status']], on='id', how='inner')
#
#     df_median_season = data[['zipcode', 'season', 'price']].groupby(['zipcode', 'season']).median().reset_index()
#     df_median_season.columns = ['zipcode', 'season', 'price_median_season']
#
#     data = pd.merge(data, df_median_season, on=['zipcode', 'season'], how='inner')
#
#     data_sell = pd.DataFrame()
#     data_sell[['id', 'zipcode', 'season', 'price', 'price_median', 'condition', 'condition_description', 'status']] = \
#         data[
#             ['id', 'zipcode', 'season', 'price', 'price_median_season', 'condition', 'condition_description', 'status']]
#     data_sell = data_sell[data_sell['status'] == 'Buy'].reset_index(drop=True)
#
#     for i in range(len(data_sell)):
#         if data_sell.loc[i, 'price'] < data_sell.loc[i, 'price_median']:
#             data_sell.loc[i, 'sell_price'] = data_sell.loc[i, 'price'] * 1.3
#
#         else:
#             data_sell.loc[i, 'sell_price'] = data_sell.loc[i, 'price'] * 1.1
#
#         data_sell.loc[i, 'gain'] = data_sell.loc[i, 'sell_price'] - data_sell.loc[i, 'price']
#
#     data_sell.drop(columns=['status'], axis=1, inplace=True)
#
#     return data_sell
#
#
# def get_data_sell_full(data_sell, data):
#     data_sell_full = pd.merge(data_sell, data[['id', 'lat', 'long', 'date', 'bedrooms',
#                                                'bathrooms',  'sqft_living', 'floors', 'waterfront', 'yr_built']], on='id', how='inner')
#
#     return data_sell_full
#
#
# def region_overview(data_sell_full, geofile):
#     st.title('Region Overview')
#
#     tab1, tab2, tab3 = st.tabs(['Portfolio Density', 'Sell Price Density', 'Gain Density'])
#     # c1, c2 = st.columns((1, 1))
#
#     # df = data
#     # df = data.sample(10, random_state=1045)
#     df1 = data_sell_full
#
#     # Base map - Folium
#     density_map = folium.Map(location=[data_sell_full['lat'].mean(), data_sell_full['long'].mean()],
#                              default_zoom_start=15, width='100%')
#     marker_cluster = MarkerCluster().add_to(density_map)
#     for name, row in df1.iterrows():
#         iframe = folium.IFrame = (
#             f"Selling price ${row['sell_price']:,.2f}. Purchased for ${row['price']:,.2f} on: {row['date']:%Y-%m-%d}. \nFeatures: {row['sqft_living']} sqft, "
#             f"{row['bedrooms']} bedrooms, {row['bathrooms']:.0f} bathrooms, \nYear Built: {row['yr_built']}")
#         popup = folium.Popup(iframe, min_width=200, max_width=200, min_height=200, max_height=200)
#         folium.Marker(location=[row['lat'], row['long']], popup=popup, tooltip="Click to see property information").add_to(marker_cluster)
#
#     with tab1:
#         st.header('Portfolio Density')
#         folium_static(density_map)
#
#     # Region Selling Price Map
#
#     df2 = data_sell_full[['sell_price', 'zipcode', ]].groupby('zipcode').mean().reset_index()
#     df2.columns = ['ZIP', 'PRICE']
#
#     # df = df.sample(10, random_state=1045)
#     # df = data_sell_full
#
#     geofile = geofile[geofile['ZIP'].isin(df2['ZIP'].tolist())]
#     geofile = pd.merge(geofile, df2[['ZIP', 'PRICE']], on='ZIP', how='inner')
#
#     region_price_map = folium.Map(location=[data_sell_full['lat'].median(), data_sell_full['long'].median()],
#                                   zoom_start=9.5, width='100%', height='90%')
#
#     marker_cluster_price = MarkerCluster().add_to(region_price_map)
#     for name, row in data_sell_full.iterrows():
#         iframe = folium.Iframe = (
#             f"Selling price ${row['sell_price']:,.2f}. Purchased for ${row['price']:,.2f} on: {row['date']:%Y-%m-%d}. "
#             f"Gain of ${row['gain']:,.2f} "
#             f"Features: {row['sqft_living']} sqft,"
#             f"{row['bedrooms']} bedrooms, {row['bathrooms']:.0f} bathrooms, \nYear Built: {row['yr_built']}")
#         popup = folium.Popup(iframe, min_width=200, max_width=200, min_height=200, max_height=200)
#         folium.Marker(location=[row['lat'], row['long']], popup=popup,
#                       tooltip="Click to see property information").add_to(marker_cluster_price)
#
#     region_price_map.choropleth(data=df2,
#                                 geo_data=geofile,
#                                 columns=['ZIP', 'PRICE'],
#                                 key_on='feature.properties.ZIP',
#                                 fill_color='YlOrRd',
#                                 fill_opacity=0.7,
#                                 line_opacity=0.2,
#                                 legend_name='AVERAGE SELLING PRICE')
#
#     style_function = lambda x: {'fillColor': '#ffffff',
#                                 'color': '#000000',
#                                 'fillOpacity': 0.1,
#                                 'weight': 0.1}
#     highlight_function = lambda x: {'fillColor': '#000000',
#                                     'color': '#000000',
#                                     'fillOpacity': 0.50,
#                                     'weight': 0.1}
#
#     hover = folium.features.GeoJson(
#             data=geofile,
#             style_function=style_function,
#             control=False,
#             highlight_function=highlight_function,
#             tooltip=folium.features.GeoJsonTooltip(
#                     fields=['ZIP', 'PRICE'],
#                     aliases=['Region Zipcode: ', 'Average Selling Price: '],
#                     style="background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;",
#                     localize=True
#             )
#     )
#     region_price_map.add_child(hover)
#     region_price_map.keep_in_front(hover)
#
#     with tab2:
#         st.header('Selling Price Density')
#         folium_static(region_price_map)
#
#     # Region Gain Map
#
#     df3 = data_sell_full[['gain', 'zipcode', ]].groupby('zipcode').mean().reset_index()
#     df3.columns = ['ZIP', 'GAIN']
#
#     # df = df.sample(10, random_state=1045)
#     # df = data_sell_full
#
#     geofile = geofile[geofile['ZIP'].isin(df3['ZIP'].tolist())]
#     geofile_gain = pd.merge(geofile, df3[['ZIP', 'GAIN']], on='ZIP', how='inner')
#
#     region_gain_map = folium.Map(location=[data_sell_full['lat'].mean(), data_sell_full['long'].mean()],
#                                  default_zoom_start=15)
#
#     marker_cluster_gain = MarkerCluster().add_to(region_gain_map)
#     for name, row in data_sell_full.iterrows():
#         iframe = folium.Iframe = (
#             f"Selling price ${row['sell_price']:,.2f}. Purchased for ${row['price']:,.2f} on: {row['date']:%Y-%m-%d}. "
#             f"Gain of ${row['gain']:,.2f} "
#             f"Features: {row['sqft_living']} sqft,"
#             f"{row['bedrooms']} bedrooms, {row['bathrooms']:.0f} bathrooms, \nYear Built: {row['yr_built']}")
#         popup = folium.Popup(iframe, min_width=200, max_width=200, min_height=200, max_height=200)
#         folium.Marker(location=[row['lat'], row['long']], popup=popup,
#                       tooltip="Click to see property information").add_to(marker_cluster_gain)
#
#     region_gain_map.choropleth(data=df3,
#                                geo_data=geofile_gain,
#                                columns=['ZIP', 'GAIN'],
#                                key_on='feature.properties.ZIP',
#                                fill_color='YlOrRd',
#                                fill_opacity=0.7,
#                                line_opacity=0.2,
#                                legend_name='AVERAGE GAIN')
#
#     style_function = lambda x: {'fillColor': '#ffffff',
#                                 'color': '#000000',
#                                 'fillOpacity': 0.1,
#                                 'weight': 0.1}
#     highlight_function = lambda x: {'fillColor': '#000000',
#                                     'color': '#000000',
#                                     'fillOpacity': 0.50,
#                                     'weight': 0.1}
#
#     hover_gain = folium.features.GeoJson(
#         data=geofile_gain,
#         style_function=style_function,
#         control=False,
#         highlight_function=highlight_function,
#         tooltip=folium.features.GeoJsonTooltip(
#             fields=['ZIP', 'GAIN'],
#             aliases=['Region Zipcode: ', 'Average Gain: '],
#             style="background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;",
#             localize=True
#         )
#     )
#     region_gain_map.add_child(hover_gain)
#     region_gain_map.keep_in_front(hover_gain)
#
#     with tab3:
#         st.header('Gain Density')
#         folium_static(region_gain_map)
#
#     return None
#
#
# if __name__ == "__main__":
#     # --Data Extraction
#     # Get data
#     path = 'kc_house_data.csv'
#     url = 'https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson'
#     data = get_data(path)
#     title_widths = (0.10, 1)
#
#     data = data_cleaning(data)
#     data_sell = get_data_sell(data)
#     data_sell_full = get_data_sell_full(data_sell, data)
#
#     # Get geofile
#     geofile = get_geofile(url)
#
#     region_overview(data_sell_full, geofile)
