import geopandas
import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster
import plotly.express as px
from datetime import datetime
from st_aggrid import GridOptionsBuilder, AgGrid
from st_aggrid.shared import GridUpdateMode

st.set_page_config(page_title='House Rocket Company - Properties Dashboard',  layout='wide', page_icon=':house:')


def page_header(title_widths):
    t1, t2 = st.columns(title_widths)
    t1.image('home.png', width=120)
    t2.title("House Rocket Company - Properties Dashboard")
    # t2.markdown('Dashboard by Lucas Rodrigues')
    return t1, t2


@st.cache(allow_output_mutation=True)
def get_data(path):
    data = pd.read_csv(path)
    return data


@st.cache(allow_output_mutation=True)
def get_geofile(url):
    geofile = geopandas.read_file(url)
    return geofile


def set_feature(data):
    # Add new features
    data['price_ft2'] = data['price'] / data['sqft_lot']
    return data


def overview_data(data):
    st.sidebar.title('Data Overview')
    f_attributes = st.sidebar.multiselect('Enter columns', data.columns)
    f_zipcode = st.sidebar.multiselect('Enter zipcode', data['zipcode'].unique())

    st.title('Data Overview Updating')
    st.header('Data Overview Updating')

    if (f_zipcode != []) & (f_attributes != []):
        data = data.loc[data['zipcode'].isin(f_zipcode), f_attributes]

    elif (f_zipcode != []) & (f_attributes == []):
        data = data.loc[data['zipcode'].isin(f_zipcode), :]

    elif (f_zipcode == []) & (f_attributes != []):
        data = data.loc[:, f_attributes]

    else:
        data = data.copy()

    st.dataframe(data)
    c1, c2 = st.columns((1, 1))

    # Average metrics
    df1 = data[['id', 'zipcode']].groupby('zipcode').count().reset_index()
    df2 = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    df3 = data[['sqft_living', 'zipcode']].groupby('zipcode').count().reset_index()
    df4 = data[['price_ft2', 'zipcode']].groupby('zipcode').count().reset_index()

    # Merging
    m1 = pd.merge(df1, df2, on='zipcode', how='inner')
    m2 = pd.merge(m1, df3, on='zipcode', how='inner')
    df = pd.merge(m2, df4, on='zipcode', how='inner')

    df.columns = ['ZIPCODE', 'TOTAL HOUSES', 'PRICE', 'SQFT LIVING',
                  'PRICE/m2']

    c1.header('Average Values')
    c1.dataframe(df, height=600)

    # Descriptive Statistics
    num_attributes = data.select_dtypes(include=['int64', 'float64'])
    media = pd.DataFrame(num_attributes.apply(np.mean))
    mediana = pd.DataFrame(num_attributes.apply(np.median))
    std = pd.DataFrame(num_attributes.apply(np.std))

    max_ = pd.DataFrame(num_attributes.apply(np.max))
    min_ = pd.DataFrame(num_attributes.apply(np.min))

    df1 = pd.concat([max_, min_, media, mediana, std], axis=1).reset_index()
    df1.columns = ['attributes', 'maxi', 'min', 'mean', 'median', 'std']
    c2.header('Descriptive Analysis')
    c2.dataframe(df1, height=600)

    return None


def region_overview(data, geofile):
    st.title('Region Overview')

    c1, c2 = st.columns((1, 1))
    c1.header('Portfolio Density')

    # df = data
    df = data.sample(10, random_state=1045)

    # Base map - Folium
    density_map = folium.Map(location=[data['lat'].mean(), data['long'].mean()],
                             default_zoom_start=15, width='85%')

    marker_cluster = MarkerCluster().add_to(density_map)
    for name, row in df.iterrows():
        iframe = folium.IFrame = (
            'Sold R${0} on: {1}. Features: {2} sqft, {3} bedrooms, {4} bathrooms, Year Built: {5}'.format(row['price'],
                                                                                                          row['date'],
                                                                                                          row['sqft_living'],
                                                                                                          row['bedrooms'],
                                                                                                          row['bathrooms'],
                                                                                                          row['yr_built']))
        popup = folium.Popup(iframe, min_width=200, max_width=200)
        folium.Marker(location=[row['lat'], row['long']], popup=popup).add_to(marker_cluster)

    with c1:
        folium_static(density_map)

    # Region Price Map
    c2.header('Price Density')

    df = data[['price', 'zipcode', ]].groupby('zipcode').mean().reset_index()
    df.columns = ['ZIP', 'PRICE']

    df = df.sample(10, random_state=1045)

    geofile = geofile[geofile['ZIP'].isin(df['ZIP'].tolist())]

    region_price_map = folium.Map(location=[data['lat'].mean(), data['long'].mean()],
                                  default_zoom_start=15, width='85%')

    region_price_map.choropleth(data=df,
                                geo_data=geofile,
                                columns=['ZIP', 'PRICE'],
                                key_on='feature.properties.ZIP',
                                fill_color='YlOrRd',
                                fill_opacity=0.7,
                                line_opacity=0.2,
                                legend_name='AVERAGE PRICE')

    with c2:
        folium_static(region_price_map)

    return None


def commercial_distribution(data):
    st.sidebar.title('Commercial Options')
    st.title('Commercial Attributes')

    # ---------- Average Price per Year
    st.header('Average Price per Year')

    # Filters
    min_year_built = int(data['yr_built'].min())
    max_year_built = int(data['yr_built'].max())

    st.sidebar.subheader('Select Maximum Year Built')
    f_year_built = st.sidebar.slider('Year Built', min_value=min_year_built, max_value=max_year_built,
                                     value=max_year_built)

    # Data Selecion
    df = data.loc[data['yr_built'] < f_year_built]
    df = df[['yr_built', 'price']].groupby('yr_built').mean().reset_index()

    # Plot
    fig = px.line(df, x='yr_built', y='price')
    st.plotly_chart(fig, use_container_width=True)

    # ---------- Average Price per Day
    st.header('Average Price per Day')

    # Filters
    min_date = datetime.strptime(data['date'].min(), '%Y-%m-%d')
    max_date = datetime.strptime(data['date'].max(), '%Y-%m-%d')

    f_date = st.sidebar.slider('Date', min_value=min_date, max_value=max_date, value=max_date)
    # Data Selection
    data['date'] = pd.to_datetime(data['date'])
    df = data.loc[data['date'] < f_date]
    df = df[['date', 'price']].groupby('date').mean().reset_index()

    # Plot
    fig = px.line(df, x='date', y='price', title='Average Price per Day')
    st.plotly_chart(fig, use_container_width=True)

    # ---------- Histogramas
    st.header('Price Distribution')
    st.sidebar.subheader('Select Max Price')

    # Filters
    min_price = int(data['price'].min())
    max_price = int(data['price'].max())
    avg_price = int(data['price'].mean())

    f_price = st.sidebar.slider('Price', min_value=min_price, max_value=max_price, value=avg_price)

    # Data Selection
    df = data.loc[data['price'] < f_price]

    # Plot
    fig = px.histogram(df, x='price', nbins=50)
    st.plotly_chart(fig, use_container_width=True)

    return None


def attributes_distribution(data):
    # ===============================================
    # Distributed dos imóveis por categorias físicas
    # ===============================================

    st.sidebar.title('House Attributes Options')
    st.sidebar.title('Attributes Options')
    st.title('House Attributes')

    # Filters
    f_bedrooms = st.sidebar.selectbox('Max number of bedrooms', data['bedrooms'].sort_values().unique(), index=(len(data['bedrooms'].sort_values().unique())-1))

    f_bathrooms = st.sidebar.selectbox('Max number of bathrooms', data['bathrooms'].sort_values().unique(), index=(len(data['bathrooms'].sort_values().unique())-1))

    f_floors = st.sidebar.selectbox('Max number of floors', data['floors'].sort_values().unique(), index=len(data['floors'].sort_values().unique())-1)

    f_waterview = st.sidebar.checkbox('Only houses with waterview')

    c1, c2 = st.columns(2)

    # Houses per bedrooms
    c1.header('Houses per bedrooms')
    df = data.loc[data['bedrooms'] < f_bedrooms]
    fig = px.histogram(df, x='bedrooms', nbins=19)
    c1.plotly_chart(fig, use_container_width=True)

    # Houses per bathrooms
    c2.header('Houses per bathrooms')
    df = data.loc[data['bathrooms'] < f_bathrooms]
    fig = px.histogram(data, x='bathrooms', nbins=19)
    c2.plotly_chart(fig, use_container_width=True)

    # Houses per floors
    c1, c2 = st.columns(2)
    c1.header('Houses per floor')
    df = data.loc[data['floors'] < f_floors]
    fig = px.histogram(df, x='floors', nbins=19)
    c1.plotly_chart(fig, use_container_width=True)

    # Houses per waterview
    c2.header('Houses per floor')
    if f_waterview:
        df = data[data['waterfront'] == 1]
    else:
        df = data.copy()

    fig = px.histogram(df, x='waterfront', nbins=10)
    c2.plotly_chart(fig, use_container_width=True)

    return None


# Definindo uma tabela interativa

def tabela_interativa(df: pd.DataFrame):
    options = GridOptionsBuilder.from_dataframe(
        df, enableRowGroup=True, enableValue=True, enablePivot=True
    )

    options.configure_side_bar()

    options.configure_selection("single")
    selection = AgGrid(
        df,
        enable_enterprise_modules=True,
        gridOptions=options.build(),
        theme="light",
        update_mode=GridUpdateMode.MODEL_CHANGED,
        allow_unsafe_jscode=True,
    )
    return selection

if __name__ == "__main__":
    # --Data Extraction
    # Get data
    path = 'kc_house_data.csv'
    url = 'https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson'
    data = get_data(path)
    title_widths = (0.15, 1)

    # Convert date type
    data['date'] = pd.to_datetime(data['date']).dt.strftime('%Y-%m-%d')
    print(data.dtypes)
    # Get geofile
    geofile = get_geofile(url)

    # --Data Transformation
    page_header(title_widths)

    data = set_feature(data)

    overview_data(data)

    region_overview(data, geofile)

    commercial_distribution(data)

    attributes_distribution(data)

    selection = tabela_interativa(df=data)

