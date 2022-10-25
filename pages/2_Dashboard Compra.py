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

# st.set_page_config(page_title='House Rocket Company - Sell Dashboard', layout='centered', page_icon=':house:')
st.set_page_config(page_title='House Rocket Company - Dashboard de Compras', layout='centered', page_icon=':house:')


def page_header(title_widths):
    t1, t2 = st.columns(title_widths)
    t1.title('')
    t1.image('https://cdn-icons-png.flaticon.com/512/6760/6760104.png', use_column_width=True)
    # t2.title("House Rocket Company - Sell Dashboard")
    t2.title("House Rocket Company - Dashboard de Compras")
    return t1, t2


@st.cache(allow_output_mutation=True)
def get_data(path):
    data = pd.read_csv(path)
    return data


@st.cache(allow_output_mutation=True)
def get_geofile(url):
    geofile = geopandas.read_file(url)
    return geofile


def data_cleaning(data):
    # Convert date type
    # Convertendo os tipos de dados
    data['date'] = pd.to_datetime(data['date'])
    data['bathrooms'] = data['bathrooms'].astype(dtype='int64')
    data['floors'] = data['floors'].astype(dtype='int64')

    # Removing rows containing zeros
    # Removendo linhas que possuem zeros
    data = data[(data['bathrooms'] != 0) & (data['bedrooms'] != 0)]

    # Rounding up the values of columns "bathrooms" and "floors"
    # Arredondando os valores das colunas "banheiros" e "andares"
    data['bathrooms'] = data['bathrooms'].apply(np.ceil)
    data['floors'] = data['floors'].apply(np.ceil)

    # Creating columns converting square feet to square meters
    # Criando colunas convertendo pés quadrados para metros quadrados
    data['m2_living'] = data['sqft_living'].apply(lambda x: x / 10.7639).astype(dtype='int64')
    data['m2_lot'] = data['sqft_lot'].apply(lambda x: x / 10.7639).astype(dtype='int64')
    data['m2_above'] = data['sqft_above'].apply(lambda x: x / 10.7639).astype(dtype='int64')
    data['m2_basement'] = data['sqft_basement'].apply(lambda x: x / 10.7639).astype(dtype='int64')
    data['m2_living15'] = data['sqft_living15'].apply(lambda x: x / 10.7639).astype(dtype='int64')
    data['m2_lot15'] = data['sqft_lot15'].apply(lambda x: x / 10.7639).astype(dtype='int64')

    # Creating a column "condition_descritption" with text description for the properties' condition
    # Criando a coluna "condition_descritption" com a descrição em texto das condições dos imóveis
    data['condition_description'] = data['condition'].apply(lambda x: 'excellent' if x == 5 else
                                                                      'good'      if x == 4 else
                                                                      'fair'      if x == 3 else
                                                                      'poor'      if x == 2 else
                                                                      'bad')

    # # - Creating the column "season" of each property selling date
    # # "day of year" ranges for the northern hemisphere
    # spring = range(80, 172)
    # summer = range(172, 264)
    # fall = range(264, 355)
    # # winter for anyone else
    #
    # data['season'] = data['date'].apply(lambda x: 'spring' if x.timetuple().tm_yday in spring else
    #                                               'summer' if x.timetuple().tm_yday in summer else
    #                                               'fall'   if x.timetuple().tm_yday in fall   else
    #                                               'winter')

    # - Criando a coluna "season" para cada data de venda dos imóveis
    # intervalos para "dia do ano" para o hemisfério norte
    spring = range(80, 172)
    summer = range(172, 264)
    fall = range(264, 355)
    # winter (inverno) para todos os outros dias

    data['season'] = data['date'].apply(lambda x: 'spring' if x.timetuple().tm_yday in spring else
                                                  'summer' if x.timetuple().tm_yday in summer else
                                                  'fall' if x.timetuple().tm_yday in fall else
                                                  'winter')

    return data


def get_data_purchase(data):
    # - Calculating the price median per zipcode
    # Grouping by zipcodes and calculating the price medians
    df_median = data[['zipcode', 'price']].groupby('zipcode').median().reset_index()
    df_median.columns = ['zipcode', 'price_median']

    # Creating the columns "price_median" for each row with its zipcode
    data = pd.merge(data, df_median, on='zipcode', how='inner')

    # - Creating the purchase table
    data_purchase = pd.DataFrame()
    data_purchase[['id', 'zipcode', 'price', 'price_median', 'condition', 'condition_description']] = data[
        ['id', 'zipcode', 'price', 'price_median', 'condition', 'condition_description']]

    for i in range(len(data_purchase)):
        if (data_purchase.loc[i, 'price'] < data_purchase.loc[i, 'price_median']) & \
                (data_purchase.loc[i, 'condition'] >= 4):
            data_purchase.loc[i, 'status'] = 'Buy'
        else:
            data_purchase.loc[i, 'status'] = 'Do not buy'

    return data_purchase


def get_data_purchase_full(data_purchase, data):
    data_purchase_full = pd.merge(data_purchase, data[['id', 'lat', 'long', 'date', 'bedrooms',
                                                       'bathrooms',  'sqft_living', 'floors', 'waterfront',
                                                       'yr_built']], on='id', how='inner')

    return data_purchase_full


def interactive_purchase(df: pd.DataFrame):
    # st.header('Properties overview')
    st.header('Visão geral dos Imóveis')
    st.markdown('Nesta tabela encontram-se as informações de todos os imóveis disponíveis.')
    st.markdown('Para navegar, utilize as barras de rolagem vertical e horizontal e os filtros localizados no lado'
                'direito da tabela.')
    options = GridOptionsBuilder.from_dataframe(
        df, enableRowGroup=True, enableValue=True, enablePivot=True
    )

    options.configure_side_bar()

    options.configure_column('price',
                             type=["numericColumn", "numberColumnFilter", "customNumericFormat"],
                             precision=2)
    options.configure_column('price_median', header_name='price median',
                             type=["numericColumn", "numberColumnFilter", "customNumericFormat"],
                             precision=2)
    options.configure_column('condition_description', header_name='condition description')
    options.configure_column('sqft_living', header_name='living total area (sqft)')
    options.configure_column('waterfront', header_name='has waterview')
    options.configure_column('yr_built', header_name='year of construction')

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


def region_overview(data, geofile):
    # st.header('Properties overview')
    st.header('Visão geral por Região')

    # tab1, tab2 = st.tabs(['Portfolio Density', 'Price Density'])
    tab1, tab2 = st.tabs(['Densidade do Portfólio', 'Densidade de Preços'])

    df = data.sample(10, random_state=1045)

    # Base map - Folium
    # Mapa base - Folium
    density_map = folium.Map(location=[data['lat'].mean(), data['long'].mean()],
                             default_zoom_start=15, width='85%')
    marker_cluster = MarkerCluster().add_to(density_map)
    # for name, row in df.iterrows():
    #     iframe = folium.IFrame = (f"Sold for ${row['price']:,.2f} on: {row['date']:%Y-%m-%d}.  "
    #                               f"Condition: {row['condition_description']}. Features: {row['sqft_living']} sqft, "
    #                               f"{row['bedrooms']} bedrooms, {row['bathrooms']:.0f} bathrooms, "
    #                               f"Year Built: {row['yr_built']}")
    #     popup = folium.Popup(iframe, min_width=200, max_width=200)
    #     folium.Marker(location=[row['lat'], row['long']], popup=popup).add_to(marker_cluster)

    for name, row in df.iterrows():
        iframe = folium.IFrame = (f"Preço: ${row['price']:,.2f}. Data da venda: {row['date']:%Y-%m-%d}.  "
                                  f"Condição: {row['condition_description']}. Atributos: {row['sqft_living']} sqft, "
                                  f"{row['bedrooms']} quarto, {row['bathrooms']:.0f} banheiros, "
                                  f"Ano de Construção: {row['yr_built']}")
        popup = folium.Popup(iframe, min_width=200, max_width=200)
        folium.Marker(location=[row['lat'], row['long']], popup=popup).add_to(marker_cluster)

    with tab1:
        # st.header('Portfolio Density')
        st.header('Densidade do Portfólio')
        folium_static(density_map)

    # Region Price Map
    # Mapa de Preços por Região

    df = data[['price', 'zipcode', ]].groupby('zipcode').mean().reset_index()
    df.columns = ['ZIP', 'PRICE']

    df = df.sample(10, random_state=1045)

    geofile = geofile[geofile['ZIP'].isin(df['ZIP'].tolist())]

    region_price_map = folium.Map(location=[data['lat'].mean(), data['long'].mean()],
                                  default_zoom_start=15, width='85%')

    marker_cluster_price = MarkerCluster().add_to(region_price_map)
    # for name, row in data_purchase_full.iterrows():
    #     iframe = folium.Iframe = (
    #         f"Sold for ${row['price']:,.2f}. Purchased on: {row['date']:%Y-%m-%d}. "
    #         f"Condition: {row['condition_description']}. "
    #         f"Features: {row['sqft_living']} sqft,"
    #         f"{row['bedrooms']} bedrooms, {row['bathrooms']:.0f} bathrooms, \nYear Built: {row['yr_built']}")
    #     popup = folium.Popup(iframe, min_width=200, max_width=200, min_height=200, max_height=200)
    #     folium.Marker(location=[row['lat'], row['long']], popup=popup,
    #                   tooltip="Click to see property information").add_to(marker_cluster_price)

    for name, row in data_purchase_full.iterrows():
        iframe = folium.Iframe = (
            f"Preço: ${row['price']:,.2f}. Comprada em: {row['date']:%Y-%m-%d}. "
            f"Condição: {row['condition_description']}. "
            f"Atributos: {row['sqft_living']} sqft,"
            f"{row['bedrooms']} quartos, {row['bathrooms']:.0f} banheiros, \nAno de Construção: {row['yr_built']}")
        popup = folium.Popup(iframe, min_width=200, max_width=200, min_height=200, max_height=200)
        folium.Marker(location=[row['lat'], row['long']], popup=popup,
                      tooltip="Clique para ver as informações do imóvel").add_to(marker_cluster_price)

    # region_price_map.choropleth(data=df,
    #                             geo_data=geofile,
    #                             columns=['ZIP', 'PRICE'],
    #                             key_on='feature.properties.ZIP',
    #                             fill_color='YlOrRd',
    #                             fill_opacity=0.7,
    #                             line_opacity=0.2,
    #                             legend_name='AVERAGE PRICE')

    region_price_map.choropleth(data=df,
                                geo_data=geofile,
                                columns=['ZIP', 'PRICE'],
                                key_on='feature.properties.ZIP',
                                fill_color='YlOrRd',
                                fill_opacity=0.7,
                                line_opacity=0.2,
                                legend_name='PREÇO MÉDIO')

    with tab2:
        # st.header('Price Density')
        st.header('Densidade de Preços')
        folium_static(region_price_map)

    return None


def commercial_distribution(data):
    # st.sidebar.title('Commercial Options')
    # st.title('Commercial Attributes')
    st.sidebar.title('Opções Comerciais')
    st.title('Atributos Comerciais')

    # # ---------- Average Price per Year
    # st.header('Average Price per Year')
    # ---------- Preço Médio por Ano
    st.header('Preço Médio por Ano')

    # Filters
    # Filtros
    min_year_built = int(data['yr_built'].min())
    max_year_built = int(data['yr_built'].max())

    # st.sidebar.subheader('Select Maximum Year Built')
    st.sidebar.subheader('Selecione o Ano de Construção Máximo')
    f_year_built = st.sidebar.slider('Year Built', min_value=min_year_built, max_value=max_year_built,
                                     value=max_year_built)

    # Data Selecion
    # Seleção de Dados
    df = data.loc[data['yr_built'] < f_year_built]
    df = df[['yr_built', 'price']].groupby('yr_built').mean().reset_index()

    # Plot
    # Gráfico
    fig = px.line(df, x='yr_built', y='price')
    st.plotly_chart(fig, use_container_width=True)

    # # ---------- Average Price per Day
    # st.header('Average Price per Day')
    # ---------- Average Price per Day
    st.header('Preços Médios por Dia')

    # Filters
    # Filtros
    data['date1'] = pd.to_datetime(data['date']).dt.strftime('%Y-%m-%d')
    min_date = datetime.strptime(data['date1'].min(), '%Y-%m-%d')
    max_date = datetime.strptime(data['date1'].max(), '%Y-%m-%d')

    f_date = st.sidebar.slider('Date', min_value=min_date, max_value=max_date, value=max_date)

    # Data Selection
    # Seleção de Dados
    data['date'] = pd.to_datetime(data['date'])
    df = data.loc[data['date'] < f_date]
    df = df[['date', 'price']].groupby('date').mean().reset_index()

    # Plot
    # Gráfico
    fig = px.line(df, x='date', y='price', title='Average Price per Day')
    st.plotly_chart(fig, use_container_width=True)

    # # ---------- Histograms
    # st.header('Price Distribution')
    # st.sidebar.subheader('Select Max Price')
    # ---------- Histogramas
    st.header('Distribuição de Preços')
    st.sidebar.subheader('Selecione o Preço Máximo')

    # Filters
    # Filtros
    min_price = int(data['price'].min())
    max_price = int(data['price'].max())
    avg_price = int(data['price'].mean())

    f_price = st.sidebar.slider('Price', min_value=min_price, max_value=max_price, value=avg_price)

    # Data Selection
    # Seleção de Dados
    df = data.loc[data['price'] < f_price]

    # Plot
    # Gráfico
    fig = px.histogram(df, x='price', nbins=50)
    st.plotly_chart(fig, use_container_width=True)

    return None


def attributes_distribution(data):
    # ===============================================
    # Distribuição dos imóveis por categorias físicas
    # ===============================================

    # st.sidebar.title('House Attributes Options')
    # st.sidebar.title('Attributes Options')
    # st.title('House Attributes')

    st.sidebar.title('Opções de Atributos dos Imóveis')
    st.sidebar.title('Opções de Atributos')
    st.title('Atributos dos Imóveis')

    # # Filters
    # f_bedrooms = st.sidebar.selectbox('Max number of bedrooms', data['bedrooms'].sort_values().unique(),
    #                                   index=(len(data['bedrooms'].sort_values().unique())-1))
    #
    # f_bathrooms = st.sidebar.selectbox('Max number of bathrooms', data['bathrooms'].sort_values().unique(),
    #                                    index=(len(data['bathrooms'].sort_values().unique())-1))
    #
    # f_floors = st.sidebar.selectbox('Max number of floors', data['floors'].sort_values().unique(),
    #                                 index=len(data['floors'].sort_values().unique())-1)
    #
    # # f_waterview = st.sidebar.checkbox('Only houses with waterview')

    # Filtros
    f_bedrooms = st.sidebar.selectbox('Número máximo de quartos', data['bedrooms'].sort_values().unique(),
                                      index=(len(data['bedrooms'].sort_values().unique()) - 1))

    f_bathrooms = st.sidebar.selectbox('Número máximo de banheiros', data['bathrooms'].sort_values().unique(),
                                       index=(len(data['bathrooms'].sort_values().unique()) - 1))

    f_floors = st.sidebar.selectbox('Número máximo de andares', data['floors'].sort_values().unique(),
                                    index=len(data['floors'].sort_values().unique()) - 1)

    f_waterview = st.sidebar.checkbox('Apenas imóveis com vista para água')

    c1, c2 = st.columns(2)

    # # Houses per bedrooms
    # c1.header('Houses per number of bedrooms')
    # df = data.loc[data['bedrooms'] < f_bedrooms]
    # fig = px.histogram(df, x='bedrooms', nbins=19)
    # c1.plotly_chart(fig, use_container_width=True)
    #
    # # Houses per bathrooms
    # c2.header('Houses per number of bathrooms')
    # df = data.loc[data['bathrooms'] < f_bathrooms]
    # fig = px.histogram(data, x='bathrooms', nbins=19)
    # c2.plotly_chart(fig, use_container_width=True)
    #
    # # Houses per floors
    # c1, c2 = st.columns(2)
    # c1.header('Houses per number of floor')
    # df = data.loc[data['floors'] < f_floors]
    # fig = px.histogram(df, x='floors', nbins=19)
    # c1.plotly_chart(fig, use_container_width=True)
    #
    # # Houses per waterview
    # c2.header('Houses with of without waterview')
    # if f_waterview:
    #     df = data[data['waterfront'] == 1]
    # else:
    #     df = data.copy()

    # Imóveis por número de quartos
    c1.header('Imóveis por número de quartos')
    df = data.loc[data['bedrooms'] < f_bedrooms]
    fig = px.histogram(df, x='bedrooms', nbins=19)
    c1.plotly_chart(fig, use_container_width=True)

    # Imóveis por número de banheiros
    c2.header('Imóveis por número de banheiros')
    df = data.loc[data['bathrooms'] < f_bathrooms]
    fig = px.histogram(df, x='bathrooms', nbins=19)
    c2.plotly_chart(fig, use_container_width=True)

    # Imóveis por número de andares
    c1, c2 = st.columns(2)
    c1.header('Imóveis por número de andares')
    df = data.loc[data['floors'] < f_floors]
    fig = px.histogram(df, x='floors', nbins=19)
    c1.plotly_chart(fig, use_container_width=True)

    # Imóveis com vista para a água
    c2.header('Imóveis com vista para água')
    if f_waterview:
        df = data[data['waterfront'] == 1]
    else:
        df = data.copy()

    fig = px.histogram(df, x='waterfront', nbins=3)
    c2.plotly_chart(fig, use_container_width=True)

    return None


if __name__ == "__main__":
    # --Data Extraction
    # Get data

    # --Extração de Dados
    # Carregar Dados
    path = 'kc_house_data.csv'
    url = 'https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson'
    data = get_data(path)
    title_widths = (0.15, 0.85)

    data = data_cleaning(data)
    data_purchase = get_data_purchase(data)
    data_purchase_full = get_data_purchase_full(data_purchase, data)

    # Get geofile
    # Carregar geofile
    geofile = get_geofile(url)

    page_header(title_widths)
    selection = interactive_purchase(df=data_purchase_full[['id', 'zipcode', 'price', 'price_median',
                                                            'condition_description', 'status', 'bedrooms', 'bathrooms',
                                                            'sqft_living', 'floors', 'waterfront', 'yr_built']])
    region_overview(data, geofile)

    commercial_distribution(data)

    attributes_distribution(data)
