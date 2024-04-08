import pandas as pd
import numpy as np
from PIL import Image
from streamlit_extras.add_vertical_space import add_vertical_space
import seaborn as sns
import streamlit as st
from streamlit_player import st_player
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_option_menu import option_menu
import streamlit_extras.metric_cards as metric_cards
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import gaussian_kde
import plotly.express as px
sns.set_style("dark")
import geopandas as gpd
image=Image.open("air.png")

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
    page_title="Airbnb Analysis",
    page_icon=image,
    )

with st.sidebar:
    st.markdown("<hr style='border: 2px solid #FF385C;'>", unsafe_allow_html=True)
    
    selected = option_menu("Main Menu", ["Home", 'Data Exploration','EDA'], 
        icons=['house-door-fill','bar-chart-fill','credit-card'], menu_icon="cast", default_index=0,styles={
        "container": {"padding": "0!important", "background-color": "gray"},
        "icon": {"color": "rgb(235, 48, 84)", "font-size": "25px","font-family":"inherit"}, 
        "nav-link": {"font-family":"inherit","font-size": "22px", "color": "#ffffff","text-align": "left", "margin":"0px", "--hover-color": "#84706E"},
        "nav-link-selected": {"font-family":"inherit","background-color": "azure","color": "#FF385C","font-size": "25px"},
    })
    st.markdown("<hr style='border: 2px solid #FF385C;'>", unsafe_allow_html=True)


st.markdown(""" <style> button[data-baseweb="tab"] > di v[data-testid="stMarkdownContainer"] > p {font-size: 28px;} </style>""", unsafe_allow_html=True)



st.markdown("<h1 style='text-align: center; font-size: 38px; color: #FF385C ; font-weight: 700;font-family:inherit;'>Airbnb Listings Data Analysis</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border: 2px solid beige;'>", unsafe_allow_html=True)

if selected == "Home":

    about="""Airbnb is a platform that allows individuals to rent out their homes or properties to travelers, 
    offering a wide range of accommodations and experiences worldwide.
    """
    image=Image.open("Emblem.png")
    st.image(image,use_column_width=True,caption=about)
    add_vertical_space(1)


    st.markdown("<h3 style='font-size: 30px;text-align:left; font-family:inherit;color: #FF385C;'> What is Airbnb?  </h3>", unsafe_allow_html=True)

    st.markdown("""<p  style='text-align: left; font-size: 18px; color: #ffffff; font-weight: 400;font-family:inherit;'>  
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;      
                        Airbnb is an online marketplace that connects people who want to rent out
                their property with people who are looking for accommodations,
                typically for short stays. Airbnb offers hosts a relatively easy way to
                earn some income from their property.Guests often find that Airbnb rentals
                are cheaper and homier than hotels.
                </p>""", unsafe_allow_html=True)


    add_vertical_space(1)

    st_player(url=("https://www.youtube.com/watch?v=dA2F0qScxrI"))

    add_vertical_space(1)


    st.markdown("<h3 style='font-size: 30px;text-align:left; font-family:inherit;color: #FF385C;'> About Airbnb:  </h3>", unsafe_allow_html=True)
    about="""Airbnb was born in 2007 when two Hosts welcomed three guests to their San Francisco home, and has since grown to over 5 million Hosts who have welcomed over 
    1.5 billion guest arrivals in almost every country across the globe. Every day, Hosts offer unique stays and experiences that make it possible for guests to connect 
    with communities in a more authentic way.
    """

    st.markdown(f"""<p  style='text-align: left; font-size: 18px; color: #ffffff; font-weight: 400;font-family:inherit;'>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    {about} </p>""", unsafe_allow_html=True)
       
@st.cache_resource(ttl=None)
def filter_df(df,col,filter):
    if filter != 'All':
        df=df[col == filter].reset_index(drop= True)
    else:
        df=df.copy()
    return df

@st.cache_data(ttl=None,persist="disk")
def create_plotly_charts(data, chart_type, _x_column, y_column, widthv=None,heightv=None,grid=False,**kwargs):
    if chart_type == 'Pie':
        fig = px.pie(data, values=y_column, names=_x_column, **kwargs)
    if chart_type == 'lne':
        fig=px.line(data,x=_x_column,y=y_column,**kwargs)
    elif chart_type == 'Treemap':
        fig=px.treemap(data,path=_x_column,values=y_column,**kwargs)
    elif chart_type=='densitymap':
        fig=px.density_mapbox(data,lat=_x_column,lon=y_column,**kwargs)
    elif chart_type == 'Bar':
        if 'category_orders' in kwargs:
            category_orders = kwargs.pop('category_orders')
        else:
            category_orders = None
        if 'text_auto' in kwargs:
            kwargs.pop('text_auto')  # Remove 'text_auto' from kwargs
            fig = px.bar(data, x=_x_column, y=y_column, text=y_column, **kwargs).update_layout(width=widthv, height=heightv)
        else:
            fig = px.bar(data, x=_x_column, y=y_column, **kwargs).update_layout(width=widthv, height=heightv)

    if not grid:
        if chart_type == 'Bar':
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=False)
    return fig

df=pd.read_csv("final_df.csv")

country= st.sidebar.selectbox("Select the Country",["All"] + list(map(str, df["country"].unique())))

reg=list(df["region"].unique())
if country  != 'All':
    reg=df.loc[df.country==country,'region'].unique()

region= st.sidebar.selectbox("Select the Region",["All"] + list(map(str, reg)))



if selected == "Data Exploration":
    st.markdown('<style>div.css-1jpvgo6 {font-size: 16px; font-weight: bolder;font-family:inherit; } </style>', unsafe_allow_html=True)

    tab, tab1, tab2, tab3, tab4= st.tabs(["***LISTINGS COUNT***","***PRICE ANALYSIS***","***AVAILABILITY ANALYSIS***","***LOCATION BASED***", "***GEOSPATIAL VISUALIZATION***"])
    with tab:

        col1,col2=st.columns(2)

        
        df0=filter_df(df,df.country,country)    
        df0=filter_df(df0,df0.region,region)
        
        property_counts = pd.DataFrame(df0.groupby(['property_type']).agg({'_id':'count','price':'median'})).reset_index()
        room_counts = pd.DataFrame(df0.groupby(['room_type']).agg({'_id':'count','price':'median'})).reset_index()

        st.markdown("<hr style='border: 2px solid #FF385C;'>", unsafe_allow_html=True)
        
        st.header("**Listings Count by Property Type and Room Type:**")
        
        colpie,space , colcount=st.columns([1,.00001,1])
        
        with colpie:

            st.plotly_chart(create_plotly_charts(property_counts, 'Pie', 'property_type', '_id',hover_data=['price'],hole=0.35,color_discrete_sequence=px.colors.sequential.Magma,)
            .update_traces(textinfo='percent+value',hoverinfo='label+text',visible=True,textfont=dict(size=15,color='white',)
                
                          ,hoverlabel=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='white')),hovertemplate='<b>Property Type:</b> %{label}<br>' +
                      '<b>Average Price:</b> %{customdata[0]:.2f}<br>' )
            .update_layout(width=400, height=400,   legend_font=dict(size=13,color='white'),legend=dict(bgcolor='rgba(0,0,0,0)'),legend_title_text='Property Type',legend_title_font=dict(size=14,color='white'))
            ,use_container_width=True)

        space.markdown("<div style='border-left: 3px solid #FF385C;font-family:PhonePeSans; height: 400px;'></div>", unsafe_allow_html=True)
            
        with colcount:
             st.plotly_chart(create_plotly_charts(room_counts, 'Pie', 'room_type', '_id',hover_data=['price'],hole=0.35,color_discrete_sequence=px.colors.sequential.Magma)
             .update_traces(textinfo='percent+value',hoverinfo='label+text',visible=True,textfont=dict(size=15,color='white',)
                
                          ,hoverlabel=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='white')),hovertemplate='<b>Room Type:</b> %{label}<br>' +
                      '<b>Average Price:</b> %{customdata[0]:.2f}<br>' )
            .update_layout(width=400, height=400,   legend_font=dict(size=13,color='white'),legend=dict(bgcolor='rgba(0,0,0,0)'),legend_title_text='Room Type',legend_title_font=dict(size=14,color='white'))
            ,use_container_width=True)

        st.markdown("<hr style='border: 2px solid #FF385C;'>", unsafe_allow_html=True)

        st.header("**Listing count in terms of cancellation policy and comparing with Property & Room type:**")
        
        cancel_room = df0.groupby(['cancellation_policy','property_type']).size().reset_index(name='count').sort_values(by='count',ascending=False)
        
        a,b,c=st.columns([1,.0001,1])

        a.plotly_chart(create_plotly_charts(cancel_room,'Bar','cancellation_policy','count',color_discrete_sequence=px.colors.qualitative.Safe,color='property_type')
        .update_traces(hoverlabel=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='white')),visible=True,showlegend=True,textfont=dict(size=18, color='white'),)
            .update_layout(width=400, height=400,xaxis={'categoryorder': 'total descending'},yaxis={'categoryorder': 'total ascending'},   legend_font=dict(size=13,color='white'),legend=dict(bgcolor='rgba(0,0,0,0)'),legend_title_text='Property Type',legend_title_font=dict(size=14,color='white'))
            ,use_container_width=True)
        
        b.markdown("<div style='border-left: 3px solid #FF385C;font-family:PhonePeSans; height: 400px;'></div>", unsafe_allow_html=True)

        cancel_prpty = df0.groupby(['cancellation_policy','room_type']).size().reset_index(name='count').sort_values(by='count',ascending=False)

        c.plotly_chart(create_plotly_charts(cancel_prpty,'Bar','cancellation_policy','count',color_discrete_sequence=px.colors.qualitative.Safe_r,color='room_type')
        .update_traces(hoverlabel=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='white')),visible=True,showlegend=True,textfont=dict(size=18, color='white'),)
            .update_layout(width=400, height=400,xaxis={'categoryorder': 'total descending'},yaxis={'categoryorder': 'total ascending'},   legend_font=dict(size=13,color='white'),legend=dict(bgcolor='rgba(0,0,0,0)'),legend_title_text='Room Type',legend_title_font=dict(size=14,color='white'))
            ,use_container_width=True)
        
        st.markdown("<hr style='border: 2px solid #FF385C;'>", unsafe_allow_html=True)

        st.header("**Listings Count by Region and Room Type:**")
    

        room_type_counts = df0.groupby(['region','room_type']).size().reset_index(name='count').sort_values(by='count',ascending=False)
        
        
        st.plotly_chart(create_plotly_charts(room_type_counts,'Bar','region','count',color_discrete_sequence=px.colors.qualitative.Dark2,color='room_type')
       .update_traces(hoverlabel=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='white')),visible=True,showlegend=True,textfont=dict(size=18, color='white'),)
            .update_layout(width=400, height=400,xaxis={'categoryorder': 'total descending'},yaxis={'categoryorder': 'total ascending'},   legend_font=dict(size=13,color='azure'),legend=dict(bgcolor='rgba(0,0,0,0)'),legend_title_text='Room Type',legend_title_font=dict(size=14,color='azure'))
            ,use_container_width=True)

        # st.markdown("<hr style='border: .5px  #FF385C;'>", unsafe_allow_html=True)
        
        st.header("**Listings Count by Region and Property Type:**")
        room_type_ctry = df0.groupby(['region', 'property_type']).size().reset_index(name='count').sort_values(by='count',ascending=False)
        
        st.plotly_chart(create_plotly_charts(room_type_ctry,'Bar','region','count',color_discrete_sequence=px.colors.qualitative.Dark2,color='property_type')
        .update_traces(hoverlabel=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='white')),visible=True,showlegend=True,textfont=dict(size=18, color='white'),)
            .update_layout(width=400, height=400,xaxis={'categoryorder': 'total descending'},yaxis={'categoryorder': 'total ascending'},   legend_font=dict(size=13,color='azure'),legend=dict(bgcolor='rgba(0,0,0,0)'),legend_title_text='Room Type',legend_title_font=dict(size=14,color='azure'))
            ,use_container_width=True)

        
        st.markdown("<hr style='border: 2px solid #FF385C;'>", unsafe_allow_html=True)



    
        a,b,c=st.columns([1,.0001,1])

        c.header("**Listings Count by country and Room Type:**")

        room_type_counts = df0.groupby(['country','room_type']).size().reset_index(name='count').sort_values(by='count',ascending=False)
        
        
        c.plotly_chart(create_plotly_charts(room_type_counts,'Bar','country','count',color_discrete_sequence=px.colors.qualitative.Set1,color='room_type')
        .update_traces(hoverlabel=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='white')),visible=True,showlegend=True,textfont=dict(size=18, color='white'),)
            .update_layout(width=400, height=400,xaxis={'categoryorder': 'total descending'},yaxis={'categoryorder': 'total ascending'},   legend_font=dict(size=13,color='azure'),legend=dict(bgcolor='rgba(0,0,0,0)'),legend_title_text='Room Type',legend_title_font=dict(size=14,color='azure'))
            ,use_container_width=True)

        b.markdown("<div style='border-left: 3px solid #FF385C;font-family:PhonePeSans; height: 500px;'></div>", unsafe_allow_html=True)
        
        a.header("**Listings Count by country and Property Type:**")
        room_type_ctry = df0.groupby(['country', 'property_type']).size().reset_index(name='count').sort_values(by='count',ascending=False)
        
        a.plotly_chart(create_plotly_charts(room_type_ctry,'Bar','country','count',color_discrete_sequence=px.colors.qualitative.Set1_r,color='property_type')
        .update_traces(hoverlabel=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='white')),visible=True,showlegend=True,textfont=dict(size=18, color='white'),)
            .update_layout(width=400, height=400,xaxis={'categoryorder': 'total descending'},yaxis={'categoryorder': 'total ascending'},   legend_font=dict(size=13,color='azure'),legend=dict(bgcolor='rgba(0,0,0,0)'),legend_title_text='Room Type',legend_title_font=dict(size=14,color='azure'))
            ,use_container_width=True)

        
        st.markdown("<hr style='border: 2px solid #FF385C;'>", unsafe_allow_html=True)

        st.header('Top 15 hosts interms of Listings count with Superhost breakup:')

        top_host=pd.DataFrame(df0.groupby(['host_id','superhost']).agg({'number_of_reviews':'count','host_total_listings_count':'sum'}).reset_index()).nlargest(15,'host_total_listings_count')

        st.plotly_chart(create_plotly_charts(top_host,'Bar','host_id','host_total_listings_count',color='superhost',color_discrete_sequence=px.colors.qualitative.Alphabet)
        .update_traces(hovertemplate='<b>Host ID:</b> %{label}<br>' +
                      '<b>Count:</b> %{y:,.2f}<br>' ,hoverlabel=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='white')),
                      visible=True,showlegend=True,textfont=dict(size=12, color='#ffffff'),textposition='outside',texttemplate='%{y:,.2f} ')
            .update_layout(width=400, height=450,xaxis={'categoryorder': 'total descending'},yaxis={'title': 'Price', 'categoryorder': 'total ascending'},   legend_font=dict(size=13,color='white'),legend=dict(bgcolor='rgba(0,0,0,0)'),legend_title_text='Room Type',legend_title_font=dict(size=14,color='white'),barmode='stack', )
            .update_xaxes(title='Host_id', type='category') ,use_container_width=True)

        st.markdown("<hr style='border: 2px solid #FF385C;'>", unsafe_allow_html=True)


    with tab1:
        
        # col1,col2=st.columns(2)
        # country= col1.selectbox("Select the Country",["All"] + list(map(str, df["country"].unique())),key=1)
        
        # reg=list(df["region"].unique())

        # if country  != 'All':
        #     reg=df.loc[df.country==country,'region'].unique()

        # region= col2.selectbox("Select the Region",["All"] + list(map(str, reg)),key=2)

        # df0=filter_df(df,df.country,country)
        # df0=filter_df(df0,df0.region,region)

        st.markdown("<hr style='border: 2px solid #FF385C;'>", unsafe_allow_html=True)


        st.header("**Average price in terms of Property Type and Room Type:**")

        a,b,c=st.columns([1,.0001,1])

        price_prpty = df0.groupby('property_type')['price'].median().reset_index(name='price').sort_values(by='price',ascending=False)

        a.plotly_chart(create_plotly_charts(price_prpty,'Bar','property_type','price',text='price',color_discrete_sequence=px.colors.qualitative.Dark2,color='property_type')
       .update_traces(hovertemplate='<b>Property Type:</b> %{label}<br>' +
                      '<b>Average Price:</b> %{y:.2f}<br>' ,hoverlabel=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='white')),
                      visible=True,showlegend=False,textfont=dict(size=12, color='#ffffff'),textposition='outside',texttemplate='%{y:,.2f} ')
            .update_layout(width=400, height=450,xaxis={'categoryorder': 'total descending'},yaxis={'categoryorder': 'total ascending'},   legend_font=dict(size=13,color='white'),legend=dict(bgcolor='rgba(0,0,0,0)'),legend_title_text='Property Type',legend_title_font=dict(size=14,color='white'))
            ,use_container_width=True)
        
        b.markdown("<div style='border-left: 3px solid #FF385C;font-family:PhonePeSans; height: 400px;'></div>", unsafe_allow_html=True)

        price_room = df0.groupby('room_type')['price'].median().reset_index(name='price').sort_values(by='price',ascending=False)

        c.plotly_chart(create_plotly_charts(price_room,'Bar','room_type','price',color_discrete_sequence=px.colors.qualitative.Dark2,color='room_type')
        .update_traces(hovertemplate='<b>Room Type:</b> %{label}<br>' +
                      '<b>Average Price:</b> %{y:.2f}<br>' ,hoverlabel=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='white')),
                      visible=True,showlegend=False,textfont=dict(size=12, color='#ffffff'),textposition='outside',texttemplate='%{y:,.2f} ')
            .update_layout(width=400, height=450,xaxis={'categoryorder': 'total descending'},yaxis={'categoryorder': 'total ascending'},   legend_font=dict(size=13,color='white'),legend=dict(bgcolor='rgba(0,0,0,0)'),legend_title_text='Room Type',legend_title_font=dict(size=14,color='white'))
            ,use_container_width=True)
        
        st.markdown("<hr style='border: 2px solid #FF385C;'>", unsafe_allow_html=True)


        st.header("**Average price in terms of Cancellation policy and Accomodates**")
        
        a,b,c=st.columns([.6,.0001,1])

        price_cancel = df0.groupby('cancellation_policy')['price'].median().reset_index(name='price').sort_values(by='price',ascending=False)

        a.plotly_chart(create_plotly_charts(price_cancel,'Bar','cancellation_policy','price',color_discrete_sequence=px.colors.qualitative.Dark2,color='cancellation_policy')
        .update_traces(hovertemplate='<b>Policy Type:</b> %{label}<br>' +
                      '<b>Average Price:</b> %{y:.2f}<br>' ,hoverlabel=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='white')),
                      visible=True,showlegend=False,textfont=dict(size=12, color='#ffffff'),textposition='outside',texttemplate='%{y:,.2f} ')
            .update_layout(width=400, height=400,xaxis={'categoryorder': 'total descending'},yaxis={'categoryorder': 'total ascending'},   legend_font=dict(size=13,color='white'),legend=dict(bgcolor='rgba(0,0,0,0)'),legend_title_text='Property Type',legend_title_font=dict(size=14,color='white'))
            ,use_container_width=True)
        
        b.markdown("<div style='border-left: 3px solid #FF385C;font-family:PhonePeSans; height: 400px;'></div>", unsafe_allow_html=True)


        
        
        price_accom = df0.groupby('accommodates')['price'].median().reset_index(name='price').sort_values(by='price',ascending=False)
        
        c.plotly_chart(create_plotly_charts(price_accom,'Bar','accommodates','price',color='accommodates',color_continuous_scale='Viridis_r',)
        .update_traces(hovertemplate='<b>Accommodates Count:</b> %{label}<br>' +
                      '<b>Average Price:</b> %{y:.2f}<br>' ,hoverlabel=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='white')),
                      visible=True,showlegend=False,textfont=dict(size=12, color='#ffffff'),textposition='outside',texttemplate='%{y:,.2f} ')
            .update_layout(width=400, height=400,xaxis={'categoryorder': 'total descending'},yaxis={'title': 'Price', 'categoryorder': 'total ascending'},   legend_font=dict(size=13,color='white'),legend=dict(bgcolor='rgba(0,0,0,0)'),legend_title_text='Room Type',legend_title_font=dict(size=14,color='white'),barmode='stack', )
            .update_xaxes(title='Accommodates', type='category') ,use_container_width=True)

        st.markdown("<hr style='border: 2px solid #FF385C;'>", unsafe_allow_html=True)
        

        st.header("**Price Trends Over Time:**")
        df0['last_scraped'] = pd.to_datetime(df0['last_scraped'])


        time=df0.groupby(['room_type','last_scraped'])['price'].median().reset_index(name='price').sort_values(by='last_scraped')
        time['last_scraped'] = pd.to_datetime(time['last_scraped'])

        fig=px.line(data_frame=time,x='last_scraped',y='price',color='room_type',color_discrete_sequence=px.colors.qualitative.Dark2,)\
            .update_traces(hovertemplate='<b>Last_Scraped:</b> %{x}<br>' +
                      '<b>Average Price:</b> %{y:.2f}<br>' ,hoverlabel=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='white')),
                      visible=True,showlegend=True,textfont=dict(size=12, color='#ffffff'),textposition='top center',texttemplate='%{y:,.2f} ')\
        .update_layout(width=400, height=400,xaxis={'categoryorder': 'total descending'},yaxis={'title': 'Price', 'categoryorder': 'total ascending'},   legend_font=dict(size=13,color='white'),legend=dict(bgcolor='rgba(0,0,0,0)'),legend_title_text='Room Type',legend_title_font=dict(size=14,color='white'),barmode='stack', )
        
        st.plotly_chart(fig,use_container_width=True)
        


        time=df0.groupby(['superhost','last_scraped'])['price'].median().reset_index(name='price').sort_values(by='last_scraped')

        fig=px.line(data_frame=time,x='last_scraped',y='price',color='superhost',color_discrete_sequence=px.colors.qualitative.Dark2,)\
            .update_traces(hovertemplate='<b>Last_Scraped:</b> %{x}<br>' +
                      '<b>Average Price:</b> %{y:.2f}<br>' ,hoverlabel=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='white')),
                      visible=True,showlegend=True,textfont=dict(size=12, color='#ffffff'),textposition='top center',texttemplate='%{y:,.2f} ')\
        .update_layout(width=400, height=400,xaxis={'categoryorder': 'total descending'},yaxis={'title': 'Price', 'categoryorder': 'total ascending'},   legend_font=dict(size=13,color='white'),legend=dict(bgcolor='rgba(0,0,0,0)'),legend_title_text='superhost',legend_title_font=dict(size=14,color='white'),barmode='stack', )
        
        st.plotly_chart(fig,use_container_width=True)

        

        st.markdown("<hr style='border: 2px solid #FF385C;'>", unsafe_allow_html=True)

        st.header('Price interms of country, Region with No of reviews breakup')

        a,b,c=st.columns([.8,.0001,1])
        
        price_cty=pd.DataFrame(df0.groupby('country').agg({'number_of_reviews':'count','price':'median'}).reset_index())

        a.plotly_chart(create_plotly_charts(price_cty,'Bar','country','price',color='number_of_reviews',color_continuous_scale='cividis',)
        .update_traces(hovertemplate='<b>Country:</b> %{label}<br>' +
                      '<b>Average Price:</b> %{y:.2f}<br>' ,hoverlabel=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='white')),
                      visible=True,showlegend=False,textfont=dict(size=12, color='#ffffff'),textposition='outside',texttemplate='%{y:,.2f} ')
            .update_layout(coloraxis_colorbar=dict(title='Reviews'),width=400, height=450,xaxis={'categoryorder': 'total descending'},yaxis={'title': 'Price', 'categoryorder': 'total ascending'},   legend_font=dict(size=13,color='white'),legend=dict(bgcolor='rgba(0,0,0,0)'),legend_title_text='Room Type',legend_title_font=dict(size=14,color='white'),barmode='stack', )
            .update_xaxes(title='Country', type='category') ,use_container_width=True)

        b.markdown("<div style='border-left: 3px solid #FF385C;font-family:PhonePeSans; height: 400px;'></div>", unsafe_allow_html=True)

        price_region=pd.DataFrame(df0.groupby('region').agg({'number_of_reviews':'count','price':'median'}).reset_index())

        c.plotly_chart(create_plotly_charts(price_region,'Bar','region','price',color='number_of_reviews',color_continuous_scale='cividis',)
        .update_traces(hovertemplate='<b>Region:</b> %{label}<br>' +
                      '<b>Average Price:</b> %{y:.2f}<br>' ,hoverlabel=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='white')),
                      visible=True,showlegend=False,textfont=dict(size=12, color='#ffffff'),textposition='outside',texttemplate='%{y:,.2f} ')
            .update_layout(coloraxis_colorbar=dict(title='Reviews'),width=400, height=450,xaxis={'categoryorder': 'total descending'},yaxis={'title': 'Price', 'categoryorder': 'total ascending'},   legend_font=dict(size=13,color='white'),legend=dict(bgcolor='rgba(0,0,0,0)'),legend_title_text='Room Type',legend_title_font=dict(size=14,color='white'),barmode='stack', )
            .update_xaxes(title='Region', type='category') ,use_container_width=True)

        st.markdown("<hr style='border: 2px solid #FF385C;'>", unsafe_allow_html=True)
        

        
        st.header('Price interms of Minimun, Maximum nights with No of reviews breakup')

        a,b,c=st.columns([.8,.0001,1])
        
        price_min=pd.DataFrame(df0.groupby('minimum_nights').agg({'number_of_reviews':'count','price':'median'}).reset_index())

        a.plotly_chart(create_plotly_charts(price_min,'Bar','minimum_nights','price',color='number_of_reviews',color_continuous_scale='Inferno',)
        .update_traces(hovertemplate='<b>Minimum Nights:</b> %{label}<br>' +
                      '<b>Average Price:</b> %{y:.2f}<br>' ,hoverlabel=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='white')),
                      visible=True,showlegend=False,textfont=dict(size=12, color='#ffffff'),textposition='outside',texttemplate='%{y:,.2f} ')
            .update_layout(coloraxis_colorbar=dict(title='Reviews'),width=400, height=450,xaxis={'categoryorder': 'total descending'},yaxis={'title': 'Price', 'categoryorder': 'total ascending'},   legend_font=dict(size=13,color='white'),legend=dict(bgcolor='rgba(0,0,0,0)'),legend_title_text='Reviews',legend_title_font=dict(size=14,color='white'),barmode='stack', )
            .update_xaxes(title='Minimum Night', type='category') ,use_container_width=True)

        b.markdown("<div style='border-left: 3px solid #FF385C;font-family:PhonePeSans; height: 400px;'></div>", unsafe_allow_html=True)

        price_max=pd.DataFrame(df0.groupby('maximum_nights').agg({'number_of_reviews':'count','price':'median'}).reset_index())

        c.plotly_chart(create_plotly_charts(price_max,'Bar','maximum_nights','price',color='number_of_reviews',color_continuous_scale='Inferno_r',)
        .update_traces(hovertemplate='<b>Region:</b> %{label}<br>' +
                      '<b>Average Price:</b> %{y:.2f}<br>' ,hoverlabel=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='white')),
                      visible=True,showlegend=False,textfont=dict(size=12, color='#ffffff'),textposition='outside',texttemplate='%{y:,.2f} ')
            .update_layout(coloraxis_colorbar=dict(title='Reviews'),width=400, height=450,xaxis={'categoryorder': 'total descending'},yaxis={'title': 'Price', 'categoryorder': 'total ascending'},   legend_font=dict(size=13,color='white'),legend=dict(bgcolor='rgba(0,0,0,0)'),legend_title_text='Reviews',legend_title_font=dict(size=14,color='white'),barmode='stack', )
            .update_xaxes(title='Maximum Night', type='category') ,use_container_width=True)

        st.markdown("<hr style='border: 2px solid #FF385C;'>", unsafe_allow_html=True)


        a,b,c=st.columns([.8,.0001,1])

        
        a.header('Top 15 hosts interms of price with Superhost breakup:')

        top_host=pd.DataFrame(df0.groupby(['host_id','superhost']).agg({'number_of_reviews':'count','price':'sum'}).reset_index()).nlargest(15,'price')

        a.plotly_chart(create_plotly_charts(top_host,'Bar','host_id','price',color='superhost',color_discrete_sequence=px.colors.qualitative.Alphabet)
        .update_traces(hovertemplate='<b>Host ID:</b> %{label}<br>' +
                      '<b>Average Price:</b> %{y:.2f}<br>' ,hoverlabel=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='white')),
                      visible=True,showlegend=False,textfont=dict(size=12, color='#ffffff'),textposition='outside',texttemplate='%{y:,.2f} ')
            .update_layout(width=400, height=450,xaxis={'categoryorder': 'total descending'},yaxis={'title': 'Price', 'categoryorder': 'total ascending'},   legend_font=dict(size=13,color='white'),legend=dict(bgcolor='rgba(0,0,0,0)'),legend_title_text='Super Host',legend_title_font=dict(size=14,color='white'),barmode='stack', )
            .update_xaxes(title='Host Id', type='category') ,use_container_width=True)

        
        
        b.markdown("<div style='border-left: 3px solid #FF385C;font-family:PhonePeSans; height: 600px;'></div>", unsafe_allow_html=True)
        
        
        c.header('Price difference in terms of Host Response Rate:')

        top_host=pd.DataFrame(df0.groupby(['host_response_rate','superhost']).agg({'number_of_reviews':'count','price':'sum'}).reset_index()).nlargest(15,'price')

        c.plotly_chart(create_plotly_charts(top_host,'Bar','host_response_rate','price',color='superhost',color_discrete_sequence=px.colors.qualitative.Alphabet)
        .update_traces(hovertemplate='<b>Host ID:</b> %{label}<br>' +
                      '<b>Average Price:</b> %{y:.2f}<br>' ,hoverlabel=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='white')),
                      visible=True,showlegend=True,textfont=dict(size=12, color='#ffffff'),textposition='outside',texttemplate='%{y:,.2f} ')
            .update_layout(width=400, height=450,xaxis={'categoryorder': 'total descending'},yaxis={'title': 'Price', 'categoryorder': 'total ascending'},   legend_font=dict(size=13,color='white'),legend=dict(bgcolor='rgba(0,0,0,0)'),legend_title_text='Super Host',legend_title_font=dict(size=14,color='white'),barmode='stack', )
            .update_xaxes(title='Host Response Rate', type='category') ,use_container_width=True)

        st.markdown("<hr style='border: 2px solid #FF385C;'>", unsafe_allow_html=True)

    with tab2:
        
        
        st.markdown("<hr style='border: 2px solid #FF385C;'>", unsafe_allow_html=True)


        st.header("**Availability in terms of Property Type and Room Type:**")

        a,b,c=st.columns([1,.0001,1])

        avail_prpty = df0.groupby('property_type')['annual_availability'].median().reset_index(name='availability').sort_values(by='availability',ascending=False)

        a.plotly_chart(create_plotly_charts(avail_prpty,'Bar','property_type','availability',text='availability',color_discrete_sequence=px.colors.qualitative.Dark2,color='property_type')
       .update_traces(hovertemplate='<b>Property Type:</b> %{label}<br>' +
                      '<b>Annual Availability:</b> %{y:.0f} - Days<br>' ,hoverlabel=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='white')),
                      visible=True,showlegend=False,textfont=dict(size=12, color='#ffffff'),textposition='outside',texttemplate='%{y:,.0f} - Days ')
            .update_layout(width=400, height=450,xaxis={'categoryorder': 'total descending'},yaxis={'categoryorder': 'total ascending'},   legend_font=dict(size=13,color='white'),legend=dict(bgcolor='rgba(0,0,0,0)'),legend_title_text='Property Type',legend_title_font=dict(size=14,color='white'))
            ,use_container_width=True)
        
        b.markdown("<div style='border-left: 3px solid #FF385C;font-family:PhonePeSans; height: 400px;'></div>", unsafe_allow_html=True)

        avail_room = df0.groupby('room_type')['annual_availability'].median().reset_index(name='annual_availability').sort_values(by='annual_availability',ascending=False)

        c.plotly_chart(create_plotly_charts(avail_room,'Bar','room_type','annual_availability',color_discrete_sequence=px.colors.qualitative.Dark2,color='room_type')
        .update_traces(hovertemplate='<b>Room Type:</b> %{label}<br>' +
                      '<b>Annual Availability:</b> %{y:.0f} - Days<br>' ,hoverlabel=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='white')),
                      visible=True,showlegend=False,textfont=dict(size=12, color='#ffffff'),textposition='outside',texttemplate='%{y:,.0f} - Days')
            .update_layout(width=400, height=450,xaxis={'categoryorder': 'total descending'},yaxis={'categoryorder': 'total ascending'},   legend_font=dict(size=13,color='white'),legend=dict(bgcolor='rgba(0,0,0,0)'),legend_title_text='Room Type',legend_title_font=dict(size=14,color='white'))
            ,use_container_width=True)
        
        st.markdown("<hr style='border: 2px solid #FF385C;'>", unsafe_allow_html=True)

        st.header("**Availability in terms of Cancellation Policy :**")

        avail_res_rate=df0.groupby('cancellation_policy')['annual_availability'].median().reset_index(name='annual_availability').sort_values(by='annual_availability',ascending=False)

        st.plotly_chart(create_plotly_charts(avail_res_rate,'Bar','cancellation_policy','annual_availability',color_discrete_sequence=px.colors.qualitative.Dark2,color='cancellation_policy')
        .update_traces(hovertemplate='<b>Cancellation Policy:</b> %{label}<br>' +
                      '<b>Annual Availability:</b> %{y:.0f} - Days<br>' ,hoverlabel=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='white')),
                      visible=True,showlegend=True,textfont=dict(size=12, color='#ffffff'),textposition='outside',texttemplate='%{y:,.0f} - Days')
            .update_layout(width=400, height=450,xaxis={'categoryorder': 'total descending'},yaxis={'categoryorder': 'total ascending'},   legend_font=dict(size=13,color='white'),legend=dict(bgcolor='rgba(0,0,0,0)'),legend_title_font=dict(size=14,color='white'))
            ,use_container_width=True)
        

        st.markdown("<hr style='border: 2px solid #FF385C;'>", unsafe_allow_html=True)

        st.header("**Availability in terms of Country and Region:**")

        a,b,c=st.columns([.80,.0001,1])

        avail_prpty = df0.groupby('country')['annual_availability'].median().reset_index(name='availability').sort_values(by='availability',ascending=False)

        a.plotly_chart(create_plotly_charts(avail_prpty,'Bar','country','availability',text='availability',color_discrete_sequence=px.colors.qualitative.Dark2,color='country')
       .update_traces(hovertemplate='<b>Contry:</b> %{label}<br>' +
                      '<b>Annual Availability:</b> %{y:.0f} - Days<br>' ,hoverlabel=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='white')),
                      visible=True,showlegend=False,textfont=dict(size=12, color='#ffffff'),textposition='outside',texttemplate='%{y:,.0f} - Days ')
            .update_layout(width=400, height=450,xaxis={'categoryorder': 'total descending'},yaxis={'categoryorder': 'total ascending'},   legend_font=dict(size=13,color='white'),legend=dict(bgcolor='rgba(0,0,0,0)'),legend_title_text='Property Type',legend_title_font=dict(size=14,color='white'))
            ,use_container_width=True)
        
        b.markdown("<div style='border-left: 3px solid #FF385C;font-family:PhonePeSans; height: 400px;'></div>", unsafe_allow_html=True)

        avail_room = df0.groupby('region')['annual_availability'].median().reset_index(name='annual_availability').sort_values(by='annual_availability',ascending=False)

        c.plotly_chart(create_plotly_charts(avail_room,'Bar','region','annual_availability',color_discrete_sequence=px.colors.qualitative.Dark2,color='region')
        .update_traces(hovertemplate='<b>Region:</b> %{label}<br>' +
                      '<b>Annual Availability:</b> %{y:.0f} - Days<br>' ,hoverlabel=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='white')),
                      visible=True,showlegend=False,textfont=dict(size=12, color='#ffffff'),textposition='outside',texttemplate='%{y:,.0f} - Days')
            .update_layout(width=400, height=450,xaxis={'categoryorder': 'total descending'},yaxis={'categoryorder': 'total ascending'},   legend_font=dict(size=13,color='white'),legend=dict(bgcolor='rgba(0,0,0,0)'),legend_title_text='Room Type',legend_title_font=dict(size=14,color='white'))
            ,use_container_width=True)


    with tab3:
        
        st.header("**Listing counts  and Price in  terms of Country :**")

        a,b,c=st.columns([1,.0001,1])

        avail_prpty = df0.groupby('country')['_id'].count().reset_index(name='Count').sort_values(by='Count',ascending=False)

        a.plotly_chart(create_plotly_charts(avail_prpty,'Bar','country','Count',text='Count',color_discrete_sequence=px.colors.qualitative.Dark2,color='country')
       .update_traces(hovertemplate='<b>Contry:</b> %{label}<br>' +
                      '<b>Count:</b> %{y} <br>' ,hoverlabel=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='white')),
                      visible=True,showlegend=False,textfont=dict(size=12, color='#ffffff'),textposition='outside',texttemplate='%{y} ')
            .update_layout(width=400, height=450,xaxis={'categoryorder': 'total descending'},yaxis={'categoryorder': 'total ascending'},   legend_font=dict(size=13,color='white'),legend=dict(bgcolor='rgba(0,0,0,0)'),legend_title_text='Property Type',legend_title_font=dict(size=14,color='white'))
            ,use_container_width=True)
        
        b.markdown("<div style='border-left: 3px solid #FF385C;font-family:PhonePeSans; height: 400px;'></div>", unsafe_allow_html=True)

        avail_room = df0.groupby('country')['price'].median().reset_index(name='Avg_Price').sort_values(by='Avg_Price',ascending=False)

        c.plotly_chart(create_plotly_charts(avail_room,'Bar','country','Avg_Price',color_discrete_sequence=px.colors.qualitative.Dark2,color='country')
        .update_traces(hovertemplate='<b>Country:</b> %{label}<br>' +
                      '<b>Avg. Price:</b> %{y:.0f} <br>' ,hoverlabel=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='white')),
                      visible=True,showlegend=False,textfont=dict(size=12, color='#ffffff'),textposition='outside',texttemplate='%{y:,.0f} ')
            .update_layout(width=400, height=450,xaxis={'categoryorder': 'total descending'},yaxis={'categoryorder': 'total ascending'},   legend_font=dict(size=13,color='white'),legend=dict(bgcolor='rgba(0,0,0,0)'),legend_title_text='Room Type',legend_title_font=dict(size=14,color='white'))
            ,use_container_width=True)

        st.markdown("<hr style='border: 2px solid #FF385C;'>", unsafe_allow_html=True)

        st.header("**Listing counts  and Price in  terms of Region :**")

        a,b,c=st.columns([1,.0001,1])

        avail_prpty = df0.groupby('region')['_id'].count().reset_index(name='Count').sort_values(by='Count',ascending=False)

        a.plotly_chart(create_plotly_charts(avail_prpty,'Bar','region','Count',text='Count',color_discrete_sequence=px.colors.qualitative.Dark2,color='region')
       .update_traces(hovertemplate='<b>Region:</b> %{label}<br>' +
                      '<b>Count:</b> %{y} <br>' ,hoverlabel=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='white')),
                      visible=True,showlegend=False,textfont=dict(size=12, color='#ffffff'),textposition='outside',texttemplate='%{y} ')
            .update_layout(width=400, height=450,xaxis={'categoryorder': 'total descending'},yaxis={'categoryorder': 'total ascending'},   legend_font=dict(size=13,color='white'),legend=dict(bgcolor='rgba(0,0,0,0)'),legend_title_text='Property Type',legend_title_font=dict(size=14,color='white'))
            ,use_container_width=True)
        
        b.markdown("<div style='border-left: 3px solid #FF385C;font-family:PhonePeSans; height: 400px;'></div>", unsafe_allow_html=True)

        avail_room = df0.groupby('region')['price'].median().reset_index(name='Avg_Price').sort_values(by='Avg_Price',ascending=False)

        c.plotly_chart(create_plotly_charts(avail_room,'Bar','region','Avg_Price',color_discrete_sequence=px.colors.qualitative.Dark2,color='region')
        .update_traces(hovertemplate='<b>Region:</b> %{label}<br>' +
                      '<b>Avg. Price:</b> %{y:.0f} <br>' ,hoverlabel=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='white')),
                      visible=True,showlegend=False,textfont=dict(size=12, color='#ffffff'),textposition='outside',texttemplate='%{y:,.0f} ')
            .update_layout(width=400, height=450,xaxis={'categoryorder': 'total descending'},yaxis={'categoryorder': 'total ascending'},   legend_font=dict(size=13,color='white'),legend=dict(bgcolor='rgba(0,0,0,0)'),legend_title_text='Room Type',legend_title_font=dict(size=14,color='white'))
            ,use_container_width=True)
        
        st.markdown("<hr style='border: 2px solid #FF385C;'>", unsafe_allow_html=True)
        
        st.header("**Cancellation Policy  in terms of  Region :**")

        avail_prpty = df0.groupby(['region','cancellation_policy'])['_id'].count().reset_index(name='Count').sort_values(by='Count',ascending=False)

        st.plotly_chart(create_plotly_charts(avail_prpty,'Bar','region','Count',text='Count',color_discrete_sequence=px.colors.qualitative.Dark2,color='cancellation_policy')
       .update_traces(hovertemplate='<b>Region:</b> %{label}<br>' +
                      '<b>Count:</b> %{y} <br>' ,hoverlabel=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='white')),
                      visible=True,showlegend=True,textfont=dict(size=12, color='#ffffff'),textposition='outside',texttemplate='%{y} ')
            .update_layout(width=400, height=450,xaxis={'categoryorder': 'total descending'},yaxis={'categoryorder': 'total ascending'},   legend_font=dict(size=13,color='white'),legend=dict(bgcolor='rgba(0,0,0,0)'),legend_title_text='Property Type',legend_title_font=dict(size=14,color='white'))
            ,use_container_width=True)
        
        st.markdown("<hr style='border: 2px solid #FF385C;'>", unsafe_allow_html=True)

        st.header("**Cancellation Policy  in terms of  Country :**")

        avail_room = df0.groupby(['country','cancellation_policy'])['_id'].count().reset_index(name='Count').sort_values(by='Count',ascending=False)

        st.plotly_chart(create_plotly_charts(avail_room,'Bar','country','Count',color_discrete_sequence=px.colors.qualitative.Dark2,color='cancellation_policy')
        .update_traces(hovertemplate='<b>Country:</b> %{label}<br>' +
                      '<b>Count:</b> %{y:.0f} <br>' ,hoverlabel=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='white')),
                      visible=True,showlegend=True,textfont=dict(size=12, color='#ffffff'),textposition='outside',texttemplate='%{y:,.0f} ')
            .update_layout(width=400, height=450,xaxis={'categoryorder': 'total descending'},yaxis={'categoryorder': 'total ascending'},   legend_font=dict(size=13,color='white'),legend=dict(bgcolor='rgba(0,0,0,0)'),legend_title_text='Room Type',legend_title_font=dict(size=14,color='white'))
            ,use_container_width=True)
        
        st.markdown("<hr style='border: 2px solid #FF385C;'>", unsafe_allow_html=True)
        
        st.header("**Number of Reviews  in terms of  Country & Region :**")
        a,b,c=st.columns([1,.0001,1])
        
        avail_prpty = df0.groupby('region')['number_of_reviews'].sum().reset_index(name='Reviews_count').sort_values(by='Reviews_count',ascending=False)

        a.plotly_chart(create_plotly_charts(avail_prpty,'Bar','region','Reviews_count',text='Reviews_count',color_discrete_sequence=px.colors.qualitative.Dark2,color='region')
       .update_traces(hovertemplate='<b>Region:</b> %{label}<br>' +
                      '<b>Number of Reviews:</b> %{y} <br>' ,hoverlabel=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='white')),
                      visible=True,showlegend=False,textfont=dict(size=12, color='#ffffff'),textposition='outside',texttemplate='%{y} ')
            .update_layout(width=400, height=450,xaxis={'categoryorder': 'total descending'},yaxis={'categoryorder': 'total ascending'},   legend_font=dict(size=13,color='white'),legend=dict(bgcolor='rgba(0,0,0,0)'),legend_title_text='Property Type',legend_title_font=dict(size=14,color='white'))
            ,use_container_width=True)
        
        b.markdown("<div style='border-left: 3px solid #FF385C;font-family:PhonePeSans; height: 400px;'></div>", unsafe_allow_html=True)

        avail_room = df0.groupby('country')['number_of_reviews'].sum().reset_index(name='Reviews_count').sort_values(by='Reviews_count',ascending=False)

        c.plotly_chart(create_plotly_charts(avail_room,'Bar','country','Reviews_count',text='Reviews_count',color_discrete_sequence=px.colors.qualitative.Dark2,color='country')
       .update_traces(hovertemplate='<b>Country:</b> %{label}<br>' +
                      '<b>Number of Reviews:</b> %{y} <br>' ,hoverlabel=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='white')),
                      visible=True,showlegend=False,textfont=dict(size=12, color='#ffffff'),textposition='outside',texttemplate='%{y} ')
            .update_layout(width=400, height=450,xaxis={'categoryorder': 'total descending'},yaxis={'categoryorder': 'total ascending'},   legend_font=dict(size=13,color='white'),legend=dict(bgcolor='rgba(0,0,0,0)'),legend_title_text='Property Type',legend_title_font=dict(size=14,color='white'))
            ,use_container_width=True)
        st.markdown("<hr style='border: 2px solid #FF385C;'>", unsafe_allow_html=True)


    with tab4:

        st.header("**Listings Density across globe:**")
        st.info('Use filters in the sidebar to narrow down to Coutry and or Region')
        if country !='All' and region=='All':
            zoom_level = 8.5
            cen_lat=df0[df0['country']==country]['latitude'].mean()
            cen_lon=df0[df0['country']==country]['longitude'].mean()
            rad=5
            if region =='Hawaii':
                zoom_level = 6
            elif country =='United States':
                zoom_level = 2.5
        elif country !='All' and region!='All':
             zoom_level = 11
             cen_lat=df0[df0['region']==region]['latitude'].mean()
             cen_lon=df0[df0['region']==region]['longitude'].mean()
             rad=5
             if region =='Hawaii':
                 zoom_level = 6
             elif country == 'United States':
                zoom_level = 2.5
        else:
            zoom_level=1
            cen_lat=0
            cen_lon=0
            rad=7
                # dff0=df0.groupby('country').agg({'_id':'count','number_of_reviews':'sum','annual_availability':'sum','price':'sum','latitude':np.mean,'longitude':np.mean}).reset_index()

        df01=df0.groupby(['country','region']).agg({'_id':'count','number_of_reviews':'sum','annual_availability':'sum','price':'sum','latitude':np.mean,'longitude':np.mean}).reset_index()
        listings_den = px.density_mapbox(df0, lat='latitude', lon='longitude', z='_id', opacity=.7, color_continuous_scale='Magma', 
                                            mapbox_style="carto-positron", radius=rad, hover_data={"latitude": False, "longitude": False,  "country": True, 'region':True,'annual_availability':True}, 
                                            hover_name='region', center=dict(lat=cen_lat, lon=cen_lon), zoom=zoom_level)
        listings_den   .update_layout( mapbox_zoom=zoom_level,  geo=dict(scope='asia', projection_type='equirectangular'), mapbox_center={"lat": cen_lat , "lon":cen_lon}, margin={"r": 0, "t": 0, "l": 0, "b": 0}, width=800, height=550)
        listings_den.update_traces(hovertemplate='<b>%{hovertext}</b><br>Count: %{z:,.2f} <br>Country: %{customdata[3]} <br>Annual Availability - %{customdata[4]:,.20}')
        
        st.plotly_chart(listings_den, use_container_width=True)
        
        st.markdown("<hr style='border: 2px solid #FF385C;'>", unsafe_allow_html=True)

        st.header("**Listings Price Density across globe:**")

        if country !='All' and region=='All':
            zoom_level = 8.5
            cen_lat=df0[df0['country']==country]['latitude'].mean()
            cen_lon=df0[df0['country']==country]['longitude'].mean()
            rad=10
            if region =='Hawaii':
                zoom_level = 6
            elif country =='United States':
                zoom_level = 2.5
        elif country !='All' and region!='All':
             zoom_level = 11
             cen_lat=df0[df0['region']==region]['latitude'].mean()
             cen_lon=df0[df0['region']==region]['longitude'].mean()
             rad=10
             if region =='Hawaii':
                 zoom_level = 6
             elif country == 'United States':
                zoom_level = 2.5
        else:
            zoom_level=1
            cen_lat=0
            cen_lon=0
            rad=10
        dff1=df0.groupby(['country','region']).agg({'_id':'count','number_of_reviews':'sum','annual_availability':'sum','price':'sum','latitude':np.mean,'longitude':np.mean}).reset_index()
        listings_price_den = px.density_mapbox(dff1, lat='latitude', lon='longitude', z='price', opacity=1, color_continuous_scale='Magma', 
                                            mapbox_style="carto-positron", radius=rad, hover_data={"latitude": False, "longitude": False, "region": True, "country": True, 'annual_availability':True,'number_of_reviews':True}, 
                                            hover_name='region', center=dict(lat=cen_lat, lon=cen_lon), zoom=zoom_level)
        listings_price_den   .update_layout( mapbox_zoom=zoom_level,  geo=dict(scope='asia', projection_type='equirectangular'), mapbox_center={"lat": cen_lat , "lon":cen_lon}, margin={"r": 0, "t": 0, "l": 0, "b": 0}, width=800, height=550)
        listings_price_den.update_traces(hovertemplate='<b>%{hovertext}</b><br>Price: %{z:,.2f} <br>Region: %{customdata[2]} <br>Availability  - %{customdata[4]}<br>Reviews Count - %{customdata[5]}')

        st.plotly_chart(listings_price_den, use_container_width=True)

        st.markdown("<hr style='border: 2px solid #FF385C;'>", unsafe_allow_html=True)

        st.header("**Listings Reviews Density across globe:**")

        dff0=df0.groupby('country').agg({'_id':'count','number_of_reviews':'sum','host_total_listings_count':'sum','annual_availability':'sum','price':'sum','latitude':np.mean,'longitude':np.mean}).reset_index()
        aa = px.scatter_geo(dff0, lat='latitude', lon='longitude', size='number_of_reviews',color_discrete_sequence=px.colors.sequential.Viridis,
                                             text='country',size_max=25, hover_data={ 'country':True,"price": True, "number_of_reviews": True},
                                           projection='equirectangular',color_continuous_scale='Magma',  ).update_traces(textfont_color='#000000') .update_layout(width=1000, height=800,geo=dict(center=dict(lat=cen_lat, lon=cen_lon)))
    
        st.plotly_chart(aa, use_container_width=True)
        
        def mode_func(x):
            value_counts = x.value_counts()
            mode_value = value_counts.idxmax()
    
            return mode_value
        
        st.markdown("<hr style='border: 2px solid #FF385C;'>", unsafe_allow_html=True)
        st.header("**Listings Availability  Density across globe:**")
        # dff0=df0.groupby('country').agg({'_id':'count','number_of_reviews':'sum','annual_availability':'sum','price':'sum','latitude':np.mean,'longitude':np.mean}).reset_index()
    
        aa = px.scatter_geo(dff0, lat='latitude', lon='longitude', size='annual_availability',color_continuous_scale="Magma",color_discrete_sequence=px.colors.sequential.Magma,
                                             text='country',size_max=25, hover_data={ 'longitude':False,'latitude':False,'country':True,"price": True, "annual_availability": True},
                                           projection='equirectangular', ).update_traces(textfont_color='#000000') .update_layout(width=1000, height=800,geo=dict(center=dict(lat=cen_lat, lon=cen_lon)))
    
        st.plotly_chart(aa, use_container_width=True)
        st.markdown("<hr style='border: 2px solid #FF385C;'>", unsafe_allow_html=True)

        st.header("**Listings Availability  Density across globe:**")

    
        aa = px.scatter_geo(dff0, lat='latitude', lon='longitude', size='host_total_listings_count',color_continuous_scale="Electric",color_discrete_sequence=px.colors.sequential.Electric,
                                             text='country',size_max=25, hover_data={'longitude':False,'latitude':False, 'country':True,"price": True, "annual_availability": True,'host_total_listings_count':True},
                                           projection='equirectangular', ).update_traces(textfont_color='#000000') .update_layout(width=1000, height=800,geo=dict(center=dict(lat=cen_lat, lon=cen_lon)))
    
        st.plotly_chart(aa, use_container_width=True)
        st.markdown("<hr style='border: 2px solid #FF385C;'>", unsafe_allow_html=True)

elif selected == 'EDA':

    with open('Airbnb.html', 'r',encoding='utf-8') as file:
        html_content = file.read()

    st.components.v1.html(html_content,width=1000, height=56330)
