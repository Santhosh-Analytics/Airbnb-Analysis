import pandas as pd
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
        "container": {"padding": "0!important", "background-color": "#242a44"},
        "icon": {"color": "rgb(235, 48, 84)", "font-size": "25px","font-family":"inherit"}, 
        "nav-link": {"font-family":"inherit","font-size": "22px", "color": "#ffffff","text-align": "left", "margin":"0px", "--hover-color": "#84706E"},
        "nav-link-selected": {"font-family":"inherit","background-color": "#ffffff  ","color": "#FF385C","font-size": "25px"},
    })
    st.markdown("<hr style='border: 2px solid #FF385C;'>", unsafe_allow_html=True)


st.markdown(""" <style> button[data-baseweb="tab"] > di v[data-testid="stMarkdownContainer"] > p {font-size: 28px;} </style>""", unsafe_allow_html=True)



st.markdown("<h1 style='text-align: center; font-size: 38px; color: #FF385C ; font-weight: 700;font-family:inherit;'>Airbnb Listings Data Analysis</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border: 2px solid #FF385C;'>", unsafe_allow_html=True)

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
       

def filter_df(df,col,filter):
    if filter != 'All':
        df=df[col == filter].reset_index(drop= True)
    else:
        df=df.copy()
    return df
df=pd.read_csv("final_df.csv")

if selected == "Data Exploration":
    st.markdown('<style>div.css-1jpvgo6 {font-size: 16px; font-weight: bolder;font-family:PhonePeSans; } </style>', unsafe_allow_html=True)

    tab, tab1, tab2, tab3, tab4, tab5= st.tabs(["***LISTINGS COUNT***","***PRICE ANALYSIS***","***AVAILABILITY ANALYSIS***","***LOCATION BASED***", "***GEOSPATIAL VISUALIZATION***", "***TOP CHARTS***"])
    with tab:
        st.title("**Listing count in terms of cancellation policy and room type**")
        country= st.selectbox("Select the Country",["All"] + list(map(str, df["country"].unique())))

        df0=filter_df(df,df.country,country)

        fig, axs = plt.subplots(1, 2, figsize=(16, 5))
        plt.subplot(121)
        sns.histplot(data=df0,x='price',bins=30,kde=True,hue='room_type')
        plt.subplot(122)
        sns.histplot(data=df0,x='price',kde=True,hue='cancellation_policy')
        st.pyplot(fig,use_container_width=False)

        cancellation_order = df0['cancellation_policy'].value_counts().index

        # Create subplots
        fig, axs = plt.subplots(1, 2, figsize=(16, 4), gridspec_kw={'width_ratios': [6, 10]})   

        # First subplot
        axs[0].set_title('Listing Count by Cancellation Policy')
        sns.countplot(x=df0['cancellation_policy'], hue=df0.cancellation_policy, order=cancellation_order, ax=axs[0])
        axs[0].tick_params(axis='x', rotation=10)  # Rotate x-axis labels

        # Second subplot
        axs[1].set_title('Listing count by Cancellation Policy and Room Type')
        sns.countplot(x=df0['cancellation_policy'], hue=df0.room_type, order=cancellation_order, ax=axs[1])
        axs[1].tick_params(axis='x', rotation=10)  # Rotate x-axis labels
        st.pyplot(fig,use_container_width=False)

        st.header("**Listings Count by room type and country**")

        fig, axs = plt.subplots(1, 2, figsize=(16, 4), gridspec_kw={'width_ratios': [6, 10]})
        axs[0].set_title('Listing Count by Room Type')
        sns.countplot(x=df0['room_type'],hue=df0.room_type,stat="percent", ax=axs[0])
        sns.countplot(x=df0['room_type'],hue=df0.country,stat="percent", ax=axs[1])
        axs[1].set_title('Listings Count by room type and country')
        st.pyplot(fig,use_container_width=False)


    with tab1:
        st.title("**Average price in terms of Cancellation policy and Accomodates**")
        col1,col2=st.columns(2)
        country= col1.selectbox("Select the Country",["All"] + list(map(str, df["country"].unique())),key=1)
        room= col2.selectbox("Select the Roon Type",["All"] + list(map(str, df["room_type"].unique())),key=0)

        df=filter_df(df,df.country,country)
        df=filter_df(df,df.room_type,room)

        


        df1 = df.groupby(['cancellation_policy'])['price'].mean().reset_index()
        df2 = df.groupby(['accommodates'])['price'].mean().reset_index()

        df1=df1.sort_values(by='price',ascending=False)
        df2=df2.sort_values(by='price',ascending=False)


        fig, axs = plt.subplots(1, 2, figsize=(16,4), gridspec_kw={'width_ratios': [8, 10]})

        sns.barplot(data=df1,x='cancellation_policy',y='price',hue='cancellation_policy',ax=axs[0])        
        sns.barplot(data=df2, x="accommodates", y="price", ax=axs[1],hue='accommodates',order=df2['accommodates'])

        plt.tight_layout()
        st.pyplot(fig)

        st.title("**Average price, Security Deposit, Cleasing Fee for each room type**")
        col1,col2=st.columns(2)
        country= col1.selectbox("Select the Country",["All"] + list(map(str, df["country"].unique())),key=2)

        df=filter_df(df,df.country,country)
    
        


        df1 = df.groupby(['cancellation_policy'])['price'].mean().reset_index()
        df2 = df.groupby(['accommodates'])['price'].mean().reset_index()

        df1=df1.sort_values(by='price',ascending=False)
        df2=df2.sort_values(by='price',ascending=False)


        fig, axs = plt.subplots(1, 2, figsize=(16,4), gridspec_kw={'width_ratios': [8, 10]})

        sns.barplot(data=df1,x='cancellation_policy',y='price',hue='cancellation_policy',ax=axs[0])        
        sns.barplot(data=df2, x="accommodates", y="price", ax=axs[1],hue='accommodates',order=df2['accommodates'])

        plt.tight_layout()
        st.pyplot(fig)