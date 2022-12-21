import streamlit as st
import pandas as pd
import numpy as np

st.title('Brain Network')

@st.cache
def loadData():
    matrix = pd.read_csv('matrix.csv', index_col = 0)
#     colorlist = np.array(dataiku.Dataset("colorlist").get_dataframe().set_index('col_0')['col_1'])
#     colornumbs = np.array(dataiku.Dataset("colornumbs").get_dataframe().set_index('col_0')['col_1'])
#     lineList = np.array(dataiku.Dataset("lineList").get_dataframe().set_index('col_0')['col_1'])
#     sublist = np.array(dataiku.Dataset("sublist").get_dataframe().set_index('col_0')['col_1']) 
    
#     matrix.columns = lineList
#     matrix.index = lineList
    
    return matrix#, colorlist, colornumbs, lineList, sublist


