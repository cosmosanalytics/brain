import streamlit as st
import pandas as pd
import numpy as np

st.title('Brain Network')

@st.cache
def loadData():
    matrix = pd.read_csv('matrix.csv', index_col = 0)
    colorlist = np.array(pd.read_csv('colorlist', index_col = 0)['col_1'])
    colornumbs = np.array(pd.read_csv('colornumbs', index_col = 0)['col_1'])
    lineList = np.array(pd.read_csv('lineList', index_col = 0)['col_1'])
    sublist = np.array(pd.read_csv('sublist', index_col = 0)['col_1']) 
    
    matrix.columns = lineList
    matrix.index = lineList
    return matrix, colorlist, colornumbs, lineList, sublist

matrix, colorlist, colornumbs, lineList, sublist = loadData()
st.write(matrix)

