import streamlit as st
import pandas as pd
import numpy as np

st.title('Brain Network')

@st.cache
def loadData():
    matrix = pd.read_csv('matrix.csv', index_col = 0)
    colorlist = np.array(pd.read_csv('colorlist.csv', index_col = 0)[1])
    colornumbs = np.array(pd.read_csv('colornumbs.csv', index_col = 0)[1])
    lineList = np.array(pd.read_csv('lineList.csv', index_col = 0)[1])
    sublist = np.array(pd.read_csv('sublist.csv', index_col = 0)[1]) 
    
    matrix.columns = lineList
    matrix.index = lineList
    return matrix, colorlist, colornumbs, lineList, sublist

matrix, colorlist, colornumbs, lineList, sublist = loadData()
st.write(matrix)

