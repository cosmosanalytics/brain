import streamlit as st
import pandas as pd
import numpy as np
from itertools import permutations 

st.title('Brain Network')

@st.cache
def loadData():
    matrix = pd.read_csv('matrix.csv', index_col = 0)
    colorlist = np.array(pd.read_csv('colorlist.csv', index_col = 0)['0'])
    colornumbs = np.array(pd.read_csv('colornumbs.csv', index_col = 0)['0'])
    lineList = np.array(pd.read_csv('lineList.csv', index_col = 0)['0'])
    sublist = np.array(pd.read_csv('sublist.csv', index_col = 0)['0']) 
    
    matrix.columns = lineList
    matrix.index = lineList
    return matrix, colorlist, colornumbs, lineList, sublist

matrix, colorlist, colornumbs, lineList, sublist = loadData()
Nodes = st.multiselect('Select Node(s)', lineList)
Links = st.multiselect('Select Link(s)', list(permutations(lineList, 2)))
st.write(matrix)
st.write(colorlist)

