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

def defineG(matrix, threshold, Nodes, Links):
    matrix = abs(matrix); matrix[matrix<=threshold] = 0    
    matrix[matrix.index.isin(Nodes)] = 0 ; matrix[matrix.columns.isin(Nodes)] = 0
    for i in Links: matrix.loc[i]=0   
    print(matrix)

    G = nx.from_numpy_matrix(np.array(matrix))
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    return G

matrix, colorlist, colornumbs, lineList, sublist = loadData()
Nodes = st.multiselect('Select Node(s)', lineList)
Links = st.multiselect('Select Link(s)', list(permutations(lineList, 2)))
threshold = st.slider('Threshold', 0, 1, 0, step=0.1)





