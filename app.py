import streamlit as st
import pandas as pd
import numpy as np
from itertools import permutations 
import networkx as nx
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
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

# def centrality_calc(G, lineList):
#     G_distance_dict = {(e1, e2): 1 / abs(weight) for e1, e2, weight in G.edges(data='weight')}
#     nx.set_edge_attributes(G, G_distance_dict, 'distance')
#     closeness = pd.Series(nx.closeness_centrality(G, distance='distance')); closeness.index = lineList
#     betweenness = pd.Series(nx.betweenness_centrality(G, weight='distance', normalized=True)); betweenness.index = lineList 
#     eigen = pd.Series(nx.eigenvector_centrality(G, weight='weight')); eigen.index = lineList 
#     pagerank = pd.Series(nx.pagerank(G, weight='weight')); pagerank.index = lineList 
#     clustering = pd.Series(nx.clustering(G, weight='weight')); clustering.index = lineList 
#     mean_clutering = nx.average_clustering(G, weight='weight') 
#     return closeness, betweenness, eigen, pagerank, clustering, mean_clutering

def brainNX(G, colorlist, colornumbs, lineList, sublist):
    strength = G.degree(weight='weight')
    strengths = {node: val for (node, val) in strength}
    nx.set_node_attributes(G, dict(strength), 'strength') # Add as nodal attribute
    normstrenghts = {node: val * 1/(len(G.nodes)-1) for (node, val) in strength}
    nx.set_node_attributes(G, normstrenghts, 'strengthnorm') # Add as nodal attribute
    normstrengthlist = np.array([val * 1/(len(G.nodes)-1) for (node, val) in strength])    
    
    def Convert(lst): 
        res_dct = {i : lst[i] for i in range(0, len(lst))} 
        return res_dct

    nx.set_node_attributes(G, Convert(lineList), 'area')
    nx.set_node_attributes(G, Convert(colorlist), 'color')
    nx.set_node_attributes(G, Convert(sublist), 'subnet')
    nx.set_node_attributes(G, Convert(colornumbs), 'colornumb')

    fig, ax = plt.subplots(figsize=(20,20))
    edgewidth = [ d['weight'] for (u,v,d) in G.edges(data=True)]
    pos = nx.spring_layout(G, scale=5)
    nx.draw(G, pos, with_labels=True, width=np.power(edgewidth, 2), edge_color='grey', node_size=normstrengthlist*20000, 
            labels=Convert(lineList), font_color='black', node_color=colornumbs/10, cmap=plt.cm.Spectral, alpha=0.7, font_size=9)
    st.pyplot(fig)
    
    fig, ax = plt.subplots(figsize=(20,3))
    G1 = nx.from_pandas_edgelist(pd.DataFrame({'lineList':lineList, 'sublist':sublist}), 'lineList', 'sublist')
    pos = nx.circular_layout(G1)
    nx.draw(G1, pos)#, with_labels=True, node_size=2000)#, labels=Convert(lineList), font_color='black', node_color=colornumbs/10, cmap=plt.cm.Spectral, alpha=0.7, font_size=9)
    st.pyplot(fig)

col1, col2 = st.columns(2)
with col1:
    matrix, colorlist, colornumbs, lineList, sublist = loadData()
    Nodes = st.multiselect('Select Node(s)', lineList)
    Links = st.multiselect('Select Link(s)', list(permutations(lineList, 2)))
    threshold = st.slider('Threshold', 0.0, 1.0, 0.0)
    G = defineG(matrix, threshold, Nodes, Links)
#     closeness, betweenness, eigen, pagerank, clustering, mean_clutering = centrality_calc(G,lineList)   
#     fig, ax = plt.subplots(figsize=(20, 3)); ax = closeness.sort_values(ascending=False).plot.bar(); ax.set_title('Closeness'); st.pyplot(fig)  
#     fig, ax = plt.subplots(figsize=(20, 3)); ax = betweenness.sort_values(ascending=False).plot.bar(); ax.set_title('Betweenness'); st.pyplot(fig) 
#     fig, ax = plt.subplots(figsize=(20, 3)); ax = eigen.sort_values(ascending=False).plot.bar(); ax.set_title('Eigen'); st.pyplot(fig)
#     fig, ax = plt.subplots(figsize=(20, 3)); ax = pagerank.sort_values(ascending=False).plot.bar(); ax.set_title('Pagerank'); st.pyplot(fig) 
with col2:   
#     fig, ax = plt.subplots(figsize=(20, 3)); ax = clustering.sort_values(ascending=False).plot.bar(); ax.set_title('Clustering, average='+str(mean_clutering)); st.pyplot(fig) 
    brainNX(G, colorlist, colornumbs, lineList, sublist)







