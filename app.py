import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from itertools import permutations 
import networkx as nx
import matplotlib.pyplot as plt
import scipy
import scipy.cluster.hierarchy as sch

st.set_page_config(layout="wide")
st.title('Brain Network')

def plot_corr(corr):
    fig, ax = plt.subplots(figsize=(40,40))
    cax = ax.matshow(corr, cmap='Blues')
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90);
    plt.yticks(range(len(corr.columns)), corr.columns);
    cbar = fig.colorbar(cax, ticks=[-1, 0, 1], aspect=40, shrink=.8)
    st.pyplot(fig)  
    
@st.cache
def loadData():
    matrix = pd.read_csv('matrix.csv', index_col = 0)
    colorlist = pd.read_csv('colorlist.csv', index_col = 0)['0']
    colornumbs = pd.read_csv('colornumbs.csv', index_col = 0)['0']
    lineList = pd.read_csv('lineList.csv', index_col = 0)['0']
    sublist = pd.read_csv('sublist.csv', index_col = 0)['0'] 
    refDF = pd.DataFrame({'colorlist':colorlist, 'lineList':lineList, 'sublist':sublist})

    matrix.columns = lineList
    matrix.index = lineList
    return matrix, np.array(colorlist), np.array(colornumbs), np.array(lineList), np.array(sublist), refDF

def defineG(matrix0, threshold, Regions_Nodes, Nodes, Links):
    matrix = abs(matrix0); matrix[matrix<=threshold] = 0  
    matrix[matrix.index.isin(Regions_Nodes)] = 0 ; matrix[matrix.columns[matrix.columns.isin(Regions_Nodes)]] = 0
    matrix[matrix.index.isin(Nodes)] = 0 ; matrix[matrix.columns[matrix.columns.isin(Nodes)]] = 0
    for i in Links: 
        matrix.loc[i]=0   
    if st.checkbox('Show matrix'):
        st.write(matrix)
    G = nx.from_numpy_matrix(np.array(matrix))
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    return G, matrix

def centrality_calc(G, lineList):
    G_distance_dict = {(e1, e2): 1 / abs(weight) for e1, e2, weight in G.edges(data='weight')}
    nx.set_edge_attributes(G, G_distance_dict, 'distance')
    closeness = pd.Series(nx.closeness_centrality(G, distance='distance')); closeness.index = lineList
    betweenness = pd.Series(nx.betweenness_centrality(G, weight='distance', normalized=True)); betweenness.index = lineList 
    clustering = pd.Series(nx.clustering(G, weight='weight')); clustering.index = lineList 
    mean_clutering = nx.average_clustering(G, weight='weight')      
    return closeness, betweenness, clustering, mean_clutering

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

matrix, colorlist, colornumbs, lineList, sublist, refDF = loadData()    
col1, col2 = st.columns(2)
with col1:
    Regions = st.multiselect('Select Region(s) to Remove', set(sublist))
    Regions_Nodes = refDF[refDF['sublist'].isin(Regions)]['lineList'].values
    Nodes = st.multiselect('Select Node(s) to Remove', lineList)
    Links = st.multiselect('Select Link(s) to Remove', list(permutations(lineList, 2)))
    threshold = st.slider('Threshold to Filter', 0.0, 1.0, 0.0)
    G, matrix1 = defineG(matrix, threshold, Regions_Nodes, Nodes, Links)
    closeness, betweenness, clustering, mean_clutering = centrality_calc(G,lineList) 
    
    tab1, tab2 = st.tabs(["Bar Chart", "Distribution Chart"])
    with tab1:
        fig, axes = plt.subplots(3, 1, figsize=(20, 15)); 
        closeness.plot.bar(color=refDF['colorlist'], ax=axes[0]); axes[0].set_title('Closeness');  
        betweenness.plot.bar(color=refDF['colorlist'], ax=axes[1]); axes[1].set_title('Betweenness'); 
        clustering.plot.bar(color=refDF['colorlist'], ax=axes[2]); axes[2].set_title('Clustering, average='+str(mean_clutering)); 
        st.pyplot(fig)     
    with tab2:
        fig, axes = plt.subplots(3, 1, figsize=(20, 15)); 
        sns.distplot(closeness, kde=False, norm_hist=False, ax=axes[0]); axes[0].set_xlabel('Closeness'); axes[0].set_ylabel('Counts')
        sns.distplot(betweenness, kde=False, norm_hist=False, ax=axes[1]); axes[1].set_xlabel('Betweenness'); axes[1].set_ylabel('Counts')
        sns.distplot(clustering, kde=False, norm_hist=False, ax=axes[2]); axes[2].set_xlabel('Clustering Coefficient'); axes[2].set_ylabel('Counts'); 
        axes[2].set_title('average path length is '+str(round(nx.average_shortest_path_length(G, weight='distance'),2)))
        st.pyplot(fig)            
with col2: 
    def color_colorlist(val):
        color = val
        return f'background-color: {color}'
    refDF_agg = refDF.groupby(['sublist','colorlist'])['lineList'].apply(lambda x: ','.join(x)).reset_index()
    st.dataframe(refDF_agg.style.applymap(color_colorlist, subset=['colorlist']),use_container_width=True)     
    tab1, tab2, tab3 = st.tabs(["Network", "Clustered CorrCoef Matrix", "Left/Right CorrCoef Matrix"])
    
    matrix_order = matrix.abs().copy()
    X = matrix_order.values
    d = sch.distance.pdist(X)   
    L = sch.linkage(d, method='complete')
    ind = sch.fcluster(L, 0.5*d.max(), 'distance')    
    
    with tab1:      
        brainNX(G, colorlist, colornumbs, lineList, sublist)
    with tab2:
        m_tab2 = matrix1.copy()
        columns = [m_tab2.columns.tolist()[i] for i in list((np.argsort(ind)))]
        m_tab2 = m_tab2[columns]; m_tab2 = m_tab2.T; 
        m_tab2 = m_tab2[columns]; m_tab2 = m_tab2.T; 
        plot_corr(m_tab2)        
    with tab3:
        m_tab3 = matrix1.copy()
        columns = [m_tab3.columns.tolist()[i] for i in list((np.argsort(ind)))]        
        columns_L = [col for col in columns if col.lstrip()[0]=='L']
        columns_R = [col for col in columns if col.lstrip()[0]!='L']
        columns = columns_L + columns_R
        m_tab3 = m_tab3[columns]; m_tab3 = m_tab3.T; 
        m_tab3 = m_tab3[columns]; m_tab3 = m_tab3.T; 
        plot_corr(m_tab3)      
