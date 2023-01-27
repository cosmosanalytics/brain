import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from itertools import permutations 
import networkx as nx
import matplotlib.pyplot as plt
import scipy
import scipy.cluster.hierarchy as sch
import itertools
import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc
from ndlib.viz.mpl.DiffusionTrend import DiffusionTrend
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="wide")
st.title('Brain Network')

def plot_corr(corr):
    fig, ax = plt.subplots(figsize=(20,20))
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

def defineG(matrix0, threshold, Regions_Nodes, Nodes, LinkNodesToWeaken, LinkNodesToStrengthen):
    matrix = abs(matrix0); matrix[matrix<=threshold] = 0  
    matrix = matrix[matrix.index.isin(Regions_Nodes)][matrix.columns[matrix.columns.isin(Regions_Nodes)]]
    matrix = matrix[matrix.index.isin(Nodes)][matrix.columns[matrix.columns.isin(Nodes)]]
    matrix.loc[matrix.index.isin(LinkNodesToWeaken), matrix.columns.isin(LinkNodesToWeaken)] = 0
    matrix.loc[matrix.index.isin(LinkNodesToStrengthen), matrix.columns.isin(LinkNodesToStrengthen)] = 0.5
    np.fill_diagonal(matrix.values, 0)

    G = nx.from_numpy_array(np.array(matrix))
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

def brainNX(G, lineList):
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

    fig, ax = plt.subplots(figsize=(20,17))
    edgewidth = [ d['weight'] for (u,v,d) in G.edges(data=True)]
    pos = nx.spring_layout(G, scale=5)
    nx.draw(G, pos, with_labels=True, width=np.power(edgewidth, 1), edge_color='red', node_size=normstrengthlist*20000, 
            labels=Convert(lineList), font_color='black', alpha=0.7, font_size=9)
    st.pyplot(fig)

def dynBrainNX(G,beta,gamma,infected_nodes):    
    model = ep.SIRModel(G)
    cfg = mc.Configuration()
    cfg.add_model_parameter('beta', beta) # infection rate
    cfg.add_model_parameter('gamma', gamma) # recovery rate
    cfg.add_model_initial_configuration("Infected", infected_nodes)
    model.set_initial_status(cfg)
    iterations = model.iteration_bunch(100, node_status=True)
    trends = model.build_trends(iterations)  
    fig, ax = plt.subplots(figsize=(20,3))
    viz = DiffusionTrend(model, trends)
    fig = viz.plot()
    st.pyplot(fig)
    return iterations

matrix, colorlist, colornumbs, lineList, sublist, refDF = loadData()    
col1, col2 = st.columns(2)
with col1:
    Regions = st.multiselect('Select Region(s) to Focus', set(sublist), set(sublist))
    Regions_Nodes = refDF[refDF['sublist'].isin(Regions)]['lineList'].values
    Nodes = st.multiselect('Select Node(s) to Focus', Regions_Nodes, Regions_Nodes)
    LinkNodesToWeaken = st.multiselect('Select Links in between Node(s) to Weaken', Regions_Nodes)
    LinkNodesToStrengthen = st.multiselect('Select Links in between Node(s) to Strengthen', Regions_Nodes)
    threshold = st.slider('Threshold to Filter', 0.0, 1.0, 0.0)
    G, matrix1 = defineG(matrix, threshold, Regions_Nodes, Nodes, LinkNodesToWeaken, LinkNodesToStrengthen)
    if st.checkbox('Show matrix'):
        st.write(matrix1)    
    closeness, betweenness, clustering, mean_clutering = centrality_calc(G,Nodes) 
    
    tab1, tab2 = st.tabs(["Bar Chart", "Distribution Chart"])
    with tab1:
        fig, ax = plt.subplots(figsize=(20, 4)); closeness.plot.bar(); ax.set_title('Closeness'); st.pyplot(fig)
        fig, ax = plt.subplots(figsize=(20, 4)); betweenness.plot.bar(); ax.set_title('Betweenness'); st.pyplot(fig)
        fig, ax = plt.subplots(figsize=(20, 4)); clustering.plot.bar(); ax.set_title('Clustering, average='+str(mean_clutering)); st.pyplot(fig)   
    with tab2:
        fig, axes = plt.subplots(3, 1, figsize=(20, 15)); 
        sns.distplot(closeness, kde=False, norm_hist=False, ax=axes[0]); axes[0].set_xlabel('Closeness'); axes[0].set_ylabel('Counts')
        sns.distplot(betweenness, kde=False, norm_hist=False, ax=axes[1]); axes[1].set_xlabel('Betweenness'); axes[1].set_ylabel('Counts')
        sns.distplot(clustering, kde=False, norm_hist=False, ax=axes[2]); axes[2].set_xlabel('Clustering Coefficient'); axes[2].set_ylabel('Counts'); 
        axes[2].set_title('average path length is '+str(round(nx.average_shortest_path_length(G, weight='distance'),2))+'Clustering, average='+str(round(mean_clutering,4)))
        st.pyplot(fig)            
with col2: 
    tab1, tab2, tab3 = st.tabs(["Brain Network Chart", "Clustered CorrCoef Matrix", "Left/Right CorrCoef Matrix"])

    matrix_order = matrix1.copy()
    X = matrix_order.values
    d = sch.distance.pdist(X)   
    L = sch.linkage(d, method='complete')
    ind = sch.fcluster(L, 0.5*d.max(), 'distance')    
    
    with tab1:  
        brainNX(G, matrix1.index)
        beta = st.slider('infection rate', 0.0, 0.01, 0.001, step=0.001, format='%2.3f')
        gamma = st.slider('recovery rate', 0.0, 0.1, 0.01)
        infected_nodes = st.multiselect('Select Infected Node(s)', Regions_Nodes)
        iterations = dynBrainNX(G,beta,gamma,infected_nodes)
        df = pd.DataFrame(iterations)
        dff = df['status'].apply(lambda x: pd.Series(x))
        dff.columns = matrix1.columns
        st.table(dff.T.style.applymap(lambda x: "background-color: blue" if x==0 else "background-color: yellow" if x==1 else "background-color: green" if x==2 else "background-color: white"))
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
