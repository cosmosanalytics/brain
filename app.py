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
import ndlib.models.opinions as opn
from ndlib.viz.mpl.DiffusionTrend import DiffusionTrend
#st.set_option('deprecation.showPyplotGlobalUse', False)
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
    matrix = matrix.loc[Regions_Nodes,Regions_Nodes]

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
'''
def dynBrainNX(g,epsilon,init):
    model = opn.WHKModel(g)
    config = mc.Configuration()
    config.add_model_parameter("epsilon", epsilon)
    for e in g.edges:
        config.add_edge_configuration("weight", e, g.get_edge_data(*e)['weight'])          
    model.set_initial_status(config)

    initial_statuses = {node: i for node,i in zip(g.nodes(),init)}  # custom initial statuses: values in [-1, 1]
    model.status = initial_statuses
    model.initial_status = initial_statuses    
    
    iterations = model.iteration_bunch(100, node_status=True)
    return iterations
'''

import networkx as nx

def dynBrainNX(g, epsilon, init):
    model = opn.WHKModel(g)
    config = mc.Configuration()
    config.add_model_parameter("epsilon", epsilon)

    # Identify nodes with negative initial states
    negative_nodes = [node for node, value in zip(g.nodes(), init) if value < 0]

    if negative_nodes:
        # Create a copy of the graph without negative nodes
        g_without_negatives = g.copy()
        g_without_negatives.remove_nodes_from(negative_nodes)

        # Find all pairs of nodes that were connected through negative nodes
        affected_pairs = []
        for node1 in g.nodes():
            for node2 in g.nodes():
                if node1 < node2 and node1 not in negative_nodes and node2 not in negative_nodes:
                    path = nx.shortest_path(g, node1, node2)
                    if any(node in negative_nodes for node in path):
                        affected_pairs.append((node1, node2))

        # Find the shortest bypass paths for affected pairs
        bypass_edges = set()
        for node1, node2 in affected_pairs:
            if nx.has_path(g_without_negatives, node1, node2):
                bypass_path = nx.shortest_path(g_without_negatives, node1, node2)
                bypass_edges.update(zip(bypass_path[:-1], bypass_path[1:]))

        # Calculate the total weight to redistribute
        total_weight_to_redistribute = sum(g.degree(node, weight='weight') for node in negative_nodes)

        # Redistribute the weight
        if bypass_edges:
            weight_per_edge = total_weight_to_redistribute / len(bypass_edges)
            
            for e in g.edges():
                if e in bypass_edges or (e[1], e[0]) in bypass_edges:
                    original_weight = g.get_edge_data(*e)['weight']
                    new_weight = original_weight + weight_per_edge
                    config.add_edge_configuration("weight", e, new_weight)
                else:
                    # Keep the original weight for other edges
                    config.add_edge_configuration("weight", e, g.get_edge_data(*e)['weight'])
        else:
            # If there are no bypass edges, keep original weights
            for e in g.edges():
                config.add_edge_configuration("weight", e, g.get_edge_data(*e)['weight'])
    else:
        # If there are no negative nodes, keep original weights
        for e in g.edges():
            config.add_edge_configuration("weight", e, g.get_edge_data(*e)['weight'])

    model.set_initial_status(config)

    initial_statuses = {node: i for node, i in zip(g.nodes(), init)}  # custom initial statuses: values in [-1, 1]
    model.status = initial_statuses
    model.initial_status = initial_statuses

    iterations = model.iteration_bunch(100, node_status=True)
    return iterations


matrix, colorlist, colornumbs, lineList, sublist, refDF = loadData()    
col1, col2 = st.columns(2)
with col1:
    # Regions = st.multiselect('Select Region(s) to Focus', set(sublist), set(sublist))
    # Regions = st.multiselect('Select Region(s) to Focus', set(sublist), ['DMN'])
    # Regions_Nodes = refDF[refDF['sublist'].isin(Regions)]['lineList'].values
    Regions_Nodes = ['LPG4','LP1','RC1','LSPL1','RAG1','LAG1',\ #DMN
                     'LC1','LC2','LH1','RH1','RH2',\ #LIM
                     'LIC2','RFP1','RFP2','LFP1','LFP2',\ #FP
                     'RPG2','LT2','LPG8','RPG10',\ #VA
                     'RAG2','RP1','RT1','RIC1','RT2','LPG12',\ #SM
                     'RSPL1','LPG6','RPG8','LIC3','B1',\ #VIS
                     'RAG1','LAG1'] #MA
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
        st.write('The idea behind the WHK formulation is that the opinion of agent i at time t+1, will be given by the average opinion by its, selected, ϵ-neighbor.')
        epsilon = st.slider('epsilon-neighbor', 0.0, 1.0, 0.5)
        SM = pd.Series(st.text_input('SENSORIMOTOR NODES TO FOCUS: (RAG2, RP1, RT1, RIC1, RT2, LPG12)', '0.0, 0.0, 0.0, 0.0, 0.0, 0.0').split(',')).astype(float)
        DMN = pd.Series(st.text_input('DEFAULT MODE NETWORK NODES TO FOCUS: (LPG4', LP1, RC1, LSPL1, RAG1, LAG1)', '0.0, 0.0, 0.0, 0.0, 0.0, 0.0').split(',')).astype(float)
        LIM = pd.Series(st.text_input('LIMBIC NODES TO FOCUS: (LC1, LC2, LH1, RH1, RH2)', '0.0, 0.0, 0.0, 0.0, 0.0').split(',')).astype(float)
        VIS = pd.Series(st.text_input('VIS NODES TO FOCUS: (RSPL1, LPG6, RPG8, LIC3, B1)', '0.0, 0.0, 0.0, 0.0, 0.0').split(',')).astype(float)
        FP = pd.Series(st.text_input('FP NODES TO FOCUS: (LIC2, RFP1, RFP2, LFP1, LFP2)', '0.0, 0.0, 0.0, 0.0, 0.0').split(',')).astype(float)
        VA = pd.Series(st.text_input('VA NODES TO FOCUS: (RPG2, LT2, LPG8, RPG10)','0.0, 0.0, 0.0, 0.0').split(',')).astype(float)
        MS = pd.Series(st.text_input('MISCELLANEOUS : (RAG1,LAG1)', '0.0, 0.0').split(',')).astype(float)
                           
        init = pd.concat([SM, DMN, LIM, VIS, FP, VA, MS])
 
        if st.button('simulation'):
            iterations = dynBrainNX(G,epsilon,init)
            df = pd.DataFrame(iterations)
            dff = df['status'].apply(lambda x: pd.Series(x))
            dff.columns = matrix1.columns
            st.table(dff.T.style.background_gradient(axis=None, cmap='seismic'))
            fig, ax = plt.subplots(figsize=(20, 10));
            dff.plot(ax=ax).legend(loc='best')
            st.pyplot(fig)
            res = dff.T
            res = res[res.columns[-1]]
            st.table(res[res<-0.99].index)
            st.table(res[res>0.99].index)        
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
