import streamlit as st
import ast
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
    
    # Increase font size for x-axis tick labels
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=6)
    
    # Increase font size for y-axis tick labels
    plt.yticks(range(len(corr.columns)), corr.columns, fontsize=6)
    
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
    matrix.loc[matrix.index.isin(LinkNodesToWeaken), :] = 0.01;  matrix.loc[:, matrix.columns.isin(LinkNodesToWeaken)] = 0.01
    matrix.loc[matrix.index.isin(LinkNodesToStrengthen), :] = 0.499; matrix.loc[:, matrix.columns.isin(LinkNodesToStrengthen)] = 0.499
  
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

def dynBrainNX(g,epsilon,init,additional_states):
# def dynBrainNX(g,epsilon,init):
    model = opn.WHKModel(g)
    config = mc.Configuration()
    config.add_model_parameter("epsilon", epsilon)
    for e in g.edges:
        config.add_edge_configuration("weight", e, g.get_edge_data(*e)['weight'])          
    model.set_initial_status(config)

    initial_statuses = {node: i for node,i in zip(g.nodes(),init)}  # custom initial statuses: values in [-1, 1]
    model.status = initial_statuses
    model.initial_status = initial_statuses    
    ####################

    iterations = []
    for i in range(100):
        if i in additional_states:
            # Update the model status with additional states
            for node, state in additional_states[i].items():
                model.status[node] = state

        # Perform a single iteration
        iteration_result = model.iteration(node_status=True)
        iterations.append(iteration_result) 
  
    ###################    
    # iterations = model.iteration_bunch(100, node_status=True)
    return iterations

matrix, colorlist, colornumbs, lineList, sublist, refDF = loadData()    
#col1, col2 = st.columns(2)
#with col1:
###################
# Regions = st.multiselect('Select Region(s) to Focus', set(sublist), set(sublist))
# Regions = st.multiselect('Select Region(s) to Focus', set(sublist), ['DMN'])
Regions_Nodes = ['RPC1', 'RPC2', 'RPC3', 'RPC4', 'RPC5', 'LPC1', 'LPC2', 'LPC3', 'LPC4', 'RCGpd1', 'RCGpd2', 'LCGpd1', 'RAG1', 'RAG2', 'LAG1',\
                 'RH1', 'RH2', 'LH1', \
                 'RPG1', 'RPG2', 'RPG3', 'RPG4', 'RPG5', 'RPG6', 'RPG7', 'RPG8', 'RPG9', 'RPG10', 'RPG11', \
                 'LPG1', 'LPG2', 'LPG3', 'LPG4', 'LPG5', 'LPG6', 'LPG7', 'LPG8', 'LPG9', 'LPG10', 'LPG11', 'LPG12', 'LPG13', 'LA1',\
                 'RIC1', 'RIC2', 'LIC1', 'LIC2', 'LIC3', 'RCGad1', 'RCGad2', 'RCGad3', 'RCGad4', 'LCC1',\
                 'RMFG1', 'RMFG2', 'RMFG3', 'RMFG4', 'LMFG1', 'LMFG2', 'LMFG3', 'LMFG4', 'RSPL1', 'LSPL1', 'LSPL2',\
                 'RT1', 'RT2', 'LT1', 'LT2']
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
##################    
#with col2: 
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

    DMN = pd.Series(st.text_input('DEFAULT MODE NETWORK NODES TO FOCUS: (RPC1,RPC2,RPC3,RPC4,RPC5,LPC1,LPC2,LPC3,LPC4,RCGpd1,RCGpd2,LCGpd1,RAG1,RAG2,LAG1)', '0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0').split(',')).astype(float)
    LIM = pd.Series(st.text_input('LIMBIC NODES TO FOCUS: (RH1,RH2,LH1,RPG1,RPG2,RPG3,RPG4,RPG5,RPG6,RPG7,RPG8,RPG9,RPG10,RPG11,LPG1,LPG2,LPG3,LPG4,LPG5,LPG6,LPG7,LPG8,LPG9,LPG10,LPG11,LPG12,LPG13,LA1)', \
                                  '0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0').split(',')).astype(float)
    VA = pd.Series(st.text_input(' VA NODES TO FOCUS: (RIC1,RIC2,LIC1,LIC2,LIC3,RCGad1,RCGad2,RCGad3,RCGad4,LCC1)', '0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0').split(',')).astype(float)
    FP = pd.Series(st.text_input('FP NODES TO FOCUS: (RMFG1,RMFG2,RMFG3,RMFG4,LMFG1,LMFG2,LMFG3,LMFG4,RSPL1,LSPL1,LSPL2)', '0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0').split(',')).astype(float)
    SM = pd.Series(st.text_input('SM NODES TO FOCUS: (RT1,RT2,LT1,LT2)', '0.0, 0.0, 0.0, 0.0').split(',')).astype(float)                       
    init = pd.concat([DMN, LIM, VA, FP, SM])

    def parse_additional_states(input_string):
        try:
            # Convert the input string to a dictionary
            input_dict = ast.literal_eval(input_string)
            
            # Validate and convert the input
            additional_states = {}
            for iteration, states in input_dict.items():
                iteration = int(iteration)
                additional_states[iteration] = {int(node): float(state) for node, state in states.items()}
            
            return additional_states
        except:
            st.error("Invalid input format. Please check your input and try again.")
            return None    

    additional_states_input = st.text_input(
        """Additional States {
    'RPC1': 0, 'RPC2': 1, 'RPC3': 2, 'RPC4': 3, 'RPC5': 4, 
    'LPC1': 5, 'LPC2': 6, 'LPC3': 7, 'LPC4': 8, 
    'RCGpd1': 9, 'RCGpd2': 10, 'LCGpd1': 11, 
    'RAG1': 12, 'RAG2': 13, 'LAG1': 14,
    'RH1': 15, 'RH2': 16, 'LH1': 17,
    'RPG1': 18, 'RPG2': 19, 'RPG3': 20, 'RPG4': 21, 'RPG5': 22, 'RPG6': 23, 'RPG7': 24, 'RPG8': 25, 'RPG9': 26, 'RPG10': 27, 'RPG11': 28,
    'LPG1': 29, 'LPG2': 30, 'LPG3': 31, 'LPG4': 32, 'LPG5': 33, 'LPG6': 34, 'LPG7': 35, 'LPG8': 36, 'LPG9': 37, 'LPG10': 38, 'LPG11': 39, 'LPG12': 40, 'LPG13': 41, 'LA1': 42,
    'RIC1': 43, 'RIC2': 44, 'LIC1': 45, 'LIC2': 46, 'LIC3': 47, 
    'RCGad1': 48, 'RCGad2': 49, 'RCGad3': 50, 'RCGad4': 51, 'LCC1': 52,
    'RMFG1': 53, 'RMFG2': 54, 'RMFG3': 55, 'RMFG4': 56, 
    'LMFG1': 57, 'LMFG2': 58, 'LMFG3': 59, 'LMFG4': 60, 
    'RSPL1': 61, 'LSPL1': 62, 'LSPL2': 63,
    'RT1': 64, 'RT2': 65, 'LT1': 66, 'LT2': 67
}""",
        '{10: {1: 0.499},}',
        help="Enter additional states as a dictionary. Format: {iteration: {node: state, ...}, ...}"
    )
    additional_states = parse_additional_states(additional_states_input)

    if st.button('simulation'):
        iterations = dynBrainNX(G,epsilon,init, additional_states)
        # iterations = dynBrainNX(G,epsilon,init)
        df = pd.DataFrame(iterations)
        dff = df['status'].apply(lambda x: pd.Series(x))
        dff.columns = matrix1.columns
        st.write(dff.T.style.background_gradient(axis=None, cmap='seismic'))
        # st.table(dff.T.style.background_gradient(axis=None, cmap='seismic'))
        fig, ax = plt.subplots(figsize=(20, 10));
        dff.plot(ax=ax)#.legend(loc='best')

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', ncol=2)
        
        ax.set_ylim(bottom=-1, top=1)
        st.pyplot(fig)
        res = dff.T
        res = res[res.columns[-1]]
        st.write(res[res<-0.99].index)
        st.write(res[res>0.99].index)        
with tab2:
    m_tab2 = matrix1.copy()
    columns = [m_tab2.columns.tolist()[i] for i in list((np.argsort(ind)))]
    m_tab2 = m_tab2[columns]; m_tab2 = m_tab2.T; 
    m_tab2 = m_tab2[columns]; m_tab2 = m_tab2.T; 
    if st.checkbox('Show matrix 2'):
       st.write(m_tab2)      
    plot_corr(m_tab2)       
with tab3:
    m_tab3 = matrix1.copy()
    columns = [m_tab3.columns.tolist()[i] for i in list((np.argsort(ind)))]        
    columns_L = [col for col in columns if col.lstrip()[0]=='L']
    columns_R = [col for col in columns if col.lstrip()[0]!='L']
    columns = columns_L + columns_R
    m_tab3 = m_tab3[columns]; m_tab3 = m_tab3.T; 
    m_tab3 = m_tab3[columns]; m_tab3 = m_tab3.T; 
    if st.checkbox('Show matrix 3'):
       st.write(m_tab3)           
    plot_corr(m_tab3)      
