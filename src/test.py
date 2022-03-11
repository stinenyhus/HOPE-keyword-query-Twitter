import pandas as pd
from functools import partial
import streamlit as st
import os
import datetime
import math
import networkx as nx
from nltk.util import bigrams 
from bokeh.palettes import Spectral4
from bokeh.models import (BoxSelectTool, Circle, EdgesAndLinkedNodes, HoverTool,
                          MultiLine, NodesAndLinkedEdges, Range1d, TapTool,
                          Range1d, ColumnDataSource, LabelSet)
from bokeh.plotting import from_networkx
from bokeh.transform import linear_cmap

def plot_bigrams(G,
                 word_freq: dict,
                 co_occurence: dict,
                 pos, 
                 palette_nodes: list,
                 palette_edges: list,  
                 title: str):

    """Plots node graph
    
    Args:
        G: A Networkx graph
        word_freq (dict):  A dictionary of bigram members and their frequencies across the dataset
        co_occurence (dict): A dictionaty of bigrams and their co-occurence values
        pos (dict): A dictionary of positions keyed by node  
        palette_nodes (list): A colour palette for the nodes 
        palette_edges (list):  A colour palette for the edges
        title (str): The title of the node graph

    Returns:
          plot: The node graph
    """    

    from bokeh.plotting import figure
    nx.set_node_attributes(G, name = 'freq', values = word_freq)
    nx.set_edge_attributes(G, name = 'co_occurence', values = co_occurence)

    node_highlight_color = Spectral4[1]
    edge_highlight_color = Spectral4[2]   

    color_nodes = 'freq'
    color_edges = 'co_occurence'

    plot = figure(tools = 'pan,wheel_zoom,save,reset', 
                  active_scroll ='wheel_zoom',
                  title = title)

    plot.title.text_font_size = '20px'

    plot.add_tools(HoverTool(tooltips = None), TapTool(), BoxSelectTool())
    network_graph = from_networkx(G, pos, scale = 10, center = (0, 0))
    print(network_graph.node_renderer.data_source.data[color_nodes])
    min_col_val_node = min(network_graph.node_renderer.data_source.data[color_nodes])
    max_col_val_node = max(network_graph.node_renderer.data_source.data[color_nodes])

    min_col_val_edge = min(network_graph.edge_renderer.data_source.data[color_edges])
    max_col_val_edge = max(network_graph.edge_renderer.data_source.data[color_edges])
    
    network_graph.node_renderer.glyph = Circle(size = 40, 
                                               fill_color =  linear_cmap(color_nodes, 
                                                                         palette_nodes, 
                                                                         min_col_val_node, 
                                                                         max_col_val_node),
                                               fill_alpha = 1)  

    network_graph.node_renderer.hover_glyph = Circle(size = 5, 
                                                     fill_color = node_highlight_color,
                                                     line_width = 3)

    network_graph.node_renderer.selection_glyph = Circle(size = 5, 
                                                         fill_color = node_highlight_color, 
                                                         line_width = 5)

    network_graph.edge_renderer.glyph = MultiLine(line_alpha = 1, 
                                                  line_color = linear_cmap(color_edges, 
                                                                           palette_edges, 
                                                                           min_col_val_edge, 
                                                                           max_col_val_edge),
                                                  line_width = 4)

    network_graph.edge_renderer.selection_glyph = MultiLine(line_color = edge_highlight_color, line_width = 4)
    network_graph.edge_renderer.hover_glyph = MultiLine(line_color = edge_highlight_color, line_width = 4)

    network_graph.selection_policy = NodesAndLinkedEdges()
    network_graph.inspection_policy = NodesAndLinkedEdges()
       
    plot.renderers.append(network_graph)
    
    x, y = zip(*network_graph.layout_provider.graph_layout.values())
    node_labels = list(G.nodes())
    source = ColumnDataSource({'x': x, 'y': y, 'name': [node_labels[i] for i in range(len(x))]})
    labels = LabelSet(x = 'x', 
                      y ='y', 
                      text = 'name', 
                      source = source, 
                      background_fill_color = 'pink', 
                      text_font_size = '24px', 
                      background_fill_alpha = .3)
    plot.renderers.append(labels)
    return plot

def bigram_freq(freq:dict,
                G,
                scale: bool):
    """Creates a dictionary of words (present in bigrams) and their frequencies
    
    Args:
        freq (dict): A dictionary with all words from the data corpus and their frequency values 
        G: A Networkx graph
        scale: A boolean value. If true, the logarithm of frequency values will be calculated  
    
    Returns:
          freq_dict (dict): The dictionary containg words from bigrams and their frequencies
    """

    freq_dict = {}
    
    for node in G.nodes():
        for word in freq:
                 if word[0] == node:
                     if scale == True:
                         freq_dict[word[0]] = (math.log2(word[1]))*3 
                     else:
                         freq_dict[word[0]] = word[1]
    return freq_dict

def read_pkl(label: str,
             path: str,
             data_prefix: str):
    new_path = os.path.join(path, f'{label.lower()}{data_prefix}')
    df = pd.read_pickle(new_path)  

    return df

# Maybe not necessary after all? 
def convert_to_week(week_y:str):
    date = datetime.datetime.strptime(week_y +'-1',  "%W-%Y-%w")
    # as_week = datetime.datetime.strftime(date, "%V-%Y")
    as_week = date.isocalendar()
    return as_week

def scale(data: dict):
    """Calculates logarithm base 2 and multiplies the result by 3
    Args:
        data (dict): A ditctionary with values

    Returns:
          data (dict): The input dictonary with updated values
    
    """
    for key, value in data.items():
        data[key] = (math.log2(value))*3

    return data


df = read_pkl(label = "mettef", 
                      path = '../mettef_files/mettef_streamlit/',                        
                      data_prefix = '_weekly_bigrams.pkl')
df_orig = read_pkl(label = "mettef", 
                      path = '../mettef_files/mettef_streamlit/',                        
                      data_prefix = '_bigrams.pkl') 
w_freq = read_pkl(label = "mettef", 
                      path = '../mettef_files/mettef_streamlit/',                        
                      data_prefix = '_w_freq.pkl') 
w_freq = read_pkl(label = "denmark", 
                      path = '../denmark_files/denmark_streamlit/',                        
                      data_prefix = '_w_freq.pkl') 
# st.write(df.head())
# st.write(w_freq)
# st.write(len(w_freq.index))

# Reverting order so that year is first - makes sorting soo much easier!
df["week_y"]=[d.split("-")[1]+"-"+d.split("-")[0] for d in df["week_y"]]

start = min(df["week_y"])
end = max(df["week_y"])

df = df.sort_values("week_y")
df = df[["bigram", "count", "week_y"]]

st.write(df)
weeks_to_show = st.multiselect("Show week number", list(df["week_y"].unique()))
# print(weeks_to_show)

##################
# Copy from Kate #
##################

# Greens
color_palette_nodes = ['#E0FFFF',  '#bcdfeb', '#62b4cf', '#1E90FF']
color_palette_edges = ['#8fbc8f', '#3cb371', '#2e8b57', '#006400']

freq_dict = w_freq.to_dict(orient = 'split')['data']
co_occurence = dict(df[["bigram", "count"]].values)                       
                            
G = nx.Graph()
value=30
for key, value in co_occurence.items():    
    G.add_edge(key[0], key[1], weight=(value * 5))

pos = nx.spring_layout(G, k = 4)

# scale co_occurence values 
scaled_co_occurence = scale(co_occurence)              
    
# create a dict of words (from bigrams) and their frequencies across the dataset
freq = bigram_freq(freq_dict, G, scale = True)        
print(freq)
# print(scaled_co_occurence)

fig = plot_bigrams(G, freq, scaled_co_occurence, pos, color_palette_nodes, color_palette_edges, 'Bigrams')