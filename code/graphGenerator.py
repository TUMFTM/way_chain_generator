from srai.loaders.osm_way_loader import OSMNetworkType
from srai.regionalizers.geocode import geocode_to_region_gdf
from srai.embedders import ContextualCountEmbedder, CountEmbedder
from srai.neighbourhoods import H3Neighbourhood
from srai.joiners import IntersectionJoiner
from srai.regionalizers import H3Regionalizer
from srai.loaders import OSMOnlineLoader
import geopandas as gpd
import osmnx as ox
import h3
import torch
import networkx as nx

import config

def download_map(resolution, query_string):
    query = {}
    for item in query_string:
        query[item] = True

    ox.settings.overpass_endpoint = config.overpass_endpoint
    regionizer = H3Regionalizer(resolution=resolution)
    joiner = IntersectionJoiner()
    area = geocode_to_region_gdf(config.area_name)
    loader = OSMOnlineLoader()
    features = loader.load(area, query)
        # save features
    regions = regionizer.transform(area)
    joint = joiner.transform(regions, features)
    return regions, features, joint

def get_street_network():
    area = geocode_to_region_gdf(config.area_name)
    G_directed = ox.graph_from_polygon(
            area["geometry"][0],
            network_type=OSMNetworkType.DRIVE,
            retain_all=True,
            clean_periphery=True,
            truncate_by_edge=True,
        )
    G_undirected = ox.utils_graph.get_undirected(G_directed)
    G_undirected = ox.add_edge_speeds(G_undirected) 
    G_undirected = ox.add_edge_travel_times(G_undirected) 
    return G_undirected

def build_edges(regions, edge_weights=True):
    region_list = regions.reset_index()["region_id"].to_list() # ToDo use a set isntead
    adj = torch.zeros(len(region_list), len(region_list))
    edge_list = []
    neighbourhood = H3Neighbourhood()
    #this is gonna be slow... like real slow!
    counter = 0
    missing_counter = 0
    if edge_weights:
        street_network = get_street_network()
        central_points = {}
        for node in region_list:
            central_points[node] = ox.distance.nearest_nodes(street_network, h3.cell_to_latlng(node)[1], h3.cell_to_latlng(node)[0]) # swap  x and y due to h3 in lat/lng and osm and lng/lat
    for region in region_list:
        neighbours = neighbourhood.get_neighbours_at_distance(region, 1)
        for neighbour in neighbours:
            length = 1
            if edge_weights:
                if region not in central_points or neighbour not in central_points:
                    length = 1000
                else:
                    path = ox.distance.shortest_path(street_network, central_points[region], central_points[neighbour], weight="travel_time")
                    if path is None:
                        length = 10000
                    else:
                        try:
                            length = sum(street_network.edges[u, v, 0]['length'] for u, v in zip(path[:-1], path[1:]))
                        except Exception as error: 
                            length = 10000
                            print("Warning during graph generation, no correct length found")
                edge_list.append((region, neighbour, length))
            try:
                index = region_list.index(neighbour)
                adj[counter][index] = length
            except:
                missing_counter = missing_counter +1
        counter = counter + 1
        print("created graph edge: "+str(counter))
    return adj, edge_list

def create_positions_dict(graph): 
    pos_dict = {}
    for node in graph.nodes():
        try:
            pos_dict[node] = graph.nodes()[node]['pos']
        except:
            print("position not given for node "+node)
    return pos_dict

def build_graph(node_names, node_features, edge_list):
    graph = nx.Graph()
    for index, node in node_features.iterrows():
        graph.add_node(node.region_id, x=torch.from_numpy(node.drop("region_id").values.astype(np.float64)), pos=h3.cell_to_latlng(node.region_id)[::-1])
    # Add edges from adjacency matrix
    for edge in edge_list:
        graph.add_edge(edge[0], edge[1], weight=edge[2])
    return graph

def setup_graph(region_list, embeddings, edge_list):
    nodes = region_list
    features = embeddings.reset_index()
    my_graph = build_graph(nodes, features, edge_list)
    return my_graph

def create_graph(resolution, query_string):
    regions, features, joint = download_map(resolution, query_string)
    embedder = CountEmbedder()
    embeddings= embedder.transform(regions, features, joint)
    adj, edge_list = build_edges(regions, edge_weights=True)
    G = setup_graph(regions.reset_index()["region_id"].to_list(), embeddings, edge_list)
    # filter all border nodes without features
    for node in list(G.nodes()):
        if not 'x' in G.nodes()[node]:
            G.remove_node(node)
    pos_dict = create_positions_dict(G)
    return G, pos_dict
