import torch
import networkx as nx
import shapely
import h3
import numpy as np
import random
from sknetwork.path import get_distances

import config
import graphGenerator
import encoder
import mobilityAnalyzer



def create_target_vector(G, init_Node=None, gravity_center=None, random_node=None):
    """_summary_

    Args:
        G (networkx graph): The graph where a random orign target couple should be generated
        init_Node (int, optional): The index of the origin node for which a random target node should be generated. Defaults to None.
        gravity_center (int, optional): The index of the gravity center node for a generated way chain. Defaults to None.

    Returns:
        Tensor[]: An array with the feature vector of the origin node and the feature vector of the target node, and the distance between those two
        int: For debugging the index of the selected origin node
        int: For debugging the index of the selected target node
        int: index of gravity center
        int: distance to the given gravity center, defaults to 0 if no center is given
    """
    y = {}
    nodes_list = list(G.nodes())
    if init_Node is None:
        initNode = nodes_list[random.randint(0, G.number_of_nodes()-1)]
    else:
        initNode=init_Node
    adj = nx.to_numpy_matrix(G)
    y["init_features"]=G.nodes[initNode]['x']
    if random_node is None:
        randomNode = nodes_list[random.randint(0, G.number_of_nodes()-1)]
    else:
        randomNode = random_node
    y["target_features"] = G.nodes[randomNode]['x']
    y["distance"] = get_distances(adj, [nodes_list.index(initNode), nodes_list.index(randomNode)])[0][nodes_list.index(randomNode)]
    gravity_distance = 0
    if gravity_center is not None:
        gravity_distance = get_distances(adj, [nodes_list.index(gravity_center), nodes_list.index(randomNode)])[0][nodes_list.index(randomNode)]
    return {'feature_vector':y, 'init_Node': initNode, 'target_Node':randomNode, 'gravity_center': gravity_center, 'gravity_distance':gravity_distance}

def find_match_dict(origin, encodings):
    diff_dict = {key: abs(sum(value - origin)) for key, value in encodings.items()}
    return sorted(diff_dict.items(), key=lambda item: item[1])

def create_embedding(graph, X, Y, input_dimensions:int, linear_hidden_one:int, linear_hidden_two:int, linear_out:int, graph_hidden:int, graph_out:int, ensemble_hidden:int, ensemble_out:int):
    model = encoder.initialize_model(input_dimensions, linear_hidden_one, linear_hidden_two, linear_out, graph_hidden, graph_out, ensemble_hidden, ensemble_out)
    model = encoder.train_model(model, graph, X, Y, visualization=True)
    return encoder.create_latent_space(model, graph, X)

def sort_nodes_dict(G, start, target, length, gravity_center=None ,distance_gravity_center=None, phi=0., length_dict=None):
    """_summary_

    Args:
        G (networkX - Graph): The graph the search should concluded on

        start (string): the node id of the starting node for the graph search

        origin (tensor[NUM_FEATURES]): Feature set of origin node

        target (tensor[NUM_FEATURES]): Feature set of destination node

        length (float): Length from origin to destination node.

        gravity_center(int): node, that gives the gravity center of the way-chain (e.g. home)

        distance_gravity_center(float): distance from the gravity center of the node (not sure if embedded space or graph-distance)

        phi (float, optional): max deviation from origin to destination features. Defaults to 0.9.
    """
    if length_dict is None:
        length_dict = dict(nx.all_pairs_shortest_path_length(G))
    scoredNodes = {}
    # scoring function based on features
    for ind, node in enumerate(G.nodes()):
        diff = G.nodes[node]['x'] - target
        scoredNodes[node] = abs(diff).sum()
    latent_distances = [tensor.item() for tensor in list(scoredNodes.values())]
    for ind, node in enumerate(G.nodes()):
        if length != 0:
            length_factor = pow(abs((length_dict[start][node]-length)/length)+1, 2)
        else: 
            length_factor = 1
        scoredNodes[node] = scoredNodes[node]*length_factor + length_factor * np.percentile(latent_distances, 10)
    # scoring based on gravity center
    if gravity_center is not None and distance_gravity_center is not None:
        for ind, node in enumerate(G.nodes()):
            indName = list(G.nodes())[ind]
            if distance_gravity_center > 0:
                gravity_factor = pow(abs((length_dict[gravity_center][node]-distance_gravity_center)/distance_gravity_center)+1,2)
            else:
                gravity_factor = 1
            scoredNodes[node] = scoredNodes[node]*gravity_factor
    return sorted(scoredNodes.items(), key=lambda item: item[1])

def create_path_chain(tracks, additional_target=None, G = None, encoding_dict = None, length_dict = None):
    cluster_dict = {}
    current_path = []
    original_way = [] 
    #do the first track
    if G is None:
        G, node_pos_dict = graphGenerator.create_graph(resolution=config.resolution, query_string=config.query_string)
    node_features = {node: G.nodes[node]['x'] for node in G.nodes()}
    dimensions_map = len(list(node_features.items())[0][1])
    if encoding_dict is None:
        X = []
        Y = []
        for key, value in node_features.items():
            X.append(value)
            Y.append(value)
        encoding, graph_encoding, y_pred = create_embedding(G, X, Y, input_dimensions=dimensions_map, linear_hidden_one=256, linear_hidden_two=64, linear_out=config.feature_encoding_dimensions, graph_hidden=256, graph_out=config.encoding_dimensions, ensemble_hidden=64, ensemble_out=len(Y[0]))
        encoding_dict_edges = {k: v for k, v in zip(G.nodes(), graph_encoding)}
        encoding_dict_dict = {k: v for k, v in zip(G.nodes(), encoding)}

        encoding_dict = {}
        for k in encoding_dict_edges.keys():
            if not(k in encoding_dict_edges.keys()):
                print("Key missing in encoding for edges")
                encoding_dict_edges[k] = torch.zeros(next(iter(encoding_dict_edges.values())).size())
            if not(k in encoding_dict_dict.keys()):
                print("Key missing in encoding for features")
                encoding_dict_dict[k] = torch.zeros(next(iter(encoding_dict_dict.values())).size())
            encoding_dict[k] = torch.cat((encoding_dict_edges[k], encoding_dict_dict[k]), 0)

    for grid, encoding in encoding_dict.items():
        G.nodes[grid]['x'] = encoding
    if length_dict is None:
        length_dict = mobilityAnalyzer.calculate_length_dict(G)
    step=1
    cluster_graph = mobilityAnalyzer.analyze_clusters(tracks)
    gravitational_cluster = max(nx.betweenness_centrality(cluster_graph), key=nx.betweenness_centrality(cluster_graph).get)
    gravitational_grid = None
    for track in tracks:
        if track["start_cluster"][0] == gravitational_cluster:
            grav_point = shapely.Point(track.geometry[0].coords[0][0], track.geometry[0].coords[0][1])
            gravitational_grid = h3.latlng_to_cell(lat=grav_point.y, lng=grav_point.x, res=config.resolution)
        if track["end_cluster"][0] == gravitational_cluster:
            grav_point = shapely.Point(track.geometry[0].coords[-1][0], track.geometry[0].coords[-1][1])
            gravitational_grid = h3.latlng_to_cell(lat=grav_point.y, lng=grav_point.x, res=config.resolution)
    for track in tracks: 
        # check if the start point is in the same cluster as the last point?
        # if yes-> take that cluster!
        # if no ->
        #   1. Possibility: just accept that there will be a gap
        #   2. Possibility: find the route normally and then add a spline route in between. Determine the distance in between based on the gap distance in the data   
        start_point = shapely.Point(track.geometry[0].coords[0][0], track.geometry[0].coords[0][1])
        end_point = shapely.Point(track.geometry[0].coords[-1][0], track.geometry[0].coords[-1][1])
        original_starting_grid = h3.latlng_to_cell(lat=start_point.y, lng=start_point.x, res=config.resolution) # Turn coordinates other way aroudn for EPSG lon lat format
        original_ending_grid = h3.latlng_to_cell(lat=end_point.y, lng=end_point.x, res=config.resolution)
        start_vector = create_target_vector(G,init_Node=original_starting_grid, gravity_center=gravitational_grid, random_node=original_ending_grid)
        start_building = mobilityAnalyzer.get_next_building_in_grid(start_point,start_vector['init_Node'])
        original_way.append([(start_vector["init_Node"], start_building.copy())])
        end_building = mobilityAnalyzer.get_next_building_in_grid(end_point,start_vector['target_Node'])
        original_way[-1].append((start_vector['target_Node'], end_building.copy()))
        sorted_encoding = find_match_dict(encoding_dict[start_vector['init_Node']] ,encoding_dict)
        if track["start_cluster"][0] in cluster_dict:
            starting_building = cluster_dict[track["start_cluster"][0]][0]
            start_grid = [pair for pair in sorted_encoding if pair[0] == cluster_dict[track["start_cluster"][0]][1][0]][0] # TODO more dicts!
            # create start grid
        else:
            start_grid = sorted_encoding.pop(0)
            if config.anonymization_strategy["DEACTIVATE"]:
                start_grid = sorted_encoding.pop(0) 
            if config.anonymization_strategy["EPSILON"]:
                if float(start_grid[1]) < config.anonymization_epsilon:
                    for node in sorted_encoding:
                        if float(node[1]) > config.anonymization_epsilon:
                            start_grid = node
                            break
            building = start_building
            gdf = mobilityAnalyzer.get_buildings_in_grid(start_grid[0])
            mobilityAnalyzer.add_similarity_costs(gdf, building)
            if gdf.empty or gdf[gdf.similarity_costs==gdf.similarity_costs.max()].empty:
                print("gdf is still empty")
                starting_building = start_building
                starting_building.similarity_costs = -1
                starting_building.geometry = shapely.Polygon(h3.cell_to_boundary(start_grid[0], geo_json=True))
                starting_building.full_categories = 'dummy'
            else:
                starting_building = gdf[gdf.similarity_costs==gdf.similarity_costs.max()].sample(n=1).iloc[0]
            cluster_dict[track["start_cluster"][0]] = (starting_building, start_grid)
        if gravitational_cluster in cluster_dict and not(gravitational_grid is None):
            gravity_center = cluster_dict[gravitational_cluster][1][0]
            # get the distance between the original track ending and the gravitational cluster
            distance_gravity_center = length_dict[gravity_center][gravitational_grid]
        else:
            gravity_center = None
            distance_gravity_center = None
        sorted_nodes = sort_nodes_dict(G=G, start=start_grid[0], target=start_vector["feature_vector"]["target_features"], length=start_vector["feature_vector"]["distance"], gravity_center=gravity_center ,distance_gravity_center=distance_gravity_center, phi=0.9, length_dict=length_dict)
        if track["end_cluster"][0] in cluster_dict:
            ending_building = cluster_dict[track["end_cluster"][0]][0]
            if track["end_cluster"][0] == track["start_cluster"][0] or len([pair for pair in cluster_dict.items() if pair[1][1][0] == start_grid[0]]) > 0: # check if the start and end cluster are the same or if the seleted node of the start cluster is the same as the node for the end cluster
                target_node = start_grid # TODO check if a deep copy is of use here
            else:
                target_node = [pair for pair in sorted_encoding if pair[0] == cluster_dict[track["end_cluster"][0]][1][0]][0]
        else:
            if original_starting_grid == original_ending_grid:
                target_node = start_grid
            else:
                target_node = sorted_nodes.pop(0)
                if config.anonymization_strategy["DEACTIVATE"]:
                    target_node = sorted_nodes.pop(0) #
                if config.anonymization_strategy["EPSILON"]:
                    if float(target_node[1]) < config.anonymization_epsilon:
                        for node in sorted_encoding:
                            if float(node[1]) > config.anonymization_epsilon:
                                target_node = node
                                break
            building = end_building
            gdf = mobilityAnalyzer.get_buildings_in_grid(target_node[0])
            mobilityAnalyzer.add_similarity_costs(gdf, building)
            if gdf.empty or gdf[gdf.similarity_costs==gdf.similarity_costs.max()].empty:
                print("gdf is still empty")
                ending_building = building
                ending_building.similarity_costs = -1
                ending_building.geometry = shapely.Polygon(h3.cell_to_boundary(target_node[0], geo_json=True))
                ending_building.full_categories = 'dummy'
            else:
                ending_building = gdf[gdf.similarity_costs==gdf.similarity_costs.max()].sample(n=1).iloc[0]
            cluster_dict[track["end_cluster"][0]] = (ending_building, target_node)
        current_path.append({"node": start_grid[0], "start_cluster":track["start_cluster"][0], "end_cluster":track["end_cluster"][0],"building": starting_building, "original_building": start_building, "costs":float(start_grid[1]) , "step":step, "type":"start"})# add first node of path
        step = step + 1
        current_path.append({"node": target_node[0], "start_cluster":track["start_cluster"][0], "end_cluster":track["end_cluster"][0],"building": ending_building, "original_building": end_building, "costs":target_node[1] ,"step":step, "type":"end"}) # add the conescutive node to the array, note: thios function will give only on path at the beginning
        step = step + 1
    return {"original_way":original_way, "grid_paths":{'path': current_path, 'costs': 1, 'id' : 1}, "clusters":cluster_dict} # original way, sorted paths, Point-By-Point {geom: POINT, building: BUILDING_INFO, cluster:CLUSTER_ID} - in addition define a method to take these point by point information and create a networkx graph out of it