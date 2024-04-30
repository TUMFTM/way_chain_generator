import dill
import mobilityAnalyzer
import generator
import graphGenerator
import config

def calculate_route(output, continuos=False):
    routes = []
    street_network = mobilityAnalyzer.get_street_network()
    path_local=output["result"]["grid_paths"]["path"].copy()
    # make touples according to the continuos argument
    point_tuples = []
    if continuos:
        starting_building = path_local.pop()["node"]
        for node in path_local:
            target_building = node["node"]
            origin_node, target_node = mobilityAnalyzer.find_starting_point_on_graph(street_network, starting_building.geometry.centroid, target_building.geometry.centroid)
            point_tuples.append((starting_building, target_building))
            starting_building = target_building
    else:
        # make an array with all start buildings 
        start_buildings = path_local[0::2]
        # make array with all endbuildings
        end_buildings = path_local[1::2]
        # go through arrays and form tuples
        for i in range (0, len(end_buildings)):
            point_tuples.append((start_buildings[i]["building"], end_buildings[i]["building"]))
    for node in point_tuples:
        origin_node, target_node = mobilityAnalyzer.find_starting_point_on_graph(street_network, node[0].geometry.centroid, node[1].geometry.centroid)
        route, linestring = mobilityAnalyzer.routing_on_graph(street_network, origin_node, target_node, visualize=False)
        routes.append(linestring)
    return routes

def resynthesize_track(path): # load a artificial track, resynthesize it and use the 1st artificial as original
    with open(path, 'rb') as file:
        output = dill.load(file)
    routes = calculate_route(output)
    tracks = mobilityAnalyzer.cluster_tracks(tracks=routes)
    tracks = [track.to_crs(config.WGS84_EPSG) for track in tracks]
    G, node_pos_dict = graphGenerator.create_graph(area_name=config.area_name, resolution=config.resolution, query_string=config.query_string)
    length_dict = mobilityAnalyzer.calculate_length_dict(G)
    output = generator.create_path_chain(tracks, force_update_embedding=True, visualization=False, G = G, encoding_dict=encodings, length_dict=length_dict)
    return {'tracks': tracks, 'result': output}
resynthesize_track("./cached_experiments/2024_02_21/11")