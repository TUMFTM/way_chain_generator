import osmnx as ox
import h3
import geopandas as gpd
import numpy as np
from shapely import ops, LineString
from shapely.geometry import Polygon
from srai.loaders.osm_way_loader import OSMNetworkType, OSMOnlineLoader
from srai.regionalizers.geocode import geocode_to_region_gdf
import networkx as nx
import config

def h3_to_polygon(h3_index):
    coords = h3.cell_to_boundary(h3_index)
    flipped = tuple(coord[::-1] for coord in coords)
    return Polygon(flipped)


def get_buildings_in_grid(h3_index='891f8d44a47ffff'):
    tags = {}
    for item in config.query_string:
        tags[item] = True
    #h3_index = '891f8d44a47ffff'
    polygon = h3_to_polygon(h3_index)
    # Define the coordinates of your polygon vertices
    # For example, a square with coordinates: (x, y)
    polygon_coords = polygon
    # Create a Polygon object
    polygon = Polygon(polygon_coords)
    # Create a GeoDataFrame with the polygon
    gdf = gpd.GeoDataFrame({'geometry': [polygon]})
    # Optional: Set the coordinate reference system (CRS) if needed
    # For example, EPSG:4326 is WGS 84 (longitude and latitude)
    gdf.crs = config.WGS84_EPSG
    ox.settings.overpass_endpoint = config.overpass_endpoint
    loader = OSMOnlineLoader()
    features_gdf = loader.load(gdf, tags=tags)
    if 'building' not in features_gdf:
        features_gdf['building'] = np.NaN
    if 'shop' not in features_gdf:
        features_gdf['shop'] = np.NaN
    if 'amenity' not in features_gdf:
        features_gdf['amenity'] = np.NaN
    return features_gdf

def get_next_building(building_gdf, coords):
    return building_gdf.iloc[building_gdf.sindex.nearest(coords)[1][0]]

def get_next_building_in_grid(point, h3_grid):
    gdf = get_buildings_in_grid(h3_grid)
    return get_next_building(gdf, point)

def compare_buildings_general(building_ground, building_comp):
     similarity = 1
     for columm in building_comp.keys():
        if columm == "geometry" or columm == "geom":
            continue 
        if columm == "feature_id":
            continue
        if columm not in building_ground.keys():
            print("mismatch in features in building similarity")
            similarity = similarity - 1
        else:
            if building_ground[columm] == building_comp[columm]:
                similarity = similarity + 1
     return similarity

def add_similarity_costs(building_gdf, building):
    # scoring function based on features
    building_gdf['similarity_costs'] = building_gdf.apply(lambda x:compare_buildings_general(building, x), axis=1) 
    return building_gdf

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

def calculate_length_dict(G):
    # use routing
    street_network = get_street_network()
    central_points = {}
    for node in G.nodes():
        central_points[node] = ox.distance.nearest_nodes(street_network, h3.cell_to_latlng(node)[1], h3.cell_to_latlng(node)[0]) # swap  x and y due to h3 in lat/lng and osm and lng/lat
    adj = {}
    counter = 0
    for region in G.nodes():
        adj[region] = {}
        adj[region][region] = 0
        for target in G.nodes():
            if region == target:
                continue
            if target in adj:
                if region in adj[target]:
                    adj[region][target] = adj[target][region]
                    continue
            route = ox.shortest_path(street_network, central_points[region], central_points[target], weight="travel_time")
            nodes, edges = ox.graph_to_gdfs(street_network)
            if route is None:
                adj[region][target] = 10000
            else:
                route_nodes = nodes.loc[route]
                route_line = LineString(route_nodes['geometry'].tolist())
                gdf1 = gpd.GeoDataFrame(geometry=[route_line], crs=ox.settings.default_crs)
                length = gdf1.to_crs(config.METRIC_EPSG).geometry.length[0]
                adj[region][target] = length
            counter += 1
        return adj
  
def analyze_clusters(tracks):
    # if cluster=None, create cluster dict {id:Point}
    # add for cluster in clusters add Node to graph
    # adj matrix with 0
    # for track in tracks -> adj_matrix +1
    node_dict = {}
    edge_dict = {}
    for track in tracks:
        start_point = ops.Point(track.geometry[0].coords[0][0], track.geometry[0].coords[0][1])
        end_point = ops.Point(track.geometry[0].coords[-1][0], track.geometry[0].coords[-1][1])
        if track["start_cluster"][0] in node_dict.keys():
            node_dict[track["start_cluster"][0]].append(start_point)
        else:
            node_dict[track["start_cluster"][0]] = [start_point]
        if track["end_cluster"][0] in node_dict.keys():
            node_dict[track["end_cluster"][0]].append(end_point)
        else:
            node_dict[track["end_cluster"][0]] = [end_point]
        if (track["start_cluster"][0], track["end_cluster"][0]) in edge_dict.keys():
            edge_dict[(track["start_cluster"][0], track["end_cluster"][0])] += 1
        else:
            edge_dict[(track["start_cluster"][0], track["end_cluster"][0])] = 1
    # create average position per cluster
    node_pos = {}
    for key, value in node_dict.items():
        x = 0
        y = 0
        for pos in value:
            x += pos.x
        for pos in value:
            y += pos.y
        x = x/len(value)
        y = y/len(value)
        node_pos[key] = (x,y)
    G=nx.Graph()
    for key, pos in node_pos.items():
        G.add_node(key,pos=(pos[0], pos[1]))
    for key, value in edge_dict.items():
        G.add_edge(key[0], key[1], weight=value)
    return G

def find_starting_point_on_graph(G, orig_point, target_point): # ToDo use interim points on edges
    nearest_node_to_target = ox.distance.nearest_nodes(G,orig_point.x, orig_point.y)
    nearest_node_to_origin = ox.distance.nearest_nodes(G,target_point.x, target_point.y)
    return nearest_node_to_origin, nearest_node_to_target

def routing_on_graph(G, origin_node, target_node):
    route = ox.shortest_path(G, origin_node, target_node, weight="travel_time")
    if route is None:
        return [], LineString()
    if len(route) == 1:
        return route, LineString()
    nodes, edges = ox.graph_to_gdfs(G)
    route_nodes = nodes.loc[route]
    route_line = LineString(route_nodes['geometry'].tolist())
    gdf1 = gpd.GeoDataFrame(geometry=[route_line], crs=ox.settings.default_crs)
    if route is None:
        print("failure in routing")
    return  route, gdf1