# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 18:08:18 2022

@author: remns
"""

#TODO:  all functions used within get_time_data() would be better hidden.
#       For that purpose, creating a class and setting them as private methods
#       could be a good solution

import requests
import json
import os

import numpy as np
import geopy
from geopy import Nominatim
import geopandas as gpd
import pandas as pd
import seaborn as sns
import folium
from folium.plugins import HeatMap
import webbrowser
from shapely.geometry import *
from shapely.ops import split

geopy.geocoders.options.default_user_agent = "city_scraper"

geolocator_global = Nominatim()
REFERENCE_CITY = 'Munich'
REFERENCE_COORDS = geolocator_global.geocode(REFERENCE_CITY)[1]
DESTINATION = (48.093756, 11.357966)

def get_API_key():
    try:
        with open('API_key.txt', 'r') as f:
            key = f.read()
        return key
    except FileNotFoundError:
        wd = os.getcwd()
        print('You need to place a API_key.txt file containing in the working'+\
              ' directory (' + wd + ')')
        return None

def coords_to_str_for_API(coords):
    """Google API uses latitude / longitude format"""
    return str(coords[0]) + ',' + str(coords[1])

def change_global_ref_location(new_ref):
    global REFERENCE_COORDS
    global REFERENCE_CITY
    geolocator = Nominatim()
    REFERENCE_COORDS = geolocator.geocode(new_ref)[1]
    REFERENCE_CITY = new_ref

def display_map(m, name="map"):
    # In order to visualize map on browser, first save it as html
    name = name + ".html"
    m.save(name)
    webbrowser.open(name)

def get_routes_google_maps(destination_coords, origin_coords):
    """Returns a dictionary with the content of the query to Google Maps.
    Input destination coordinates and origin coordinates.
    Coordinates format; Latitude / longitude"""
    location = coords_to_str_for_API(origin_coords)
    dest = coords_to_str_for_API(destination_coords)
    key = get_API_key()
    url = 'https://maps.googleapis.com/maps/api/directions/json?origin=' +\
        location + '&destination=' + dest + '&alternatives=true&mode=transit&key='+ key
    r = requests.get(url)
    options = json.loads(r.content)
    return options

def build_routes_dict(json_content):
    """Returns a dict containing all the routes found and their relevant
    information (line names and travel times"""
    routes = json_content['routes']
    routes_dict = {}
    for r_num, route in enumerate(routes):
        routes_dict[r_num] = {}
        for leg in route['legs']:
            duration = leg['duration']['value']
            routes_dict[r_num]['time'] = duration
            lines = []
            for step in leg['steps']:
                try:
                    lines.append(step['transit_details']['line']['short_name'])
                except KeyError:
                    continue
            routes_dict[r_num]['lines'] = lines
    return routes_dict

def faster_route(routes_dict):
    """Returns the faster option for each origin-destination route in the
    dictionary"""
    min_route = list(routes_dict.keys())[0]
    min_time = routes_dict[min_route]['time']
    for route_id, data in routes_dict.items():
        if data['time'] < min_time:
            min_route, min_time = route_id, data['time']
    return routes_dict[min_route]

def get_time_data(destination, points, test_run=True, verbose=False):
    """Perform route queries for all origin "points" to "destination.
    Returns the data in a Dataframe and a list of points for which no result
    was obtained.
    WARNING: enter destination as lat-lon (Google API format)
    
    @param
        test_run: True means that time values will be randomly generated instead
                of actually calling the Google Maps API"""
    lon, lat, time, line_name, changes = [],[],[],[],[]
    failed_points = []
    for i, point in enumerate(points):
        if i%50 ==0 and verbose:
            print(i, ' points')
        lon.append(point.x)
        lat.append(point.y)
        if not test_run:
            possible_routes =  get_routes_google_maps(destination, (point.y, point.x))
            formatted_routes = build_routes_dict(possible_routes)
            try:
                best_route = faster_route(formatted_routes)
                time.append(best_route['time'])
                line_name.append(best_route['lines'])
                changes.append(len(best_route['lines']))
            except IndexError:
                failed_points.append(point)
                print('Fail number ', len(failed_points))
        else:
            if i==0:
                print("This is a TEST RUN. Transport times and lines will be randomly "
                  "generated, instead of calling Google Maps API")
            time.append(np.random.randint(10,60))
            line_name.append('DummyLine')
            changes.append(np.random.randint(0,4))
    data = pd.DataFrame(list(zip(lon, lat, time, line_name, changes)),
                        columns=['lon', 'lat', 'time', 'line_name', 'changes'])
    return data, failed_points

def sample_points_in_region(polyg, num_points = 2000, random=True):
    """
    TODO --> I don't like the current random mode. I would first do the regular
    sampling and then use the random sampling to reach num_points
    Return a list of points sampled from inside the given region ("polyg")
    @params
        polyg: Shapely Polygon object that defines the region to sample in.
        num_points: number of points to sample.
        random  = True: uniform random sampling within the polygon
                = False: return less than num_points points, only in grid pattern"""
    points = []
    minx, miny, maxx, maxy = polyg.bounds
    if not random:
        x_length = maxx - minx
        y_length = maxy - miny
        n_points_y = np.sqrt(num_points * y_length / x_length)
        n_points_x = num_points  / n_points_y
        n_points_x, n_points_y = round(n_points_x), round(n_points_y)
        x_coords = np.linspace(minx, maxx, n_points_x)
        y_coords = np.linspace(miny, maxy, n_points_y)
        for x in x_coords:
            for y in y_coords:
                p = Point(x, y)
                if polyg.contains(p):
                    points.append(p)
    print('Sampling ', num_points-len(points), ' random points')
    while len(points) < num_points:
        p = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
        if polyg.contains(p):
             points.append(p)
    return points

def create_polygon(center_coords, radius=0.075, shape='circle', center_format='latlon',
                   polygon_format='lonlat'):
    """TODO
    polygon_format needs to be lonlat to work with folium map
    """
    
    if center_format != polygon_format:
        center_coords = center_coords[::-1]
        
    if shape=='circle':
        polygon = Point(center_coords).buffer(radius)
        
    return polygon

def sample_and_draw_points_in_region(polyg, m, num_points = 800):
    """Sample random points within the reguion and draw them on the map.
    Returns both the map and the points list"""
    points = []
    while len(points) < num_points:
        minx, miny, maxx, maxy = polyg.bounds
        p = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
        if polyg.contains(p):
            points.append(p)
            m = draw_shapely_obj_on_map(p, m)
    return m, points

def geopandas_polygon(polyg, crs_code='epsg:4326'):
    crs = {'init': crs_code}
    geopolygon = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[polyg])
    return geopolygon

def draw_shapely_obj_on_map(shap_obj, map_obj=None, style_dict=None):
    """Draw the given Shapely object on a map.
    WARNING:    The polygon needs to be passed in lon-lat format.
    NOTE:       The map is created using lat-lon format for its centering
        """
    if not map_obj:
        center = [shap_obj.centroid.y, shap_obj.centroid.x]
        map_obj = create_map(center)
    if style_dict:
        style_function = lambda x:style_dict
    else:
        style_function = None
    folium.GeoJson(shap_obj,style_function=style_function).add_to(map_obj)
    folium.LatLngPopup().add_to(map_obj)
    return map_obj

def create_map(center, zoom_start=13, tiles='cartodbpositron'):
    m = folium.Map(center, zoom_start=zoom_start, tiles=tiles)
    return m

def add_heatmap(m, data):
    """Draw the Leaflet HeatMap on a map. Only works as a COUNTING Heatmap"""
    #times = (data.loc[:,'time'] - data.time.min()) / (data.time.max() - data.time.min())
    #heatmap = HeatMap(list(zip(data['lat'],data['lon'],times)),min_opacity=0.2, max_val=data["time"].max(), min_val=data["time"].min(), radius=25, blur=20)
    heatmap = HeatMap(list(zip(data['lat'],data['lon'])),min_opacity=0.2, radius=25, blur=20)
    heatmap.add_to(m)
    return m

def adapt_data_for_heatmap(data, scale_steps=10):
    """Useless function trying to make Leaflet Heatmapwork for my purpose"""
    max_time = data.time.max()
    min_time = data.time.min()
    step_size = (max_time - min_time) / scale_steps
    heat_data = data.copy()
    heat_data['n_points'] = (heat_data['time'] - min_time) / step_size
    heat_data['n_points'] = heat_data['n_points'].astype(int)
    heat_data['n_points'] = heat_data['n_points'] + 1
    heatmap_data = np.zeros((heat_data['n_points'].sum(),2))
    i = 0
    for row in heat_data.iterrows():
        n_points = row[1]['n_points']
        heatmap_data[i:i+n_points] = row[1][['lat', 'lon']]
        i+=n_points
    heatmap_df = pd.DataFrame(heatmap_data, columns=['lat', 'lon'])
    return heatmap_df

def outer_rectangle_of_polygon(polygon):
    """Just in case I am too lazy to search for the .bounds attribute in the docs"""
    return polygon.bounds

def orthogonal_grid_on_polygon_bounds(polygon, x_lines=38, y_lines=21):
    """Returns a collection equally spaced of horizontal and vertical lines that
    form a grid whose outer limits are the sides of the rectangle bounding a polygon.
    @params
        polygon: a Shapely Polygon object
        x_lines, y_lines: an int specifying how many horizontal and vertical lines
                        the grid must have.
    @returns
        horizontal_lines, vertical_lines: two lists of Shapley LineString objects
    """
    bounds = outer_rectangle_of_polygon(polygon)
    x_coords = np.linspace(bounds[0], bounds[2], x_lines)
    y_coords = np.linspace(bounds[1], bounds[3], y_lines)
    vertical_lines = []
    for x in x_coords:
        vertical_lines.append(LineString([(x,max(y_coords)), (x, min(y_coords))]))
    horizontal_lines = []
    for y in y_coords:
        horizontal_lines.append(LineString([(min(x_coords), y), (max(x_coords), y)]))
        
    return horizontal_lines, vertical_lines

def map_partition(mypol, horizontal_lines, vertical_lines):
    """Returns a collection of Shapely Polygon objects which are the pieces of
    the grid formed by the Polygon "mypol" and the LineStrings "horizontal_lines"
    and vertical_lines"""
    # Divide the main polygon in the horizontal polygons delimited by horizontal_lines
    hor_collection = []
    curr_pol = mypol
    for line in horizontal_lines:
        collection = list(split(curr_pol,line))
        if len(collection) > 1:
            first = collection[0]
            second = collection[1]
            if first.centroid.y > second.centroid.y:
                curr_pol = first
                hor_collection.append(second)
            else:
                curr_pol = second
                hor_collection.append(first)
        elif len(collection) == 1:
            curr_pol = collection[0]
            hor_collection.append(curr_pol)
    
    # Divide each horizontal polygon in smaller polygons delimited by vertical_lines
    ver_collection = []      
    for i,hor_pol in enumerate(hor_collection):
        for line in vertical_lines:
            collection = list(split(hor_pol,line))
            # Add to the collection the polygon at the left of the line
            ver_collection += [poly for poly in collection if poly.centroid.x < line.coords[0][0]]
            try:
                # Update the main horizontal polygon that is being divided.
                # In the next iteration, it is the polygon at the right of the line
                hor_pol = [poly for poly in collection if poly.centroid.x > line.coords[0][0]][0]
            except IndexError:
                break
    return ver_collection
   
def display_map_partition(m, polygon, h_lines, v_lines):
    for v_line in v_lines:
        m=draw_shapely_obj_on_map(v_line,m)
    for h_line in h_lines:
        m=draw_shapely_obj_on_map(h_line,m)
    m=draw_shapely_obj_on_map(polygon,m)
    display_map(m)

def rgb_to_hextriplet(colors, adapt_from_matplotlib=False):
    if adapt_from_matplotlib:
        colors = [tuple([int(part*255) for part in color]) for color in colors]
    return ['#%02x%02x%02x' % color for color in colors]
    
def build_data_for_polygon_heatmap(polygon_collection, data):
    """Returns a GeoDataFrame containing the Polygon grid pieces and their corresponding
    time"""
    gdf_pols = gpd.GeoDataFrame(gpd.GeoSeries(polygon_collection), columns=['polygons'])
    times = np.zeros((len(gdf_pols),1))
    # Loop over the polygon pieces to assign them a value
    for i, pol in enumerate(polygon_collection):
        ## Get its bounding box coordinates (all polygons were created by an
        #orthogonal partition, so the bounding box is representative of the area they cover)
        # If polygons where not orthogonal to the grid's axes, getting their contour
        # coordinates would be better
        #coords = list(pol.exterior.coords)
        
        minx, miny, maxx, maxy = pol.bounds
        
        ## Find data points on the polygon's boundary and interior
        # Filter nearby points in data
        max_above = maxy + (maxy-miny)/2
        min_below = miny - (maxy-miny)/2
        min_left = minx - (maxx-minx)/2
        max_right = maxx + (maxx-minx)/2
        time = 0
        while not (time > 0):
            nearby_filter_x = (data['lon'] > min_left) & (data['lon'] < max_right)
            nearby_filter_y = (data['lat'] > min_below) & (data['lat'] < max_above)
            nearby_points = data[nearby_filter_x & nearby_filter_y]
        
            # Average of the time value of the nearby points
            time = nearby_points['time'].mean()
            
            # Push the boundaries around the polygon further
            max_above += (maxy-miny)/4
            min_below -= (maxy-miny)/4
            min_left -= (maxx-minx)/4
            max_right += (maxx-minx)/4
            
        times[i,:1] = time
        
    gdf_pols['time'] = times
    return gdf_pols
    
def draw_heatmap(m, data, colors=None, borders=None):
    """Draws the heatmap with te information defined in "data" (Polygon-time pairs).
    If specified, "borders" must be a Polygon.
    """
    def assign_threshold(time, thresholds):
        indices = np.where(time >= thresholds)
        if len(indices[0])==0:
            print(indices[0], time, thresholds)
        if indices[0][-1] == len(thresholds)-1:
            
            print(time, indices)
            return indices[0][-2]
        return indices[0][-1]
    
    if colors==None:
        colors = sns.color_palette("Spectral", as_cmap=False, n_colors=10).as_hex()
        colors = colors[::-1]
    n_colors = len(colors)
    time_range = data.time.max() - data.time.min()
    interval = time_range / (n_colors)
    thresholds = np.array([data.time.min()+interval*i for i in range(n_colors+1)])
    print(len(thresholds))
    for row in data.iterrows():
        polygon = row[1]['polygons']
        time = row[1]['time']
        try:
            color = colors[assign_threshold(time, thresholds)]
        except:
            print(assign_threshold(time, thresholds))
        style_dict = {'fillOpacity':0.4,
                          'fillColor':color,
                          'color': '#00000000'}   
        m = draw_shapely_obj_on_map(polygon, m, style_dict)
    
    if borders:
        m=draw_shapely_obj_on_map(borders,m, {'fillColor':'#00000000'})
    return m


if __name__ == '__main__':     
    # Get seaborn color palette for sequential color map ('YlOrBr', for example)
    colors = sns.color_palette("YlOrBr", as_cmap=False).as_hex
    
    points = [(48.161340, 11.513443),(48.153173, 11.499893),(48.136253, 11.495043),\
              (48.112565, 11.501909),(48.115823, 11.556677),(48.141787, 11.594572),\
              (48.161025, 11.585797),(48.167244, 11.573912),(48.161738, 11.553209)]
        
    lonlat_points = [(point[1],point[0]) for point in points]
    points = lonlat_points

    mypol = Polygon(points)
    
    m = create_map(REFERENCE_COORDS)
    data_points = sample_points_in_region(mypol)
    data,_ = get_time_data(DESTINATION, data_points, test_run=False)
    
    horizontal_lines, vertical_lines = orthogonal_grid_on_polygon_bounds(mypol)
    poly_collection = map_partition(mypol, horizontal_lines, vertical_lines)
    gdf_pols = build_data_for_polygon_heatmap(poly_collection, data)
    m = draw_heatmap(m, gdf_pols)
    display_map(m, 'map_result')
    

