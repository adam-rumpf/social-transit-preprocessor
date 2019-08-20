"""Preprocessing procedures for Chicago/CTA data."""

import numpy as np
import geopy.distance as gpd
import scipy.cluster.vq as spc
import operator
import statistics

#==============================================================================
# Parameters
#==============================================================================

# Census data input/output files
tract_data = "chicago_data/raw/census/census_tracts_list_17.txt"
community_names = "chicago_data/raw/census/community_names.txt"
community_conversion = ("chicago_data/raw/census/tract_to_community.txt")
population_raw = "chicago_data/intermediate/population_raw.txt"
population_clustered = "chicago_data/intermediate/population.txt"

# Primary care facility input/output files
facility_in = "chicago_data/raw/facility/facility_address.txt"
facility_out = "chicago_data/intermediate/facility.txt"

# Transit network parameters
k_clusters = 1000 # number of stops after clustering (may be slightly less)
stop_data = "chicago_data/raw/network/stops.txt"
stop_list = "chicago_data/intermediate/all_stops.txt"
trip_data = "chicago_data/raw/network/trips.txt"
route_data = "chicago_data/raw/network/routes.txt"
time_data = "chicago_data/raw/network/stop_times.txt"
stop_cluster_file = "chicago_data/intermediate/clustered_stops.txt"
stop_cluster_lookup = "chicago_data/intermediate/cluster_lookup.txt"
line_nodes = "chicago_data/intermediate/line_nodes.txt"
line_arcs = "chicago_data/intermediate/line_arcs.txt"
transit_data = "chicago_data/intermediate/transit_data.txt"

# Output network file parameters
nid_stop = 0 # stop node type
nid_board = 1 # boarding node type
nid_pop = 2 # population center node type
nid_fac = 3 # primary care facility node type
aid_line = 0 # line arc type
aid_board = 1 # boarding arc type
aid_alight = 2 # alighting arc type
aid_walk = 3 # standard walking arc type
aid_walk_health = 4 # walking arc type to connect pop centers and facilities
final_arc_data = "chicago_data/processed/arc_data.txt"
final_node_data = "chicago_data/processed/node_data.txt"
final_transit_data = "chicago_data/processed/transit_data.txt"
mile_walk_time = 0.25*(5280/60) # minutes to walk 1 mile (given 4 ft/sec speed)

# OD matrix parameters
bus_trip_mean = 35.53 # mean user bus trip time
train_trip_mean = 57.29 # mean user train trip time
od_data_month = 10 # month for OD data
od_data_year = 2012 # year for OD data
gamma_std_dev = 20.0 # standard deviation of gamma distribution
od_data_bus = ("chicago_data/raw/od/CTA_-_Ridership_-_Avg._Weekday_Bus_Stop_"+
               "Boardings_in_October_2012.csv")
od_data_train = ("chicago_data/raw/od/CTA_-_Ridership_-__L__Station_Entries_-_"
               +"Monthly_Day-Type_Averages___Totals.csv")
cluster_boarding = "chicago_data/intermediate/stop_boardings.txt"
mode_boarding = "chicago_data/intermediate/mode_boardings.txt"
all_pairs_distance = "chicago_data/intermediate/distances.txt"
final_od_data = "chicago_data/processed/od_data.txt"

# Misc. data
finite_infinity = 10000000000 # large value to use in place of infinity
cta_fare = 2.25 # fare to board any CTA line
bus_capacity = 39 # seating capacity of New Flyer D40LF
train_capacity = 6 * 38 # seating capacity of six 5000-series cars
type_bus = 0 # type ID to use for buses
type_train = 1 # type ID to use for trains
type_remap = {3: type_bus, 1: type_train} # replacements for GTFS vehicle types
cost_bus = -1 # operating cost for a bus
cost_train = -1 # operating cost for a train
op_coef_names = ["Operating_Cost", "Fares"] # operator cost term names
op_coef = [1, cta_fare] # operator cost term coefficients
us_coef_names = ["Riding", "Walking", "Waiting"] # user cost term names
us_coef = [1, 1, 1] # user cost term coefficients
assignment_epsilon = -1 # assignment model cutoff epsilon
assignment_max = 1000 # maximum assignment model iterations
latency_names = ["alpha", "beta"] # list of latency function parameter names
alpha = 4.0
beta = (2*alpha-1)/(2*alpha-2)
latency_parameters = [alpha, beta] # list of latency function parameters
obj_names = ["FCA_Cutoff", "Gravity_Falloff"] # obj function parameter names
obj_parameters = [30.0, 1.0] # objective function parameters
misc_names = ["Horizon"] # misc parameter names
misc_parameters = [1440.0] # misc parameters
vehicle_file = "chicago_data/processed/vehicle_data.txt"
oc_file = "chicago_data/processed/operator_cost_data.txt"
uc_file = "chicago_data/processed/user_cost_data.txt"
assignment_file = "chicago_data/processed/assignment_data.txt"
objective_file = "chicago_data/processed/objective_data.txt"
problem_file = "chicago_data/processed/problem_data.txt"

#==============================================================================
# Functions
#==============================================================================

#------------------------------------------------------------------------------
def distance(x, y, taxicab=False):
    """Calculates geodesic distance (mi) between two tuples of coordinates.

    Accepts an optional argument indicating whether to use taxicab distance
    instead of the default Euclidean distance.
    """

    if taxicab == False:
        return gpd.geodesic(x, y).miles
    else:
        return min(gpd.geodesic(x, (x[0],y[1])).miles +
                   gpd.geodesic((x[0],y[1]), y).miles, gpd.geodesic(x,
                               (y[0],x[1])).miles + gpd.geodesic((y[0],x[1]),
                               y).miles)

#------------------------------------------------------------------------------
def absolute_time(t_string):
    """Changes 24-hour time string 'hh:mm:ss' to float mins since midnight."""

    num = [float(n) for n in t_string.split(':')]

    return 60*num[0] + num[1] + (num[2]/60)

#------------------------------------------------------------------------------
def census_processing(tract_file, name_file, conversion_file, output_file_raw,
                      output_file_clustered):
    """Preprocessing for census data.

    Requires five file names in order: census tract gazetteer, community area
    names, census tract to Chicago community area conversion, raw population
    center output file, and clustered population center output file.

    In order to reduce the number of population centers we cluster census tract
    data by Chicago community area. The output file should include the total
    population of each community area along with the population-weighted
    centroid of the community, calculated based on the tract-level populations
    and centroids.
    """

    # Initialize community dictionary, which will be indexed by community
    # number and will contain a list of each community name, total population,
    # and population-weighted lat/lon.
    community = {}
    with open(name_file, 'r') as f:
        i = -1
        for line in f:
            i += 1
            if i > 0:
                # Skip comment line
                dum = line.split('\t')
                community[int(dum[1])] = [dum[0], 0.0, 0.0, 0.0]

    # Create conversion dictionary, which will associate each tract ID with a
    # community number
    conversion = {}
    with open(conversion_file, 'r') as f:
        i = -1
        for line in f:
            i += 1
            if i > 0:
                # Skip comment line
                dum = line.split('\t')
                # Use GEOID as key (17 = Illinois, 031 = Cook County)
                conversion["17031"+dum[1].strip()] = int(dum[0])

    # Create tract-level dictionary, which will be indexed by tract number and
    # will contain the population, lat, and lon.
    tract = {}
    with open(tract_file, 'r') as f:
        i = -1
        for line in f:
            i += 1
            if i > 0:
                # Skip comment line
                dum = line.split('\t')
                t = dum[1].strip() # tract number
                if t in conversion:
                    tract[t] = [int(dum[2]), float(dum[8]), float(dum[9])]

    # Output a list of the tract-level population centers.
    with open(output_file_raw, 'w') as f:
        print("GEOID\tPopulation\tLat\tLon", file=f)
        for t in tract:
            print(str(t)+"\t"+str(tract[t][0])+"\t"+str(tract[t][1])+"\t"+
                  str(tract[t][2]), file=f)

    # Calculate total population and weighted average coordinates for each
    # community area.
    for t in tract:
        com = conversion[t] # community ID associated with tract t
        pop = tract[t][0] # population of tract t
        community[com][1] += pop
        community[com][2] += pop*tract[t][1] # pop-weighted lat
        community[com][3] += pop*tract[t][2] # pop-weighted lon

    # Divide community center coordinates by total population.
    for com in community:
        community[com][2] /= community[com][1]
        community[com][3] /= community[com][1]

    # Output a list of the community area-level population centers.
    with open(output_file_clustered, 'w') as f:
        print("Number\tPopulation\tLat\tLon", file=f)
        for com in community:
            print(str(com)+"\t"+str(community[com][1])+"\t"+
                  str(community[com][2])+"\t"+str(community[com][3]), file=f)

#------------------------------------------------------------------------------
def facility_processing(address_file, output_file):
    """Preprocessing for facility data.

    Requires the names of the raw facility data file and the output file,
    respectively.

    The facility input file contains alternating lines of facility names/
    addresses and coordinates. This script simply converts the file to a table
    of names, latitude, and longitude.
    """

    # Initialize a facility dictionary, indexed by name and containing lat/lon.
    facility = {}
    with open(address_file, 'r') as f:
        i = -1
        for line in f:
            i += 1
            if i > 0:
                # Skip comment line
                if i % 2 == 1:
                    # Odd lines contain facility names
                    name = line.strip().split('\t')[0].replace(' ', '_')
                else:
                    # Even lines contain facility coordinates
                    coords = line.strip("()\n").split(',')
                    facility[name] = [float(coords[0]), float(coords[1])]

    # Output a list of facility names and coordinates.
    with open(output_file, 'w') as f:
        print("Name\tLat\tLon", file=f)
        for fac in facility:
            print(str(fac)+"\t"+str(facility[fac][0])+"\t"+
                  str(facility[fac][1]), file=f)

#------------------------------------------------------------------------------
def stop_cluster(stop_data, k, output_file=None, lookup_file=None):
    """Conversion of GTFS stops to stop clusters.

    Requires the GTFS stop data file and a specified number of clusters,
    respectively. Prints the model's distortion (mean Euclidean distance
    between stop coordinate and assigned cluster centroid) and returns the list
    of cluster centroids and a lookup table associating each stop ID with an
    index from the stop cluster file.

    The optional keyword argument 'output_file' determines whether or not to
    print the clustered stops to a file. It defaults to 'None', in which case
    nothing is written. If given a file name, it will write the result to that
    file.

    In order to reduce the number of stops in our constructed network, we
    begin by clustering the listed stops into k clusters. Specifically we use
    the SciPy implementation of k-means on the geographic coordinates of the
    stops. These means represent the geographic centroids of each collection of
    stops.

    These clustered stops will be used as standins for the "real" stops for
    most purposes in the constructed network. Any other stop-specific data will
    be remapped to the nearest of the clustered stops.
    """

    # Initialize lists of stop coordinates and IDs
    stops = []
    stop_coords = []
    with open(stop_data, 'r') as f:
        i = -1
        for line in f:
            i += 1
            if i > 0:
                # Skip comment line
                dum = line.split(',')
                stops.append(dum[0])
                # We grab elements relative to the end of the line, since some
                # stop names contain commas.
                stop_coords.append([float(dum[-5]), float(dum[-4])])

    # Evaluate k-means
    codebook, distortion = spc.kmeans(stop_coords, k)
    codebook = codebook.tolist()
    print("k-means distortion: "+str(distortion))

    # Write output (if requested)
    if output_file != None:
        with open(output_file, 'w') as f:
            print("ID\tLat\tLon", file=f)
            i = 0
            for cb in codebook:
                print(str(i)+"\t"+str(cb[0])+"\t"+str(cb[1]), file=f)
                i += 1

    # For each stop ID, find the nearest clustered stop ID and output a lookup
    # table (if requested)
    if lookup_file != None:
        with open(lookup_file, 'w') as f:
            print("StopID\tClusterID", file=f)
            for i in range(len(stops)):
                print("Stop "+str(i+1)+" of "+str(len(stops)))
                # Find codebook ID that minimizes pairwise distance
                cb = codebook.index(min(codebook, key=lambda cs:
                    distance(stop_coords[i], cs)))
                print(str(stops[i])+"\t"+str(cb), file=f)

    return codebook

#------------------------------------------------------------------------------
def stop_cluster_measure(stop_file, cluster_file, lookup_file):
    """Calculates the distortion of a given cluster assignment.

    Requires a file with the original stop coordinates, the cluster
    coordinates, and the cluster lookup table.

    Prints statistics regarding the distribution of distances from each stop to
    its assigned cluster centroid.
    """

    # Build dictionary linking stop IDs to cluster IDs
    cluster = {}
    with open(lookup_file, 'r') as f:
        i = -1
        for line in f:
            i += 1
            if i > 0:
                # Skip comment line
                dum = line.split()
                cluster[dum[0]] = dum[1]

    # Build dictionary of cluster centroids
    centroid = {}
    with open(cluster_file, 'r') as f:
        i = -1
        for line in f:
            i += 1
            if i > 0:
                # Skip comment line
                dum = line.split()
                centroid[dum[0]] = (float(dum[1]), float(dum[2]))

    # Calculate list of pairwise distances
    dist = []
    with open(stop_file, 'r') as f:
        i = -1
        for line in f:
            i += 1
            if i > 0:
                # Skip comment line
                dum = line.split()
                stop = dum[0]
                coords = (float(dum[1]), float(dum[2]))
                dist.append(distance(centroid[cluster[stop]], coords))

    # Print results
    print("Statistics for distances between stops and cluster centroids (mi):")
    print("Mean =    "+str(statistics.mean(dist)))
    print("Median =  "+str(statistics.median(dist)))
    print("Std Dev = "+str(statistics.pstdev(dist)))
    print("Max =     "+str(max(dist)))

#------------------------------------------------------------------------------
def transit_processing(stop_file, trip_file, route_file, stop_time_file,
                       node_output_file, arc_output_file, route_output_file,
                       cluster_file=None, cluster_lookup=None):
    """Preprocessing for transit network data.

    Requires the following file names (respectively): GTFS stop data, GTFS trip
    data, GTFS route data, GTFS stop time data, output file for node list,
    output file for arc list, and output file for line info.

    There are also optional keyword arguments to specify the clustered stop
    file and the cluster lookup table. 'cluster_file' and 'cluster_lookup' can
    be given the names of existing file to read from, otherwise defaulting to
    'None'.

    The node and arc output files treat the cluster IDs as the stop node IDs,
    and include the boarding nodes, boarding arcs, alighting arcs, and line
    arcs for each line, along with the correct base travel times.
    """

    nodenum = -1 # current node ID
    arcnum = -1 # current arc ID

    # Write headers for arc and transit files
    with open(route_output_file, 'w') as f:
        print("ID\tName\tType\tFleet\tCircuit\tScaling", file=f)
    with open(arc_output_file, 'w') as f:
        print("ID\tType\tLine\tTail\tHead\tTime", file=f)

    # Initialize dictionaries linking cluster IDs to coordinates, and linking
    # route IDs to clustered stop IDs
    clusters = {}
    lookup = {}

    # Read cluster file while writing the initial node file
    with open(node_output_file, 'w') as fout:
        print("ID\tName\tType\tLine", file=fout)
        with open(cluster_file, 'r') as fin:
            i = -1
            for line in fin:
                i += 1
                if i > 0:
                    # Skip comment line
                    dum = line.split()
                    clusters[int(dum[0])] = [float(dum[1]), float(dum[2])]
                    if int(dum[0]) > nodenum:
                        nodenum = int(dum[0])
                    print(str(nodenum)+"\tStop"+str(nodenum)+"\t"+
                          str(nid_stop)+"\t-1", file=fout)

    with open(cluster_lookup, 'r') as f:
        i = -1
        for line in f:
            i += 1
            if i > 0:
                # Skip comment line
                dum = line.split()
                lookup[dum[0]] = int(dum[1])

    # Create lists of all route IDs and types
    routes = []
    vehicle_types = []
    with open(route_file, 'r') as f:
        i = -1
        for line in f:
            i += 1
            if i > 0:
                # Skip comment line
                dum = line.split(',')
                routes.append(dum[0])
                vehicle_types.append(dum[-4])

    # Create a dictionary to link trip IDs to route IDs
    trip_to_route = {}
    with open(trip_file, 'r') as f:
        i = -1
        for line in f:
            i += 1
            if i > 0:
                # Skip comment line
                dum = line.split(',')
                trip_to_route[dum[2]] = dum[0]

    # Collect the stops for each route. The GTFS file is exceptionally long and
    # cannot efficiently be maintained in memory all at once, so instead we
    # will process each route one-at-a-time, reading the GTFS file and only
    # collecting stops relevant to that particular route.
    for r in range(len(routes)):
        # Initialize route's earliest and latest known times, a counter of the
        # total number of stop visits, list of all of unique non-clustered
        # stops, and a dictionary of trip IDs. This dictionary will contain a
        # list of lists of the stop IDs, arrival time, departure time, and stop
        # sequence of all stops on that trip.
        earliest = np.inf
        latest = -np.inf
        visits = 0
        unique_stops = []
        trip_stops = {}

        # Read stop time file
        with open(stop_time_file, 'r') as f:
            for line in f:
                dum = line.split(',')
                if dum[0] in trip_to_route:
                    if trip_to_route[dum[0]] == routes[r]:
                        # The current line represents the current route's stop
                        if (dum[0] in trip_stops) == False:
                            # Create a new list for a new trip
                            trip_stops[dum[0]] = []
                        trip_stops[dum[0]].append([dum[3],
                                   absolute_time(dum[1]),
                                   absolute_time(dum[2]),
                                   int(dum[4])])
                        visits += 1
                        if (dum[3] in unique_stops) == False:
                            unique_stops.append(dum[3])

        # Unique clustered stops
        unique_clusters = list(set([lookup[u] for u in unique_stops]))

        # Initialize a weighted arc list. This will be indexed by tuples of
        # cluster IDs, and each entry will be a list of the known travel times
        # for that arc.
        trip_arcs = {}

        #----------------------------------------------------------------------
        # Trip loop begin
        #
        for t in trip_stops:
            # Sort the stops in ascending order of sequence number
            trip_stops[t].sort(key=operator.itemgetter(3))

            # Initialize a graph representation of the trip using a predecessor
            # and a successor dictionary. This will be indexed by cluster ID,
            # and each entry will be a list of the cluster IDs of that node's
            # predecessors and successors.
            trip_pred = {}
            trip_succ = {}

            # Initialize a dictionary of loops found on the trip. Due to the
            # stop clustering, if both endpoints of a travel link are mapped to
            # the same cluster, we will get loops in our graph. This dictionary
            # is indexed by the cluster IDs of each loop endpoint, and contains
            # the total travel time over all such loops.
            loop_list = {}

            #------------------------------------------------------------------
            # Stop loop begin
            #
            for i in range(len(trip_stops[t])-1):
                # Cluster IDs of stop endpoints
                u, v = (lookup[trip_stops[t][i][0]],
                        lookup[trip_stops[t][i+1][0]])

                # Create new node pred/succ entries for new nodes
                if (u in trip_pred) == False:
                    trip_pred[u] = []
                if (u in trip_succ) == False:
                    trip_succ[u] = []
                if (v in trip_pred) == False:
                    trip_pred[v] = []
                if (v in trip_succ) == False:
                    trip_succ[v] = []

                # Append non-loop endpoints to each others' pred/succ lists
                if u != v:
                    if (v in trip_succ[u]) == False:
                        trip_succ[u].append(v)
                    if (u in trip_pred[v]) == False:
                        trip_pred[v].append(u)

                # Arrival/departure times
                u_arrive = trip_stops[t][i][1]
                v_arrive = trip_stops[t][i+1][1]
                v_depart = trip_stops[t][i+1][2]

                # Update earliest/latest known times
                if u_arrive < earliest:
                    earliest = u_arrive
                if v_depart > latest:
                    latest = v_depart

                # Arc travel time is difference between consecutive arrivals
                link_time = v_arrive - u_arrive

                # Adjust for the issue of rolling past midnight
                if link_time < 0:
                    link_time += 1440.0 # minutes per day
                    if v_depart + 1440.0 > latest:
                        latest = v_depart + 1440.0

                # Handle the case of a loop
                if u == v:
                    # Add new loops to the loop list
                    if (u in loop_list) == False:
                        loop_list[u] = 0.0
                    # Add the loop's travel time to the loop list total
                    loop_list[u] += link_time

                # Create/append a new arc entry
                if ((u, v) in trip_arcs) == False:
                    trip_arcs[(u, v)] = []
                trip_arcs[(u, v)].append(link_time)

            #
            # Stop loop end
            #------------------------------------------------------------------

            # Distribute each loop's total travel time equally among all
            # incoming/outgoing arcs and delete the loop
            for u in loop_list:
                frac = loop_list[u]/(len(trip_pred[u])+len(trip_succ[u]))
                for v in trip_pred[u]:
                    trip_arcs[(v, u)][-1] += frac
                for v in trip_succ[u]:
                    trip_arcs[(u, v)][-1] += frac
                del trip_arcs[(u, u)]

        #
        # Trip loop end
        #----------------------------------------------------------------------

        # Compute various line attributes for the current route

        # Average the arc times
        for a in trip_arcs:
            trip_arcs[a] = np.mean(trip_arcs[a])

        # Average the visits per stop
        visits_per_stop = visits / len(unique_stops)

        # Daily time horizon
        horizon = min(latest - earliest, 1440.0)

        # Fraction of day during which the route runs
        daily_fraction = horizon / 1440.0

        # Average frequency
        frequency = visits_per_stop / horizon

        # Average time for a vehicle to complete one circuit
        circuit = 0.0
        for a in trip_arcs:
            circuit += trip_arcs[a]

        # Fleet size
        fleet = np.ceil(frequency*circuit)

        # Write route attributes to file
        with open(route_output_file, 'a') as f:
            # ID, Name, Type, Fleet, Circuit, Scaling
            print(str(r)+"\t"+str(routes[r])+"\t"+str(vehicle_types[r])+"\t"+
                  str(fleet)+"\t"+str(circuit)+"\t"+str(daily_fraction),
                  file=f)

        # Use arc data to generate new node and arc data

        # Find each cluster involved in this route and generate a new boarding
        # arc for each, along with a dictionary to associate each cluster with
        # the boarding node
        boarding = {}
        for u in unique_clusters:
            if (u in boarding) == False:
                nodenum += 1
                boarding[u] = nodenum

        # Add new boarding nodes to node file
        with open(node_output_file, 'a') as f:
            for u in boarding:
                # ID, Name, Type, Line
                print(str(boarding[u])+"\tStop"+str(u)+"_Route"+str(routes[r])+
                      "\t"+str(nid_board)+"\t"+str(r), file=f)

        # Add arcs to arc file
        with open(arc_output_file, 'a') as f:
            # Line arcs
            for a in trip_arcs:
                # ID, Type, Line, Tail, Head, Time
                arcnum += 1
                print(str(arcnum)+"\t"+str(aid_line)+"\t"+str(r)+"\t"+
                      str(boarding[a[0]])+"\t"+str(boarding[a[1]])+"\t"+
                      str(trip_arcs[a]), file=f)

            # Boarding arcs
            for u in unique_clusters:
                arcnum += 1
                print(str(arcnum)+"\t"+str(aid_board)+"\t"+str(r)+"\t"+
                      str(u)+"\t"+str(boarding[u])+"\t0", file=f)

            # Alighting arcs
            for u in unique_clusters:
                arcnum += 1
                print(str(arcnum)+"\t"+str(aid_alight)+"\t"+str(r)+"\t"+
                      str(boarding[u])+"\t"+str(u)+"\t0", file=f)

        print("Done processing route "+str(routes[r]))

#------------------------------------------------------------------------------
def add_walking(cluster_file, arc_file, cutoff=0.5):
    """Generates walking arcs between stop clusters.

    Requires the name of the arc file and the stop cluster file.

    Accepts an optional keyword argument to specify the (taxicab) distance
    cutoff (miles), within which walking arcs will be generated.

    Clusters within the cutoff distance of each other will receive a pair of
    walking arcs between them, with a travel time calculated based on the
    distance and the walking speed defined above.

    In order to reduce the number of arcs in densely-packed clusters of stops,
    we prevent arcs from being generated between pairs of stops if the
    quadrangle defined by them contains another stop.
    """

    # Read in lists of stop IDs and coordinates
    ids = []
    coords = []
    with open(cluster_file, 'r') as f:
        i = -1
        for line in f:
            i += 1
            if i > 0:
                # Skip comment line
                dum = line.split()
                ids.append(dum[0])
                coords.append((float(dum[1]), float(dum[2])))

    # Go through each unique coordinate pair and generate a dictionary of pairs
    # within the cutoff distance of each other
    count = 0
    pairs = {}
    for i in range(len(coords)):
        print("Iteration "+str(i+1)+" / "+str(len(coords)))

        for j in range(i):
            # Calculate pairwise distance
            dist = distance(coords[i], coords[j], taxicab=True)
            if dist <= cutoff:
                keep = True # whether to keep the current pair

                # Define corners of quadrangle as most extreme lat/lon in pair
                lat_min = min(coords[i][0], coords[j][0])
                lat_max = max(coords[i][0], coords[j][0])
                lon_min = min(coords[i][1], coords[j][1])
                lon_max = max(coords[i][1], coords[j][1])

                # Scan entire stop list for stops within the quadrangle
                for k in range(len(coords)):
                    if (k != i) and (k != j):
                        if ((lat_min <= coords[k][0] <= lat_max) and
                            (lon_min <= coords[k][1] <= lon_max)):
                            # Stop found in quadrangle, making pair invalid
                            keep = False
                            break

                # If no stops were found in the quadrangle, then we add the
                # pair along with their walking time to the dictionary
                if keep == True:
                    count += 1
                    pairs[(ids[i], ids[j])] = dist * mile_walk_time

    # Use the final pairs dictionary to generate the new arcs and write them to
    # the arc file
    with open(arc_file, 'r+') as f:
        # Count the number of arcs
        arcnum = -np.inf
        f.readline()
        for line in f:
            arcnum = max(arcnum, int(f.readline().split()[0]))
        arcnum += 1

        for p in pairs:
            # ID, Type, Line, Tail, Head, Time
            print(str(arcnum)+"\t"+str(aid_walk)+"\t-1\t"+str(p[0])+"\t"+
                  str(p[1])+"\t"+str(pairs[p]), file=f)
            arcnum += 1
            print(str(arcnum)+"\t"+str(aid_walk)+"\t-1\t"+str(p[1])+"\t"+
                  str(p[0])+"\t"+str(pairs[p]), file=f)
            arcnum += 1

    print("Done. Added a total of "+str(count)+" pairs of walking arcs.")

#------------------------------------------------------------------------------
def cluster_boardings(bus_data, train_data, cluster_data, cluster_lookup,
                      stop_output, mode_output):
    """Calculates total daily bus/train boardings at each clustered stop.

    Requires the names of the bus and train boarding data files, the stop
    cluster file, the cluster lookup table, and and the output file names for
    the number of boardings at each stop and by each mode.

    The bus and train stop files both include the same IDs as the GTFS file,
    and so the cluster lookup table can be used to immediately find the nearest
    cluster for most cases. However, due to the difference in dates, not every
    listed stop has an associated GTFS entry. In these cases we need to
    manually find the nearest stop.

    For each listed stop, we find the nearest cluster and associate all daily
    boardings with that cluster. The stop output file lists the total number of
    boardings at each cluster. The mode output file lists the total number of
    boardings by each mode (bus or train).
    """

    # Read in list of cluster coordinates and initialize list of stop boardings
    cluster = []
    with open(cluster_data, 'r') as f:
        i = -1
        for line in f:
            i += 1
            if i > 0:
                # Skip comment line
                dum = line.split()
                cluster.append((float(dum[1]), float(dum[2])))
    stop_boardings = [0 for c in cluster]

    # Read in cluster lookup table
    lookup = {}
    with open(cluster_lookup, 'r') as f:
        i = -1
        for line in f:
            i += 1
            if i > 0:
                # Skip comment line
                dum = line.split()
                lookup[int(dum[0])] = int(dum[1])

    # Read in bus data and add total boardings
    bus_total = 0
    with open(bus_data, 'r') as f:
        i = -1
        for line in f:
            i += 1
            if i > 0:
                # Skip comment line

                print("Bus stop "+str(i))
                dum = [d.strip() for d in line.split(',')]
                stop = int(dum[0])

                # Measure table entries from right due commas in some names
                boardings = float(dum[-5])
                coords = (float(dum[-2][2:]), float(dum[-1][:-3]))

                # Find nearest cluster to stop
                if (stop in lookup) == True:
                    # If in lookup table, simply read
                    cb = lookup[stop]
                else:
                    # Otherwise, calculate nearest stop
                    cb = cluster.index(min(cluster, key=lambda cs:
                        distance(coords, cs)))

                # Tally boardings
                bus_total += boardings
                stop_boardings[cb] += boardings

    # Read in train data and add total boardings
    train_total = 0
    with open(train_data, 'r') as f:
        i = -1
        for line in f:
            i += 1
            if i > 0:
                # Skip comment line

                dum = [d.strip() for d in line.split(',')]

                # Skip entries from the incorrect month or year
                date = [int(d) for d in dum[-5].split('/')]
                if (date[0] != od_data_month) or (date[2] != od_data_year):
                    continue

                print("Train stop "+str(i))
                stop = int(dum[0])

                # Measure table entries from right due commas in some names
                boardings = float(dum[-4])

                # Find nearest cluster to stop
                if (stop in lookup) == True:
                    # If in lookup table, simply read
                    cb = lookup[stop]

                # Tally boardings
                train_total += boardings
                stop_boardings[cb] += boardings

    # Output the totals
    with open(mode_output, 'w') as f:
        print("Mode\tBoardings", file=f)
        print("Bus\t"+str(bus_total), file=f)
        print("Train\t"+str(train_total), file=f)

    with open(stop_output, 'w') as f:
        print("ID\tBoardings", file=f)
        for i in range(len(stop_boardings)):
            print(str(i)+"\t"+str(stop_boardings[i]), file=f)

#------------------------------------------------------------------------------
def gamma(t, mu, sigma):
    """Gamma distribution with a given mean and standard deviation.

    Requires a tip time, the mean trip time, and the trip time standard
    deviation, respectively.

    Returns a relative frequency scaled so that the function value of the mean,
    itself, is exactly 1.0.

    Each OD pair receives a seed matrix value to indicate how relatively likely
    that particular trip pair is. We use the pairwise travel time to determine
    this likelihood, assuming that trip lengths follow a gamma distribution.
    """

    k = 1/(mu**((mu**2-sigma**2)/(sigma**2))*np.exp(-(mu/sigma**2)*mu))
    return k*(t**((mu**2-sigma**2)/(sigma**2))*np.exp(-(mu/sigma**2)*t))

#------------------------------------------------------------------------------
def all_times(node_file, arc_file, distance_output):
    """Calculates all pairwise travel times between stop nodes.

    Requires the node file, arc file, and the name of an output file.

    Uses the node and arc data files to build a graph, and then applies
    Dijkstra's algorithm to calculate all pairwise travel times.

    The output file is formatted as a list rather than a matrix.
    """

    # Build vertex set and subset of stop vertices
    node = [] # all nodes
    node_stop = [] # only the stop nodes
    succ = {} # list of successors of each node
    with open(node_file, 'r') as f:
        i = -1
        for line in f:
            i += 1
            if i > 0:
                # Skip comment line
                dum = line.split()
                u = int(dum[0])
                node.append(u) # new node
                succ[u] = [] # initially empty successor list
                if int(dum[2]) == nid_stop:
                    node_stop.append(u) # new stop node

    # Build successor list dictionary and arc cost dictionary
    cost = {} # cost of each arc, indexed by (tail,head)
    with open(arc_file, 'r') as f:
        i = -1
        for line in f:
            i += 1
            if i > 0:
                # Skip comment line
                dum = line.split()
                u, v = int(dum[3]), int(dum[4])
                c = float(dum[5])
                if (v in succ[u]) == False:
                    succ[u].append(v) # add a new successor
                if ((u, v) in cost) == True:
                    cost[(u, v)] = min(cost[(u, v)], c) # update cheaper arc
                else:
                    cost[(u, v)] = c # add new arc

    # Initialize distance output file
    with open(distance_output, 'w') as f:
        print("Origin\tDestination\tTime", file=f)

    #--------------------------------------------------------------------------
    # Origin loop begin
    #
    for s in node_stop:
        print("Processing stop "+str(node_stop.index(s)+1)+" / "+
              str(len(node_stop)))

        # Initialize Dijkstra data structures
        q = set(node[:]) # unprocessed node set
        q_stop = set(node_stop[:]) # unprocessed stop node set
        dist = {} # dictionary of best known distances from s
        for u in node:
            dist[u] = finite_infinity
        dist[s] = 0.0

        #----------------------------------------------------------------------
        # Dijkstra main loop begin
        #
        while len(q_stop) > 0:
            # Find the unprocessed vertex with the minimum known distance
            u = min(q, key=dist.get)

            # Remove vertex from unprocessed sets
            q.remove(u)
            if (u in q_stop) == True:
                q_stop.remove(u)

            # Update distances of all successors of the chosen node
            for v in succ[u]:
                if (v in q) == True:
                    dist_new = dist[u] + cost[(u, v)]
                    if dist_new < dist[v]:
                        dist[v] = dist_new

        #
        # Dijkstra main loop end
        #----------------------------------------------------------------------

        # Output a list of all distances from the current origin
        with open(distance_output, 'a') as f:
            for u in node_stop:
                print(str(s)+"\t"+str(u)+"\t"+str(dist[u]), file=f)
    #
    # Origin loop end
    #--------------------------------------------------------------------------

    print("All distances calculated.")

#------------------------------------------------------------------------------
def od_matrix(stop_boardings, mode_boardings, distance_file, od_output,
              threshold=0.01, cutoff=1000, seed_zero=0.25, rounding=0.5):
    """Generates an estimated OD matrix.

    Requires the stop-level boarding file, the mode boarding file, the paiwise
    stop distance file, and the name of the final OD matrix output file.

    Accepts the following optional keyword arguments:
        threshold -- Error threshold for IPF. Defaults to 0.01. The algorithm
            terminates if the maximum elementwise difference between iterations
            is below this threshold.
        cutoff -- Iteration cutoff for IPF. Defaults to 1000. Maximum number of
            iterations to conduct in case the error threshold is not reached.
        seed_zero -- Threshold for seed matrix values. Defaults to 0.25. Any
            seed matrix values that fall below this threshold will be set to
            exactly 0, meaning that it will remain at exactly 0 for the
            remainder of the IPF algorithm.
        rounding -- Threshold for use in the final OD matrix rounding. Defaults
            to 0.5. Fractional parts greater than or equal to this value result
            in rounding up, and otherwise rounding down.

    The entire process for estimating the OD matrix involves building the graph
    defined by the node and arc files, using Dijkstra's algorithm to calcualte
    all pairwise distances between stop nodes on the graph, passing these
    distances through the gamma distribution to obtain a seed value, and
    finally applying IPF on the seed matrix to obtain the correct row and
    column sums.
    """

    # Calculate weighted mean trip time
    with open(mode_boardings, 'r') as f:
        f.readline()
        bus_boardings = float(f.readline().split()[1])
        train_boardings = float(f.readline().split()[1])
        bus_frac = bus_boardings/(bus_boardings+train_boardings)
        train_frac = train_boardings/(bus_boardings+train_boardings)
        trip_mean = bus_frac*bus_trip_mean + train_frac*train_trip_mean
        print("Mean trip time = "+str(trip_mean))

    # Read stop boarding totals
    boardings = {}
    index = {} # relative list index of each stop node ID
    with open(stop_boardings, 'r') as f:
        i = -1
        for line in f:
            i += 1
            if i > 0:
                # Skip comment line
                dum = line.split()
                boardings[int(dum[0])] = float(dum[1])
                index[int(dum[0])] = i - 1

    # Initialize OD matrix by calculating gamma distribution values for
    # pairwise distances from from file
    od = np.zeros((len(boardings), len(boardings)), dtype=float)
    with open(distance_file, 'r') as f:
        i = -1
        for line in f:
            i += 1
            if i > 0:
                # Skip comment line
                if i % len(boardings) == 1:
                    print("Reading stop "+str(int(i/len(boardings))+1)+" / "+
                          str(len(boardings)))

                dum = line.split()
                oid = index[int(dum[0])]
                did = index[int(dum[1])]
                od[oid][did] = gamma(float(dum[2]), trip_mean, gamma_std_dev)

    # Normalize the seed matrix so that the total sum of all elements is the
    # correct system-wide total
    od *= sum([boardings[i] for i in boardings]) / sum(sum(od))

    # Eliminate small seed values
    if od[oid][did] <= seed_zero:
        od[oid][did] = 0.0

    # Initialize IPF error and iteration count
    max_error = np.inf
    iteration = 0

    #--------------------------------------------------------------------------
    # IPF loop begin
    #
    while (max_error > threshold) and (iteration < cutoff):
        iteration += 1
        print("IPF iteration "+str(iteration))

        od_old = od.copy()

        # Row adjustment
        row_sum = np.sum(od, 1)
        for i in boardings:
            # Multiply each element by the ratio of its target row sum to its
            # current row sum
            if row_sum[index[i]] == 0:
                od[index[i]] *= 0
            else:
                ratio = boardings[i]/row_sum[index[i]]
                od[index[i]] *= ratio

        # Column adjustment
        col_sum = np.sum(od, 0)
        for i in boardings:
            # Multiply each element by the ratio of its target row sum to its
            # current row sum
            if col_sum[index[i]] == 0:
                od[:,index[i]] *= 0
            else:
                ratio = boardings[i]/col_sum[index[i]]
                od[:,index[i]] *= ratio

        max_error = min(max_error, np.linalg.norm(od - od_old))
        print(max_error)
    #
    # IPF loop end
    #--------------------------------------------------------------------------

    if max_error <= threshold:
        print("IPF ended by achieving error threshold at "+str(iteration)+
              " iterations")
    else:
        print("IPF ended due to iteration cutoff with an error threshold of "+
              str(max_error))

    # Output results
    with open(od_output, 'w') as f:
        print("ID\tOrigin\tDestination\tVolume", file=f)
        line_index = -1
        for i in boardings:
            for j in boardings:
                oid = index[i]
                did = index[j]
                vol = od[oid][did]
                if vol % 1 >= rounding:
                    vol = int(np.ceil(vol))
                else:
                    vol = int(np.floor(vol))
                if vol > 0:
                    # Skip recording volumes that are too close to zero
                    line_index += 1
                    print(str(line_index)+"\t"+str(oid)+"\t"+str(did)+"\t"+
                          str(vol), file=f)

#------------------------------------------------------------------------------
def network_assemble(input_stop_nodes, input_line_arcs, input_pop_nodes,
                     input_fac_nodes, input_clusters, community_names,
                     output_nodes, output_arcs, cutoff=0.5):
    """Assembles most of the intermediate files into the final network files.

    Requires the following file names in order:
        core network nodes (stops and boarding)
        core network arcs (line, boarding, alighting, and walking)
        population center nodes
        facility nodes
        cluster coordinates
        community area names
        final node output
        final arc output

    Accepts an optional keyword "cutoff" for use in generating walking arcs
    between population centers/facilities/stops. Defaults to 0.5. Represents
    taxicab distance (miles) within wich to generate walking arcs.

    The network assembly process consists mostly of incorporating the
    population centers and facilities into the main network. This is done in
    mostly the same way as the walking arc script, except that each facility
    and population center is guaranteed to receive at least one walking arc,
    which is connected to the nearest stop node if none were within the cutoff.
    """

    # Read in lists of stop IDs and coordinates
    stop_ids = []
    stop_coords = []
    with open(input_clusters, 'r') as f:
        i = -1
        for line in f:
            i += 1
            if i > 0:
                # Skip comment line
                dum = line.split()
                stop_ids.append(dum[0])
                stop_coords.append((float(dum[1]), float(dum[2])))

    # Read in dictionaries indexed by population center IDs to contain the
    # population values, center names, and coordinates
    pop_names = {}
    populations = {}
    pop_coords = {}
    with open(input_pop_nodes, 'r') as f:
        i = -1
        for line in f:
            i += 1
            if i > 0:
                # Skip comment line
                dum = line.split()
                pop_id = int(dum[0])
                populations[pop_id] = int(float(dum[1]))
                pop_coords[pop_id] = (float(dum[2]), float(dum[3]))
    with open(community_names, 'r') as f:
        i = -1
        for line in f:
            i += 1
            if i > 0:
                # Skip comment line
                dum = line.split('\t')
                pop_names[int(dum[1])] = dum[0].replace(' ', '_')

    # Go through each population center and generate a dictionary of stop IDs
    # that should be linked to each center
    count = 0
    pop_links = {}
    pop_link_times = {}
    for i in pop_coords:
        print("Population center "+str(i))

        # Continue searching until we find at least one link to add
        effective_cutoff = cutoff
        pop_links[i] = []
        pop_link_times[i] = []
        while len(pop_links[i]) == 0:

            for j in range(len(stop_coords)):
                # Calculate pairwise distance
                dist = distance(pop_coords[i], stop_coords[j], taxicab=True)
                if dist <= effective_cutoff:
                    keep = True # whether to keep the current pair

                    # Define corners of quadrangle
                    lat_min = min(pop_coords[i][0], stop_coords[j][0])
                    lat_max = max(pop_coords[i][0], stop_coords[j][0])
                    lon_min = min(pop_coords[i][1], stop_coords[j][1])
                    lon_max = max(pop_coords[i][1], stop_coords[j][1])

                    # Scan entire stop list for stops within the quadrangle
                    for k in range(len(stop_coords)):
                        if k != j:
                            if ((lat_min <= stop_coords[k][0] <= lat_max) and
                                (lon_min <= stop_coords[k][1] <= lon_max)):
                                # Stop found in quadrangle, making pair invalid
                                keep = False
                                break

                    # If no stops were found in the quadrangle, then we add the
                    # pair along with their walking time to the dictionary
                    if keep == True:
                        count += 1
                        pop_links[i].append(stop_ids[j])
                        pop_link_times[i].append(dist*mile_walk_time)

            # Double the effective cutoff in case the search was unsuccessful
            # and must be repeated
            if len(pop_links[i]) == 0:
                effective_cutoff *= 2
                print("No links found. Trying again with cutoff "+
                      str(effective_cutoff))

    print("Adding a total of "+str(count)+" population walking arcs.")

    # Read in lists to contain the facility names and coordinates
    fac_names = []
    fac_coords = []
    with open(input_fac_nodes, 'r') as f:
        i = -1
        for line in f:
            i += 1
            if i > 0:
                # Skip comment line
                dum = line.split()
                fac_names.append(dum[0])
                fac_coords.append((float(dum[1]), float(dum[2])))

    # Go through each facility and generate a dictionary of stop IDs that
    # should be linked to each facility
    count = 0
    fac_links = {}
    fac_link_times = {}
    for i in range(len(fac_coords)):
        print("Facility center "+str(i+1)+" / "+str(len(fac_coords)))

        # Continue searching until we find at least one link to add
        effective_cutoff = cutoff
        fac_links[i] = []
        fac_link_times[i] = []
        while len(fac_links[i]) == 0:

            for j in range(len(stop_coords)):
                # Calculate pairwise distance
                dist = distance(fac_coords[i], stop_coords[j], taxicab=True)
                if dist <= effective_cutoff:
                    keep = True # whether to keep the current pair

                    # Define corners of quadrangle
                    lat_min = min(fac_coords[i][0], stop_coords[j][0])
                    lat_max = max(fac_coords[i][0], stop_coords[j][0])
                    lon_min = min(fac_coords[i][1], stop_coords[j][1])
                    lon_max = max(fac_coords[i][1], stop_coords[j][1])

                    # Scan entire stop list for stops within the quadrangle
                    for k in range(len(stop_coords)):
                        if k != j:
                            if ((lat_min <= stop_coords[k][0] <= lat_max) and
                                (lon_min <= stop_coords[k][1] <= lon_max)):
                                # Stop found in quadrangle, making pair invalid
                                keep = False
                                break

                    # If no stops were found in the quadrangle, then we add the
                    # pair along with their walking time to the dictionary
                    if keep == True:
                        count += 1
                        fac_links[i].append(stop_ids[j])
                        fac_link_times[i].append(dist*mile_walk_time)

            # Double the effective cutoff in case the search was unsuccessful
            # and must be repeated
            if len(fac_links[i]) == 0:
                effective_cutoff *= 2
                print("No links found. Trying again with cutoff "+
                      str(effective_cutoff))

    print("Adding a total of "+str(count)+" facility walking arcs.")

    # Write new nodes to final output files
    with open(output_nodes, 'w') as fout:
        # Comment line
        print("ID\tName\tType\tLine\tValue", file=fout)

        # Copy old node file contents
        with open(input_stop_nodes, 'r') as fin:
            i = -1
            nodenum = -1
            for line in fin:
                i += 1
                if i > 0:
                    # Skip comment line
                    dum = line.split()
                    # ID, Name, Type, Line
                    if int(dum[0]) > nodenum:
                        nodenum = int(dum[0])
                    print(dum[0]+"\t"+dum[1]+"\t"+dum[2]+"\t"+dum[3]+"\t-1",
                          file=fout)

        # Write population center nodes
        pop_nodes = {}
        for i in pop_names:
            nodenum += 1
            pop_nodes[i] = nodenum
            print(str(nodenum)+"\t"+str(i)+"_"+str(pop_names[i])+"\t"+
                  str(nid_pop)+"\t-1\t"+str(populations[i]), file=fout)

        # Write facility nodes
        fac_nodes = []
        for i in range(len(fac_names)):
            nodenum += 1
            fac_nodes.append(nodenum)
            print(str(nodenum)+"\t"+str(fac_names[i])+"\t"+str(nid_fac)+
                  "\t-1\t1", file=fout)

    # Write new arcs to output files
    with open(output_arcs, 'w') as fout:
        # Comment line
        print("ID\tType\tLine\tTail\tHead\tTime", file=fout)

        # Copy old arc file contents
        with open(input_line_arcs, 'r') as fin:
            i = -1
            arcnum = -1
            for line in fin:
                i += 1
                if i > 0:
                    # Skip comment line
                    dum = line.split()
                    # ID, Type, Line, Tail, Head, Time
                    if int(dum[0]) > arcnum:
                        arcnum = int(dum[0])
                    print(line.strip(), file=fout)

        # Write population center walking arcs
        for i in pop_links:
            for j in range(len(pop_links[i])):
                arcnum += 1
                print(str(arcnum)+"\t"+str(aid_walk_health)+"\t-1\t"+
                      str(pop_nodes[i])+"\t"+str(pop_links[i][j])+"\t"+
                      str(pop_link_times[i][j]), file=fout)
                arcnum += 1
                print(str(arcnum)+"\t"+str(aid_walk_health)+"\t-1\t"+
                      str(pop_links[i][j])+"\t"+str(pop_nodes[i])+"\t"+
                      str(pop_link_times[i][j]), file=fout)

        # Write facility walking arcs
        for i in fac_links:
            for j in range(len(fac_links[i])):
                arcnum += 1
                print(str(arcnum)+"\t"+str(aid_walk_health)+"\t-1\t"+
                      str(fac_nodes[i])+"\t"+str(fac_links[i][j])+"\t"+
                      str(fac_link_times[i][j]), file=fout)
                arcnum += 1
                print(str(arcnum)+"\t"+str(aid_walk_health)+"\t-1\t"+
                      str(fac_links[i][j])+"\t"+str(fac_nodes[i])+"\t"+
                      str(fac_link_times[i][j]), file=fout)

#------------------------------------------------------------------------------
def transit_finalization(transit_input, transit_output):
    """Converts the intermediate transit data file into the final version.

    Requires the names of the intermediate transit data file and the final
    transit data file.

    Data fields to be added for the final file include boarding fare, upper and
    lower fleet size bounds, and the values of the initial line frequency and
    capacity.
    """

    with open(transit_output, 'w') as fout:
        # Comment line
        print("ID\tName\tType\tFleet\tCircuit\tScaling\tLB\tUB\tFare\t"+
              "Frequency\tCapacity", file=fout)

        # Readh through the initial file and process each line
        with open(transit_input, 'r') as fin:
            i = -1
            for line in fin:
                i += 1
                if i > 0:
                    # Skip comment line
                    dum = line.split()

                    # Read existing values
                    labels = dum[0]+"\t"+dum[1]+"\t" # ID and Name
                    line_type = type_remap[int(dum[2])] # vehicle type
                    fleet = int(np.ceil(float(dum[3]))) # fleet size
                    circuit = float(dum[4]) # circuit time
                    scaling = float(dum[5]) # active fraction of day

                    # Set bounds
                    lb = -np.inf
                    ub = np.inf
                    if line_type == type_train:
                        vcap = train_capacity
                        # Train bounds should both equal the current fleet
                        lb = ub = fleet
                    elif line_type == type_bus:
                        vcap = bus_capacity
                        # Bus upper bound should be infinite
                        ub = finite_infinity
                        # Bus lower bound is minimum number of vehicles
                        # required to achive a frequency of 1/30 (or the
                        # initial fleet size, if that is lower)
                        lb = int(min(np.ceil(circuit/30), fleet))

                    # Calculate initial frequency and line capacity
                    freq = fleet/circuit
                    cap = vcap*freq*(1440*scaling)

                    # Write line
                    print(labels+str(line_type)+"\t"+str(fleet)+"\t"+
                          str(circuit)+"\t"+str(scaling)+"\t"+str(lb)+"\t"+
                          str(ub)+"\t"+str(cta_fare)+"\t"+str(freq)+"\t"+
                          str(cap), file=fout)

#------------------------------------------------------------------------------
def misc_files(vehicle_output, operator_output, user_output, assignment_output,
               objective_output, problem_output, transit_input):
    """Assembles various miscellaneous problem parameter files.

    Requires the following output file names (and one input file) in order:
        vehicle data
        operator cost data
        user cost data
        assignment model data
        objective function data
        miscellaneous problem data
        (input) transit data

    Most of this process consists of simply formatting the parameters defined
    above into the necessary output file format.

    The operator and user cost data files both include placeholders for the
    initial values of their respective functions, to be determined after
    evaluating them for the initial network.
    """

    # Read transit data to calculate vehicle totals
    bus_total = 0
    train_total = 0
    with open(transit_input, 'r') as f:
        i = -1
        for line in f:
            i += 1
            if i > 0:
                # Skip comment line
                dum = line.split()
                vtype = type_remap[int(dum[2])]
                fleet = int(np.ceil(float(dum[3])))

                if vtype == type_bus:
                    bus_total += fleet
                elif vtype == type_train:
                    train_total += fleet

    print("Total of "+str(bus_total)+" buses")
    print("Total of "+str(train_total)+" trains")

    # Vehicle file
    with open(vehicle_output, 'w') as f:
        # Comment line
        print("Type\tName\tUB\tCapacity\tCost", file=f)
        print(str(type_bus)+"\tBus_New_Flyer_D40LF\t"+str(bus_total)+"\t"+
              str(bus_capacity)+"\t"+str(cost_bus), file=f)
        print(str(type_train)+"\tTrain_5000-series\t"+str(train_total)+"\t"+
              str(train_capacity)+"\t"+str(cost_train), file=f)

    # Operator cost file
    with open(operator_output, 'w') as f:
        print("Field\tValue", file=f)
        print("Initial\t-1", file=f)
        print("Elements\t"+str(len(op_coef)), file=f)

        # Print cost coefficients
        for i in range(len(op_coef)):
            print(str(op_coef_names[i])+"\t"+str(op_coef[i]), file=f)

    # User cost file
    with open(user_output, 'w') as f:
        print("Field\tValue", file=f)
        print("Initial\t-1", file=f)
        print("Elements\t"+str(len(us_coef)), file=f)

        # Print cost coefficients
        for i in range(len(us_coef)):
            print(str(us_coef_names[i])+"\t"+str(us_coef[i]), file=f)

    # Assignment model parameter file
    with open(assignment_output, 'w') as f:
        print("Field\tValue", file=f)
        print("Epsilon\t"+str(assignment_epsilon), file=f)
        print("Cutoff\t"+str(assignment_max), file=f)
        print("Parameters\t"+str(len(latency_parameters)), file=f)

        # Print latency function parameters
        for i in range(len(latency_parameters)):
            print(str(latency_names[i])+"\t"+str(latency_parameters[i]),
                  file=f)

    # Objective function parameter file
    with open(objective_output, 'w') as f:
        print("Field\tValue", file=f)
        print("Elements\t"+str(len(obj_parameters)), file=f)

        # Print objective function parameters
        for i in range(len(obj_parameters)):
            print(str(obj_names[i])+"\t"+str(obj_parameters[i]), file=f)

    # Miscellaneous problem parameter file
    with open(problem_output, 'w') as f:
        print("Field\tValue", file=f)
        print("Elements\t"+str(len(misc_parameters)), file=f)

        # Print parameters
        for i in range(len(misc_parameters)):
            print(str(misc_names[i])+"\t"+str(misc_parameters[i]), file=f)

#==============================================================================
# Execution
#==============================================================================

# Comment out lines to skip portions of preprocessing.

census_processing(tract_data, community_names, community_conversion,
                  population_raw, population_clustered)

facility_processing(facility_in, facility_out)

stop_cluster(stop_data, k_clusters, output_file=stop_cluster_file,
             lookup_file=stop_cluster_lookup)

stop_cluster_measure(stop_list, stop_cluster_file, stop_cluster_lookup)

transit_processing(stop_data, trip_data, route_data, time_data, line_nodes,
                   line_arcs, transit_data, cluster_file=stop_cluster_file,
                   cluster_lookup=stop_cluster_lookup)

add_walking(stop_cluster_file, line_arcs, cutoff=0.75)

cluster_boardings(od_data_bus, od_data_train, stop_cluster_file,
                  stop_cluster_lookup, cluster_boarding, mode_boarding)

all_times(line_nodes, line_arcs, all_pairs_distance)

od_matrix(cluster_boarding, mode_boarding, all_pairs_distance, final_od_data)

network_assemble(line_nodes, line_arcs, population_clustered, facility_out,
                 stop_cluster_file, community_names, final_node_data,
                 final_arc_data, cutoff=1.0)

transit_finalization(transit_data, final_transit_data)

misc_files(vehicle_file, oc_file, uc_file, assignment_file, objective_file,
           problem_file, transit_data)
