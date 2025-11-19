import random
import tifffile as tiff
from skimage.measure import regionprops
import networkx as nx
import matplotlib.pyplot as plt
import math
from scipy.stats import geom, multivariate_normal
import numpy as np
from matplotlib.colors import ListedColormap
from tqdm import tqdm
#from tqdm.notebook import tqdm
from scipy.linalg import sqrtm
from scipy.spatial import cKDTree, KDTree
import pandas as pd
import subprocess
import glob, os

#Note: assuming images to track have the following name convention: F14_5_roi4_segmentation_CROP_processed.tif 
#      this is important for saving tracked image with the proper name
folder = "ims_to_track\\" #Gaby folder with all files to run here
images = glob.glob(os.path.join(folder,'*.tif'))

def sanitize_node_label(node):
    """Convert tuples like ('copy', 1, 0) to 'copy_1_0' for LEMON compatibility."""
    if isinstance(node, tuple):
        return '_'.join(str(part) for part in node)
    return str(node)

def convert_tracking_graph_to_lemon(G, output_filename="solver/graph.lgf"):
    nodes = list(G.nodes)
    nodes = [sanitize_node_label(node) for node in nodes]

    arcs = []
    for u, v, data in G.edges(data=True):
        u_str = sanitize_node_label(u)
        v_str = sanitize_node_label(v)
        label = data.get('label', f"{u_str}_{v_str}")
        capacity = data.get('capacity')
        cost = data.get('weight')
        arcs.append((u_str, v_str, label, capacity, cost))

    lines = []
    lines.append("@nodes\nlabel")
    lines.extend(nodes)

    lines.append("\n@arcs\nlabel capacity cost")
    for u, v, label, capacity, cost in arcs:
        lines.append(f"{u} {v} {label} {capacity} {cost}")

    lines.append("\n@attributes")
    #source = sanitize_node_label(G.graph.get('source', 'unknown_source'))
    lines.append(f"source {nodes[0]}")
    lines.append(f"target {nodes[-1]}")
    #lines.append(f"target {target}")

    lemon_text = "\n".join(lines)
    with open(output_filename, "w") as f:
        f.write(lemon_text)

def str_to_tuple(s):
    """Convert node string like 'copy_1_1' or '2_0' to tuple."""
    parts = s.split('_')
    if parts[0] == 'copy':
        return ('copy', int(parts[1]), int(parts[2]))
    else:
        return tuple(map(int, parts))

def parse_output_txt(filename):
    G = nx.DiGraph()
    flowDict = {}
    flowCost = None

    with open(filename, "r") as f:
        lines = f.readlines()

    node_section = False
    arc_section = False

    for line in lines:
        line = line.strip()

        if line.startswith("Total cost:"):
            flowCost = float(line.split(":")[1].strip())
        elif line == "@nodes":
            node_section = True
            arc_section = False
            continue
        elif line == "@arcs":
            node_section = False
            arc_section = True
            continue
        elif line.startswith("@attributes"):
            node_section = False
            arc_section = False
            continue

        # Parse nodes
        if node_section:
            if line and not line.startswith("label"):
                node = str_to_tuple(line)
                G.add_node(node)

        # Parse arcs and flow
        if arc_section:
            if line and not line.startswith("label") and not line.startswith("flow"):
                parts = line.split()
                if len(parts) >= 6:
                    u_str, v_str = parts[0], parts[1]
                    label = parts[2]
                    capacity = int(parts[3])
                    cost = int(parts[4])
                    flow = int(parts[5])

                    u, v = str_to_tuple(u_str), str_to_tuple(v_str)
                    G.add_edge(u, v, capacity=capacity, cost=cost)

                    if flow > 0:
                        if u not in flowDict:
                            flowDict[u] = {}
                        flowDict[u][v] = flow

    return flowCost, flowDict

def assign_track_ids_to_segmentation(seg_img, tracks_df, scales, distance_thresh=0.001):
    """
    Match regionprops in segmentation to tracks based on centroid proximity.
    """
    T = seg_img.shape[0]
    track_ids_used = set(tracks_df['track_id'])
    max_track_id = max(track_ids_used)
    counter = 1  # for new IDs

    # Precompute the scaling matrix
    scale_matrix = np.diag(scales)

    # Create a dictionary of tracks by timepoint
    tracks_by_time = {t + 1: tracks_df[tracks_df['t'] == t + 1] for t in range(T)}

    # Initialize an empty segmentation map
    updated_seg = np.zeros_like(seg_img, dtype=np.int32)

    for t in tqdm(range(T), desc="Processing frames"):
        seg_frame = seg_img[t]
        regions = regionprops(seg_frame.astype(int))

        # Extract tracks for the current timepoint
        tracks_t = tracks_by_time[t + 1]

        # Create a KDTree for fast nearest neighbor search
        track_positions = tracks_t[['x', 'y', 'z']].values
        track_tree = KDTree(track_positions)

        for region in regions:
            centroid_voxel = np.array(region.centroid)
            centroid_phys = scale_matrix @ centroid_voxel

            # Find the nearest track using KDTree
            dist, idx = track_tree.query(centroid_phys, distance_upper_bound=distance_thresh)

            if dist < distance_thresh:
                track_id = int(tracks_t.iloc[idx]['track_id'])
            else:
                track_id = max_track_id + counter
                counter += 1

            # Assign the track ID to the region's pixels
            coords = region.coords  # voxel coordinates (z, y, x)
            for coord in coords:
                updated_seg[t, coord[0], coord[1], coord[2]] = track_id

    return updated_seg

def get_P(seg_I0, scales): # standard
    """ Will return physical coordinates """
    regions = regionprops(seg_I0.astype(int))

    # Extract centroids and store as 2xK matrix
    centroids = np.array([region.centroid for region in regions]).T  # Transpose to get 2xK matrix
    labels = np.array([region.label for region in regions])
    #random.shuffle(labels)
    #labels = np.random.randint(1000, size=len(labels))
    #print(labels
    # return random labels
    # then our metrics should become much smaller

    return np.diag(scales)@centroids, labels

def get_P_list(seg_I0, scales): #standard
    """ Will return physical coordinates """
    regions = regionprops(seg_I0.astype(int))

    # Extract centroids and store as 2xK matrix
    centroids = [np.diag(scales)@region.centroid for region in regions]  # Transpose to get 2xK matrix
    labels = np.array([region.label for region in regions])
    #random.shuffle(labels)
    #labels = np.random.randint(1000, size=len(labels))
    #print(labels
    # return random labels
    # then our metrics should become much smaller

    return centroids, labels

def dist(p1, p2, sigma): #isotropic mahalanobis distance
    return (np.linalg.norm(p2-p1)**2)/(2*sigma**2)

def spatiotemporal_dist(delta_x, delta_t, cov, p_success=0.8):
    log_p_t = geom.logpmf(delta_t, p_success)
    log_p_x = multivariate_normal(np.zeros(3), cov).logpdf(delta_x)
    #log_p_x = multivariate_normal(mean=np.zeros(3), cov=np.diag([0.3**2, 0.3**2, 0.5**2])).logpdf(delta_x)
    #return -(log_p_t + log_p_x)
    return -(log_p_x)
    #return (np.linalg.norm(delta_x)**2)/(2*sigma**2)
    #return (np.linalg.norm(p2-p1)**2)/(2*sigma**2)

def get_source_edges(d): # verified
    return [((0, 0), (t, j)) for t in range(1, len(d) - 1) for j in range(len(d[t]))]

def get_sink_edges(d): #verified
    return [(('copy', t, j), (len(d) - 1, 0)) for t in range(1, len(d) - 1) for j in range(len(d[t]))]

def get_temporal_edges(d, w, thresh, cov): # mahalanobis version
    e = []
    cov_inv = np.linalg.inv(cov)
    cov_inv_sqrt = sqrtm(cov_inv)
    #L = np.linalg.cholesky(cov_inv)  # or use scipy.linalg.sqrtm(cov_inv) if not PSD

    for t in range(1, len(d) - 2):
        for t_plus in range(t + 1, min(t + 1 + w, len(d) - 1)):
            d_t = np.array(d[t])
            d_t_plus = np.array(d[t_plus])

            # Transform data to latent space
            z_t = np.transpose(cov_inv_sqrt @ np.transpose(d_t)) 
            z_t_plus = np.transpose(cov_inv_sqrt @ np.transpose(d_t_plus))

            tree = cKDTree(z_t_plus)
            pairs = tree.query_ball_point(z_t, r=3)

            for i, neighbors in enumerate(pairs):
                for j in neighbors:
                    #print(t, i,t_plus, j)
                    displacement = d_t_plus[j] - d_t[i]
                    dist = spatiotemporal_dist(displacement, t_plus - t, cov)
                    e.append((('copy', t, i), (t_plus, j), dist))

    return e

def build_graph(detections, k, thresh, cov):
    d = [[(0, 0)]] + detections + [[(0, 0)]] # we add two extra, that's why we do len(d)-1 while finding sink node
    G = nx.DiGraph()

    # Source node
    G.add_node((0, 0))

    # Add all original and copy nodes
    for t in range(1, len(d)-1): # skip source and sink
        for i in range(len(d[t])): # 
            orig = (t, i)
            copy = ('copy', t, i)
            G.add_node(orig, pos=d[t][i])
            G.add_node(copy, pos=d[t][i])

    # Sink node
    G.add_node((len(d) - 1, 0))

    # Source edges
    for u, v in get_source_edges(d):
        G.add_edge(u, v, capacity=int(1), weight=0)

    # Loop edges to copies (encode detection cost here)
    for t in range(1, len(d) - 1):  # skip source/sink
        for i in range(len(d[t])):
            orig = (t, i)
            copy = ('copy', t, i)
            G.add_edge(orig, copy, capacity=int(1), weight=-int(1000000))  # replace weight with -log(conf) if needed

    # Temporal edges: from copy → next original node
    for u, v, cost in get_temporal_edges(d, 8, thresh, cov):
        #print(cost)
        G.add_edge(u, v, capacity=int(1), weight=int(cost*10000))

    # Sink edges: from copy nodes
    for u, v in get_sink_edges(d):
        G.add_edge(u, v, capacity=int(1), weight=int(0))

    # Node demands
    nx.set_node_attributes(G, int(0), "demand") # set demand of 0 for all nodes
    G.nodes[(0, 0)]["demand"] = -int(k) # make sure integer flow of -k units
    G.nodes[(len(d) - 1, 0)]["demand"] = int(k) #make sure integer flow of +k units, so all flow ends up in sink

    return G, d

def extract_flow_paths(flowdict, source, sink, min_flow=1):
    G = nx.DiGraph()

    # Build a graph where edges with flow >= min_flow are included
    for u, neighbors in flowdict.items():
        for v, flow in neighbors.items():
            if flow >= min_flow:
                G.add_edge(u, v, flow=flow)

    # Find all simple paths from source to sink in this flow graph
    paths = list(nx.all_simple_paths(G, source=source, target=sink))
    return paths

def get_tracks(paths, detections): # verified
    tracks = []
 
    for i, path in enumerate(paths, 1):
        #print("path1")
        for node in path[1:-1]: # first and last are edges involving source and sink so skip
            if node[0] != 'copy':
                t, j = node
                #print(t, j)
                tracks.append((t, i, detections[t-1][j]))

    tracks = pd.DataFrame(tracks, columns=['t', 'track_id', 'position'])

    # Expand position into separate columns
    pos_df = tracks['position'].apply(pd.Series)
    pos_cols = ['x', 'y', 'z'][:pos_df.shape[1]]
    pos_df.columns = pos_cols

    tracks = pd.concat([tracks.drop(columns='position'), pos_df], axis=1)

    return tracks

## Setup Initial Parameters

#spatial_thresholds = [1.13]#np.linspace(0.1, 3, 100) # very large spatial threshold
spatial_thresholds = [1000]
scales = [0.096, 0.096, 0.33]

#cov = np.diag([3, 3, 3])

cov_fixed = np.array([[0.14984745, 0.01853253, 0.0749335 ],
 [0.01853253, 0.08066656, 0.02944603],
 [0.0749335,  0.02944603, 0.23925671]])

cov_G_labelled = np.array([
    [0.07384764, 0.01238046, 0.05048932],
    [0.01238046, 0.05005508, 0.02076806],
    [0.05048932, 0.02076806, 0.26948054]
])

cov_A_labelled = np.array([
    [0.08763301, 0.01724761, 0.05815704],
    [0.01724761, 0.05762029, 0.01916235],
    [0.05815704, 0.01916235, 0.31619810]
])

covs = [cov_G_labelled]
#covs = [cov_fixed]
sigma_est2 = 0.13
#covs = [np.diag([sigma_est2, sigma_est2, sigma_est2])]
#covs = [np.eye(cov_fixed.shape[0])]

scale_factor=0.33/0.096
z_size = 160
x_size = int(z_size*scale_factor)
y_size = int(z_size*scale_factor)

#Start the batch job

for i in images:
    ground_seg_img = tiff.imread(i)
    ground_seg_img = ground_seg_img.transpose((0, 3, 2, 1)) # t, x, y, z
    im_name = i.split('\\')[-1].split('_')[:2]
    im_name = '_'.join(im_name)    
    #ground_seg_img = ground_seg_img[:,0:x_size,0:y_size,0:z_size]
    print("Starting tracking on " + im_name)
    print(ground_seg_img.shape)

    ## Generate Data

    # ground_seg_img is unlabelled, ground_seg_I0 is labelled, Gabys tracks

    detections = []
    #Labels = []

    for t in range(ground_seg_img.shape[0]):
        frame = ground_seg_img[t]
        coords, labels = get_P_list(frame, scales)

        detections.append(coords)

    # tracking used ground_seg_img which is unlabelled

    ## Build graph in networkx

    #103

    #dfs_mot = [] #Gaby Commented out bc not used

    cov = covs[0]
    k = int(len(detections[0])/0.8)
    #k = int(len(detections[0]))
    #k = 100

    print(cov, k)

    spatial_threshold = spatial_thresholds[0]

    print("Building Graph", cov, spatial_threshold, k)

    # Build graph
    G, d = build_graph(detections, int(k), spatial_threshold, cov)

    # !rm -rf solver/*.lgf
    # !rm -rf solver/*.txt

    print("Solving Flow")
    convert_tracking_graph_to_lemon(G)
    #subprocess.run(["g++", "-o", "flow", "cost_scaling.cc", "-lemon"], cwd="solver", check=True) #Gaby commented out and replaced bc it wasn't finding my lemon folder

    #    subprocess.run([
    #        "g++",
    #        "-o", "flow",
    #        "cost_scaling.cc",
    #        "-I", "C:\\Users\\Huganir Lab\\Documents\\GitHub\\SynTrack\\lemon-1.3.1\\include",
    #        "-L", "C:\\Users\\Huganir Lab\\Documents\\GitHub\\SynTrack\\lemon-1.3.1\\lib",
    #        "-lemon"
    #    ], cwd="solver", check=True) #Gaby doesn't understand what this does, it has no specific info from this, do we need to run everytime?

    subprocess.run([
        "cl",
        "/EHsc",
        "/MD",
        "cost_scaling.cc",
        "/I", r"C:\Users\Huganir Lab\Documents\GitHub\SynTrack\lemon-1.3.1\include",
        "/link",
        "/LIBPATH:C:\\Users\\Huganir Lab\\Documents\\GitHub\\SynTrack\\lemon-1.3.1\\lib",
        "lemon.lib"
    ], cwd="solver", check=True) #Gaby Added this using chatGPT

    # Run the compiled program with k as argument and redirect output to output.txt
    with open("solver/output.txt", "w") as f:
        subprocess.run(["solver/cost_scaling.exe", str(k)], cwd="solver", stdout=f, check=True)



    print("Reading Flow")


    #assert(flowDict_filtered == flowDict_filtered)

    ## Track Extraction

    filename = "solver/output.txt"
    flowCost_cpp, flowDict_cpp = parse_output_txt(filename)

    ## Track stats => Not really necessary for batch jobs right (GABY)

    # Example usage:
    source = (0, 0)
    sink = (len(detections)+1, 0)
    paths = extract_flow_paths(flowDict_cpp, source, sink)

    # Assuming you have the 'tracks' DataFrame with 'x', 'y', 'z', and 'track_id' columns.
    tracks = get_tracks(paths, detections)

    # Assuming 'tracks' is your DataFrame
    # Sort by 'track_id' and time
    tracks = tracks.sort_values(by=['track_id', 't'])

    # Compute the step-wise change in x, y, z
    tracks[['dx', 'dy', 'dz']] = tracks.groupby('track_id')[['x', 'y', 'z']].diff()

    # Compute Euclidean step distance
    tracks['step_distance'] = np.sqrt(tracks['dx']**2 + tracks['dy']**2 + tracks['dz']**2)

    # Remove NaN from first step per track (no previous point to diff)
    step_distances = tracks['step_distance'].dropna()

    # Compute the average step size across all steps
    average_step_size = step_distances.mean()

    print("Average step size (between consecutive time steps):", average_step_size)


    ## Recolor segmentation, it should look good (since physical looks good)


    # here we relabel ground_seg_I0

    seg_new = assign_track_ids_to_segmentation(ground_seg_img, tracks, scales)
    seg_new = seg_new.transpose((0, 3, 2, 1))  # t, x, y, z

    # # 4. Save the resulting array as a TIFF file
    tiff.imwrite(f"tracked_ims\\{im_name}_syntracked.tif", np.asarray(np.expand_dims(seg_new, axis=2), dtype=np.float32),imagej=True) 

    # Convert 't' to datetime if it's not already
    tracks['t'] = pd.to_datetime(tracks['t'])

    # Get number of unique days each track appears
    track_days = tracks.groupby('track_id')['t'].nunique()

    # Calculate the mean number of days
    mean_days = track_days.mean()

    print(mean_days, average_step_size )
