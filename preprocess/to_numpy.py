import uproot
import numpy as np
import awkward as ak
from multiprocessing import Pool


#data_schema: num_clusters, max_num_trigger_primitives, (time_start, time_over_threshold, time_peak, channel, adc_integral, adc_peak)
def load_clusters(filename: str) -> ak.Array:
    file = uproot.open(filename)
    
    clusters_list = []
    for tree in file.keys():

        cluster = file[tree].arrays(library='ak')
        clusters_list.append(cluster)

    return ak.concatenate(clusters_list)

def calculate_position(tp):
    # Define the constants
    EVENTS_OFFSET = 5000
    apa_length_in_cm = 230
    wire_pitch_in_cm_collection = 0.479
    offset_between_apa_in_cm = 2.4
    time_tick_in_cm = 0.0805
    apa_width_in_cm = 4.7

    z_apa_offset = int(tp[3]) / (2560 * 2) * (apa_length_in_cm + offset_between_apa_in_cm)
    z_channel_offset = ((int(tp[3]) % 2560 - 1600) % 480) * wire_pitch_in_cm_collection
    z = wire_pitch_in_cm_collection + z_apa_offset + z_channel_offset

    y = 0
    x_signs = -1 if (int(tp[3]) % 2560 - 2080 < 0) else 1
    x = ((int(tp[2]) % EVENTS_OFFSET) * time_tick_in_cm + apa_width_in_cm / 2) * x_signs

    return [x, z]

    
def read_cluster(args):
    trigger_primitives, max_rows = args
    trigger_primitives_array = np.ones((max_rows, 6)) * -1000
    
    for i, trigger_primitive in enumerate(trigger_primitives):
        trigger_primitives_array[i] = trigger_primitive[:6]
    
    # calculate the position pad empty values to -1000
    padding_mask = trigger_primitives_array[:, 0] == -1000 
    positions = np.ones((max_rows, 2)) * -1
    for i, tp in enumerate(trigger_primitives_array):
        positions[i] = calculate_position(tp)
    positions[padding_mask] = [-1000, -1000]
    integral_peak = trigger_primitives_array[:, 4:6]
    integral_peak[padding_mask] = [-1000, -1000]
    return np.hstack([positions, integral_peak])


def to_numpy(clusters: ak.Array):
    max_num_trigger_primitives = 195 # 195 in cc
    # Use multiprocessing to parallelize the conversion
    with Pool() as pool:
        data = pool.map(read_cluster, [(cluster['matrix'], max_num_trigger_primitives) for cluster in clusters])

    data = np.array(data)
    assert data.shape[0] == len(clusters), f"Expected {len(clusters)} clusters, got {data.shape[0]}"

    return data


cc_clusters = load_clusters("data/charged_current.root")
cc_data = to_numpy(cc_clusters)
np.save("data/charged_current.npy", cc_data)

es_clusters = load_clusters("data/elastic_scatter.root")
es_data = to_numpy(es_clusters)
np.save("data/elastic_scatter.npy", es_data)