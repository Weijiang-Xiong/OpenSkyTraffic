import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def draw_od_demand(centroid_pos, od_pairs, drop_lower=None):
    
    # drop the xx percent entires with lowest demand
    if drop_lower is not None:
        od_pairs = sorted(od_pairs, key=lambda x: x[2], reverse=False)[int(len(od_pairs)*drop_lower):]
    
    centroid_pos["x"] = (centroid_pos["x"] - centroid_pos["x"].min()) / 100
    centroid_pos["y"] = (centroid_pos["y"] - centroid_pos["y"].min()) / 100

    fig, ax = plt.subplots()
    # draw the centroid points
    ponit_x, point_y = centroid_pos["x"], centroid_pos["y"]
    ax.scatter(ponit_x, point_y, s=5, color='navy', zorder=2, label="centroid")
    # for each item in od_pairs, draw a line between the two points
    for idx, (org, dst, num_veh) in enumerate(od_pairs):
        xs = [centroid_pos.loc[org].x, centroid_pos.loc[dst].x]
        ys = [centroid_pos.loc[org].y, centroid_pos.loc[dst].y]
        if idx == 0:
            ax.plot(xs, ys, color="royalblue", linewidth=0.5*np.log10(num_veh)+1, 
                    alpha=0.2, zorder=1, label="demand")
        else:
            ax.plot(xs, ys, color="royalblue", linewidth=0.5*np.log10(num_veh)+1, 
                    alpha=0.2, zorder=1)
    # set title and legend
    ax.set_title("Centroid Positions and Traffic Demand")
    ax.legend()
    # hide x and y axis
    ax.set_xticks([])
    ax.set_yticks([])

    # save pdf figure
    file_name = "centroid_pos" if drop_lower is None else "centroid_pos_drop_lower_{}".format(drop_lower)
    fig.savefig("{}/figures/{}.pdf".format(data_root, file_name), bbox_inches='tight')
    print("file saved to {}/figures/{}.pdf".format(data_root, file_name))

def draw_segment_speed_vs_point_speed(stats, save_note=None, start_min=15, end_min=20, sim_time_step=0.5):
    df = pd.DataFrame({
        "time_steps": stats["time_steps"],
        "total_dist": stats["total_dist"],
        "total_time": stats["total_time"],
        "num_vehicle": stats["num_vehicle"],
        "LD_count": stats["LD_count"],
        "LD_speed": stats["LD_speed"]
    })
    df_cp = df[(df['time_steps'] > int(start_min*60*1/sim_time_step)) &
               (df['time_steps'] < int(end_min*60*1/sim_time_step))]
    ts = df_cp['time_steps']*sim_time_step/60 # time in minutes
    segment_speed = df_cp['total_dist'] / df_cp['total_time'] * 3.6 # km/h
    # time steps with valid loop detector data
    ts_ld_speed = df_cp['time_steps'][df_cp['LD_count'] > 0]*sim_time_step/60 
    ld_speed = df_cp['LD_speed'][df_cp['LD_count'] > 0] * 3.6 # km/h
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(ts, segment_speed, s=3, label="segment speed")
    ax.scatter(ts_ld_speed, ld_speed, s=3, label="point speed")
    # set title and legend 
    ax.set_title("Segment Speed vs. Point Speed")
    # put legend to upper right corner
    ax.legend(loc="upper right") 
    # set x and y axis 
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Speed (km/h)")
    # save pdf figure
    file_base_name = "segment_speed_vs_point_speed{}".format("" if save_note is None else "_{}".format(save_note)) 
    fig.savefig("{}/figures/{}.pdf".format(data_root, file_base_name), bbox_inches='tight')
    print("file saved to {}/figures/{}.pdf".format(data_root, file_base_name))
          
def draw_different_sim_runs(session_folders, section):
    
    ts_all_runs, segment_speed_all_runs = [], []
    
    for folder in np.random.choice(session_folders, 10, replace=False):
        section_stats = json.load(open("{}//timeseries/section_statistics.json".format(folder)))
        stats = section_stats['statistics'][section]
        df = pd.DataFrame({
            "time_steps": stats["time_steps"],
            "total_dist": stats["total_dist"],
            "total_time": stats["total_time"],
            "num_vehicle": stats["num_vehicle"],
            "LD_count": stats["LD_count"],
            "LD_speed": stats["LD_speed"]
        })
        df_cp = df[(df['time_steps'] > int(15*60*2) ) & (df['time_steps'] < int(20*60*2))]
        ts = df_cp['time_steps']*0.5/60 # time steps in minutes
        segment_speed = df_cp['total_dist'] / df_cp['total_time'] * 3.6 # km/h
        ts_all_runs.append(ts)
        segment_speed_all_runs.append(segment_speed)
        
    fig, ax = plt.subplots(figsize=(10, 4))
    for ts, segment_speed in zip(ts_all_runs, segment_speed_all_runs):
        ax.scatter(ts, segment_speed, s=3)
    # set title and legend
    ax.set_title("Segment Speed in Different Simulation Runs")
    # set x and y axis
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Speed (km/h)")
    # save pdf figure
    fig.savefig("{}/figures/segment_speed_multi_runs.pdf".format(data_root), bbox_inches='tight')
    print("file saved to {}/figures/segment_speed_multi_runs.pdf".format(data_root))

if __name__ == "__main__":
    
    data_root = "datasets/simbarca"
    metadata_dir = "{}/metadata".format(data_root)

    centroid_pos = pd.read_csv("{}/centroid_pos.csv".format(metadata_dir), header=0, index_col=0)
    od_pairs = json.load(open("{}/od_pairs.json".format(metadata_dir)))['od_pairs']
    
    draw_od_demand(centroid_pos, od_pairs)
    draw_od_demand(centroid_pos, od_pairs, drop_lower=0.9)
    
    session_folders = sorted(glob.glob("{}/simulation_sessions/session_*".format(data_root)))
    section_stats = json.load(open("{}/timeseries/section_statistics.json".format(session_folders[0])))

    draw_segment_speed_vs_point_speed(section_stats['statistics']['9971'], save_note="short_queue")
    draw_segment_speed_vs_point_speed(section_stats['statistics']['9453'], save_note="long_queue")
    
    draw_different_sim_runs(session_folders, section="9971")
    
    