import os
import glob
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_theme(style="darkgrid")

from multiprocessing import Pool
from tqdm import tqdm

IDS_OF_INTEREST = [int(x) for x in open("datasets/simbarca/metadata/sections_of_interest.txt", "r").read().split(",")]

def draw_od_demand(centroid_pos, od_pairs, drop_lower=None, save_folder="./"):
    
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
    fig.savefig("{}/figures/{}.pdf".format(save_folder, file_name), bbox_inches='tight')
    plt.close(fig)
    # print("file saved to {}/figures/{}.pdf".format(save_folder, file_name))

def draw_segment_speed_vs_point_speed(stats, start_min=0, end_min=20, sim_time_step=0.5, 
    save_note=None, save_folder="./"):
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
    fig.savefig("{}/figures/{}.pdf".format(save_folder, file_base_name), bbox_inches='tight')
    plt.close(fig)
    # print("file saved to {}/figures/{}.pdf".format(save_folder, file_base_name))
          
def draw_different_sim_runs(session_folders, section, save_folder="./"):
    
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
    fig.savefig("{}/figures/segment_speed_multi_runs.pdf".format(save_folder), bbox_inches='tight')
    plt.close(fig)
    # print("file saved to {}/figures/segment_speed_multi_runs.pdf".format(save_folder))

def draw_segment_speed_color_by_num_veh(stats, start_min=0, end_min=30, sim_time_step=0.5, 
    save_note=None, save_folder="./", figsize=(15, 4)):
    
    df = pd.DataFrame({
        "time_steps": stats["time_steps"],
        "total_dist": stats["total_dist"],
        "total_time": stats["total_time"],
        "num_vehicle": stats["num_vehicle"],
    })
    cycle_len = 1.5 # 1.5 minutes per cycle
    df_cp = df[(df['time_steps'] > int(start_min*60*1/sim_time_step)) &
               (df['time_steps'] <= int(end_min*60*1/sim_time_step))]
    ts = df_cp['time_steps']*sim_time_step/60 # time in minutes
    segment_speed = df_cp['total_dist'] / df_cp['total_time'] * 3.6 # km/h
    cycle_num = df_cp['time_steps'] // (cycle_len*60 / sim_time_step) # aggregate every 1.5 min
    cycle_group = df_cp.groupby(cycle_num)
    cycle_speed = cycle_group['total_dist'].sum() / cycle_group['total_time'].sum() * 3.6 # m/s => km/h
    cycle_time = (np.sort(cycle_num.unique()) + 1) * cycle_len # min, put the point at the end of the cycle time
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(cycle_time, cycle_speed, label="Cycle Average")
    im = ax.scatter(ts, segment_speed, s=3, label="Segment Speed", c=df_cp['num_vehicle'])
    fig.colorbar(im, ax=ax)
    # set title and legend 
    ax.set_title("Segment Speed Colored by Number of Vehicles")
    # put legend to upper right corner
    ax.legend(loc="upper right") 
    # set x and y axis 
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Speed (km/h)")
    # save pdf figure
    file_base_name = "segment_speed_color_by_num_veh{}".format("" if save_note is None else "_{}".format(save_note)) 
    fig.savefig("{}/figures/{}.pdf".format(save_folder, file_base_name), bbox_inches='tight')
    plt.close(fig)
    # print("file saved to {}/figures/{}.pdf".format(save_folder, file_base_name))
    
def compare_stats_with_agg(stats, agg, section_num, start_min=0, end_min=75, sim_time_step=0.5, save_folder="./", figsize=(15, 4)):
    sec_stats = stats[str(section_num)]
    df = pd.DataFrame({
        "time_steps": sec_stats["time_steps"],
        "total_dist": sec_stats["total_dist"],
        "total_time": sec_stats["total_time"],
        "num_vehicle": sec_stats["num_vehicle"],
    })
    cycle_len = 1.5 # 1.5 minutes per cycle
    df_cp = df[(df['time_steps'] > int(start_min*60*1/sim_time_step)) &
               (df['time_steps'] <= int(end_min*60*1/sim_time_step))]
    ts = df_cp['time_steps']*sim_time_step/60 # time in minutes
    segment_speed = df_cp['total_dist'] / df_cp['total_time'] * 3.6 # km/h
    cycle_num = df_cp['time_steps'] // (cycle_len*60 / sim_time_step) # aggregate every 1.5 min
    cycle_group = df_cp.groupby(cycle_num)
    cycle_speed = cycle_group['total_dist'].sum() / cycle_group['total_time'].sum() * 3.6 # m/s => km/h
    cycle_time = (np.sort(cycle_num.unique()) + 1) * cycle_len # min, put the point at the end of the cycle time
    
    sim_start_time = np.datetime64("2005-05-10 07:45:00")
    drone_speed = agg['drone_vdist'][section_num] / agg['drone_vtime'][section_num]
    time_from_start_min = (drone_speed.index - sim_start_time) / np.timedelta64(1, "m")
    interval_to_plot = (time_from_start_min > start_min) & (time_from_start_min <= end_min)
    ts_drone = time_from_start_min[interval_to_plot]
    drone_speed = drone_speed[interval_to_plot] * 3.6 # m/s => km/h
    
    pred_speed = agg['pred_vdist'][section_num] / agg['pred_vtime'][section_num]
    time_from_start_min = (pred_speed.index - sim_start_time) / np.timedelta64(1, "m")
    interval_to_plot = (time_from_start_min > start_min) & (time_from_start_min <= end_min)
    ts_pred = time_from_start_min[interval_to_plot]
    pred_speed = pred_speed[interval_to_plot] * 3.6 # m/s => km/h
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(cycle_time, cycle_speed, label="Cycle Average")
    ax.plot(ts_drone, drone_speed, label="Drone Speed")
    ax.plot(ts_pred, pred_speed, label="Predicted Speed")
    im = ax.scatter(ts, segment_speed, s=3, label="Segment Speed", c=df_cp['num_vehicle'])
    fig.colorbar(im, ax=ax)
    # set title and legend 
    ax.set_title("Segment Speed Colored by Number of Vehicles")
    # put legend to upper right corner
    ax.legend(loc="upper right") 
    # set x and y axis 
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Speed (km/h)")
    # save pdf figure
    file_base_name = "raw_stats_vs_agg_{}".format(section_num) 
    fig.savefig("{}/figures/{}.pdf".format(save_folder, file_base_name), bbox_inches='tight')
    plt.close(fig)
    # print("file saved to {}/figures/{}.pdf".format(save_folder, file_base_name))


def compare_agg_with_sample(agg, sample, section_num, section_id_to_index, start_min=0, end_min=75, sim_time_step=0.5, save_folder="./", figsize=(15, 4)):
    
    sim_start_time = np.datetime64("2005-05-10 07:45:00")
    drone_speed = agg['drone_vdist'][section_num] / agg['drone_vtime'][section_num]
    time_from_start_min = (drone_speed.index - sim_start_time) / np.timedelta64(1, "m")
    interval_to_plot = (time_from_start_min > start_min) & (time_from_start_min <= end_min)
    ts_drone = time_from_start_min[interval_to_plot]
    drone_speed = drone_speed[interval_to_plot] * 3.6 # m/s => km/h
    
    pred_speed = agg['pred_vdist'][section_num] / agg['pred_vtime'][section_num]
    time_from_start_min = (pred_speed.index - sim_start_time) / np.timedelta64(1, "m")
    interval_to_plot = (time_from_start_min > start_min) & (time_from_start_min <= end_min)
    ts_pred = time_from_start_min[interval_to_plot]
    pred_speed = pred_speed[interval_to_plot] * 3.6 # m/s => km/h
    
    b, i = 2, section_id_to_index[section_num]
    drone_speed_sample = (sample['drone_vdist'][b, :, i, 0] / sample['drone_vtime'][b, :, i, 0]) * 3.6 # m/s => km/h
    time_in_day = sample['drone_vdist'][b, :, i, 1] # a number from 0 to 1 representing time in a day
    ts_drone_sample = (time_in_day * 24 - 7.75) * 60 # 7:45 is the start time, 7.75 hours from 0:00
    pred_speed_sample = (sample['pred_vdist'][b, :, i, 0] / sample['pred_vtime'][b, :, i, 0]) * 3.6 # m/s => km/h
    pred_time_in_day = sample['pred_vdist'][b, :, i, 1]
    ts_pred_sample = (pred_time_in_day * 24 - 7.75) * 60
        
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(ts_drone, drone_speed, label="Drone Speed")
    ax.plot(ts_pred, pred_speed, label="Predicted Speed")
    ax.plot(ts_drone_sample, drone_speed_sample, label="Input Drone Speed B2")
    ax.plot(ts_pred_sample, pred_speed_sample, label="Train Label Speed B2")
    # set title and legend
    ax.set_title("Speed Comparison between Aggregated and Sampled Data")
    # put legend to upper right corner
    ax.legend(loc="upper right")
    # set x and y axis
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Speed (km/h)")
    # save pdf figure
    file_base_name = "agg_vs_sample_{}".format(section_num)
    fig.savefig("{}/figures/{}.pdf".format(save_folder, file_base_name), bbox_inches='tight')
    plt.close(fig)
    # print("file saved to {}/figures/{}.pdf".format(save_folder, file_base_name))

if __name__ == "__main__":
    
    data_root = "datasets/simbarca"
    metadata_dir = "{}/metadata".format(data_root)
    
    session_folders = sorted(glob.glob("{}/simulation_sessions/session_*".format(data_root)))
    
    def _draw_one_folder(folder):
        # print("Processing folder: {}".format(folder))
        
        if not os.path.exists("{}/figures".format(folder)):
            os.mkdir("{}/figures".format(folder))
        # the first three are main roads, the last two are side roads
        section_ids = pd.read_csv("{}/link_bboxes_clustered.csv".format(metadata_dir))['id']
        section_id_to_index = {section_id: index for index, section_id in enumerate(sorted(section_ids))}
        
        section_stats = json.load(open("{}/timeseries/section_statistics.json".format(folder)))

        stats = section_stats['statistics']
        for section_num in IDS_OF_INTEREST:
            draw_segment_speed_color_by_num_veh(stats[str(section_num)], start_min=0, end_min=150, save_note="sec_{}".format(section_num), save_folder=folder, figsize=(25, 4))

        
        draw_segment_speed_vs_point_speed(stats['9971'], save_note="short_queue", save_folder=folder)
        draw_segment_speed_vs_point_speed(stats['9453'], save_note="long_queue", save_folder=folder)
        
        # visualize aggregated time series and generated training samples, compare with section_stats
        agg_ts_data = pickle.load(open("{}/timeseries/agg_timeseries.pkl".format(folder), "rb"))
        sample_data_npz = np.load(open("{}/timeseries/samples.npz".format(folder), "rb"))
        sample_data = {key: sample_data_npz[key] for key in sample_data_npz.keys()}
        
        for section_num in IDS_OF_INTEREST:
            compare_stats_with_agg(stats, agg_ts_data, section_num, start_min=0, end_min=150, save_folder=folder, figsize=(25, 4))
            compare_agg_with_sample(agg_ts_data, sample_data, section_num, section_id_to_index, start_min=0, end_min=150, save_folder=folder, figsize=(25, 4))
    
    def draw_one_folder(folder):
        """go to the next one if there is an error."""
        try:
            data = _draw_one_folder(folder)
            return data
        except Exception as e:
            print("Error in folder: {}".format(folder))
            print(e)
            return None
        
    with Pool(processes=8) as pool:
        all_data = list(tqdm(pool.imap(draw_one_folder, session_folders), total=len(session_folders)))
        
    # looks like putting this part below the multiprocessing will speed up the multiprocessing part, not sure about the casue
    # but is still better because the child processes won't have to worry about these variables in the parent process
    centroid_pos = pd.read_csv("{}/centroid_pos.csv".format(metadata_dir), header=0, index_col=0)
    od_pairs = json.load(open("{}/od_pairs.json".format(metadata_dir)))['od_pairs']
    
    draw_od_demand(centroid_pos, od_pairs, save_folder=data_root)
    draw_od_demand(centroid_pos, od_pairs, drop_lower=0.9, save_folder=data_root)
    draw_different_sim_runs(session_folders, section="9971", save_folder=data_root)