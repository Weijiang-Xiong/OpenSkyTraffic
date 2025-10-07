""" This script selects train and test sessions from the simulation. 
"""
import os
import re
import json
import numpy as np
from glob import glob
from argparse import ArgumentParser

def get_session_number(path):
    return int(re.search(r"session_(\d+)", path).group(1))

def check_errors(folder):
    """ Check the log files to see if anything wrong happened during the simulation and processing
    """
    errors = []
    api_log_file = "{}/api_log.log".format(folder)
    # check API log file to see if simulation finished correctly
    if not os.path.exists(api_log_file):
        errors.append("SIMULATION_NOT_STARTED")
    else:
        with open(api_log_file, "r") as f:
            
            api_log_content = f.read()
            
            if not "Simulation Finished" in api_log_content:
                errors.append("SIMULATION_NOT_FINISHED")
            
            # started a new simulation without cleaning up the previous one
            if len(re.findall("Simulation Ready", api_log_content)) > 1:
                errors.append("POSSIBLE_DUPLICATE_SIMULATION")
            
            pattern = r": (\d+) vehicles in network at time (\d+\.?\d*) min (\d+\.?\d*) s"
            matches = re.findall(pattern, api_log_content)
            if len(matches) == 0:
                errors.append("NO_VEHICLE_COUNT") # this usually don't happen
            else:
                if int(matches[-1][0]) > 1:
                    # there are vehicles stuck in the netwrok at the end of the simulation
                    errors.append("UNRESOLVED_CONGESTION")
                    
    # check processing log file
    processing_log_file = "{}/processing_log.log".format(folder)
    if not os.path.exists(processing_log_file):
        errors.append("PROCESSING_NOT_STARTED")
    else:
        with open(processing_log_file, "r") as f:
            log_content = f.read()
            if "error" in log_content.lower() or "interrupt" in log_content.lower():
                errors.append("PROCESSING_ERROR")
    
    return errors

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--session-folder", type=str, default="datasets/simbarca/simulation_sessions")    
    parser.add_argument("--output-folder", type=str, default="datasets/simbarca/metadata")
    args = parser.parse_args()
    
    train_ratio = args.train_ratio
    all_folders = glob("{}/session_*".format(args.session_folder))

    print("Checking the log files for any errors...")
    folders_with_errors = []
    for folder in all_folders:
        errors = check_errors(folder)    
        if len(errors) > 0:
            print("Folder {} has errors: {}".format(folder, errors))
            folders_with_errors.append(folder)
    folders_with_errors = sorted(folders_with_errors)
    
    # we will not use the sessions with errors
    valid_sessions = np.setdiff1d(all_folders, folders_with_errors)
    # randomly select train and test sessions
    rng = np.random.RandomState(42)
    train_idx = np.argsort(rng.rand(len(valid_sessions)))[:int(train_ratio * len(valid_sessions))]
    test_idx = np.setdiff1d(np.arange(len(valid_sessions)), train_idx)
    
    train_session_numbers = [get_session_number(p) for p in np.array(valid_sessions)[train_idx]]
    test_sessions = [get_session_number(p) for p in np.array(valid_sessions)[test_idx]]

    # write the train-test split into a json file 
    with open("{}/train_test_split.json".format(args.output_folder), "w") as f:
        json.dump({"train": sorted(train_session_numbers), "test": sorted(test_sessions)}, f)