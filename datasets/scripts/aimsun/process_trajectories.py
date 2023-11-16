import glob 
import subprocess

paths = sorted(glob.glob("/home/weijiang/Projects/Netsanut/datasets/simbarca/session_*"))

for path in paths:
    cmd = "python /home/weijiang/Projects/Netsanut/datasets/scripts/aimsun/time_series_from_traj.py --metadata_folder /home/weijiang/Projects/Netsanut/datasets/simbarca/metadata --session_folder {}".format(path)
    print("Executing command: \n {}".format(cmd))
    subprocess.run(cmd, shell=True)