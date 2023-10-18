import glob
import shutil
import subprocess

executable = "/opt/Aimsun_Next_22/aconsole"
project = "/home/weijiang/Downloads/Eixample_BASE_v7.ang"
rep_id = "10607425"
folder_pattern = "/home/weijiang/Projects/Netsanut/datasets/simbarca/session_*"

all_folders = glob.glob(folder_pattern)
for folder in all_folders:
    project_copy_path = "{}/{}".format(folder, "project_copy.ang")
    shutil.copyfile(project, project_copy_path)
    command = "cd {} && {} --project {} --command execute --target {} --force_number_of_threads 16".format(
        folder, executable, project_copy_path, rep_id)
    print("Executing command \n {}".format(command))
    subprocess.run(command, shell=True)
    
    