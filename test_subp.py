import subprocess
import os



exp_id="111"
TrackerName="tracktest112"
cmd=f"cd TrackEval && python scripts/run_mot_challenge.py --BENCHMARK FISH{exp_id} \
    --SPLIT_TO_EVAL train --TRACKERS_TO_EVAL {TrackerName} --METRICS HOTA --USE_PARALLEL False \
    # --NUM_PARALLEL_CORES 1 --DO_PREPROC False"
    # --NUM_PARALLEL_CORES 1 --DO_PREPROC False 1>/dev/null"
# cmd="cd TrackEval && ls"
# subprocess.run("pwd")
# subprocess.run([cmd]) 
os.system(cmd)
