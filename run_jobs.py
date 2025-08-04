import os, time

group = 0

batch_name = "min_elem_6_constrained_bh_wa_average_difference"

groups_list = [int(csv_file.split(".")[0]) for csv_file in os.listdir(f"data/sweep_configs/{batch_name}")]
num_groups = max(groups_list) + 1

user = os.getenv("USER")

num_running_or_queued = int(os.popen(f"squeue -u {user} | wc -l").read().strip()) - 1

while group < num_groups:
    while num_running_or_queued > 0:
        num_running_or_queued = (
            int(os.popen(f"squeue -u {user} | wc -l").read().strip()) - 1
        )
        print(
            f"On batch {group - 1}, Still running {num_running_or_queued} jobs"
        )
        time.sleep(600)

    print(f"Submitting batch #{group}")
    os.system(f"sbatch run_batch_discovery.sh {group} {batch_name}")
    time.sleep(3)
    num_running_or_queued = (
        int(os.popen(f"squeue -u {user} | wc -l").read().strip()) - 1
    )
    group += 1

