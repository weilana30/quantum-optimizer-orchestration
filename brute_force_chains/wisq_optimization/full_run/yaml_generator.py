from itertools import product
import yaml
import pandas as pd
import math

optimizers = {
    "wisq_rules": {
        "type": "wisq",
        "name": "wisq_rules",
        "target_gateset": "IBMN",
        "optimization_objective": "TWO_Q",
        "approx_epsilon": 0,
        "opt_timeout": 180,
    },
    "wisq_bqskit": {
        "type": "wisq",
        "name": "wisq_bqskit",
        "target_gateset": "IBMN",
        "optimization_objective": "TWO_Q",
        "approx_epsilon": 1e-10,
        "opt_timeout": 180,
    },
}

repo_root = "/Users/andrewweiland/UCCS_REU/quantum-optimizer-orchestration"
base_output_dir = "brute_force_chains/wisq_optimization/full_run/one_opt_wisq/reports/"

# Build runners
runners = []

for chain_len in range(1, 2):  # length 1 only
    for chain in product(optimizers.keys(), repeat=chain_len):
        chain_name = "__".join(chain)

        runners.append({
            "name": chain_name,
            "type": "chain",
            "steps": [optimizers[opt] for opt in chain],
            "save_intermediates": True,
            "output_dir": f"{base_output_dir}chains/{chain_name}",
        })

# Build circuits
circuit_df = pd.read_csv(
    "/Users/andrewweiland/UCCS_REU/quantum-optimizer-orchestration/dataset_analysis/circuits.csv"
)

local = circuit_df[circuit_df["source"] == "local"]
unique_paths = list(local["qasm_path"].unique())

circuits = []
for path in unique_paths:
    name = path.split("/")[-1].replace(".qasm", "")
    circuits.append({
        "name": name,
        "path": repo_root + "/" + path,
        "gate_set": "IBMN",
        "tags": ["experiment", name],
    })

# Split circuits into 3 batches
num_batches = 3
batch_size = math.ceil(len(circuits) / num_batches)

for batch_idx in range(num_batches):
    batch_circuits = circuits[
        batch_idx * batch_size : (batch_idx + 1) * batch_size
    ]

    batch_num = batch_idx + 1
    batch_output_dir = (
        f"brute_force_chains/wisq_optimization/full_run/"
        f"one_opt_wisq/batch_{batch_num}/reports/"
    )

    # Update runner output dirs per batch
    batch_runners = []
    for runner in runners:
        chain_name = runner["name"]
        batch_runner = runner.copy()
        batch_runner["output_dir"] = f"{batch_output_dir}chains/{chain_name}"
        batch_runners.append(batch_runner)

    config = {
        "metadata": {
            "description": "One-optimizer WISQ test using wisq_rules and wisq_bqskit only.",
            "job_info": f"cheap_1_opt_wisq_batch_{batch_num}",
            "default_output_dir": repo_root + "/" + batch_output_dir,
            "runner_timeout_seconds": 300,
        },
        "circuits": batch_circuits,
        "runners": batch_runners,
        "metrics": [
            "depth",
            "two_qubit_gates",
            "two_qubit_depth",
            "total_gates",
        ],
    }

    yaml_name = f"wisq_one_opt_batch_{batch_num}.yaml"

    with open(yaml_name, "w") as f:
        yaml.dump(config, f, sort_keys=False)

    print(
        f"Wrote {yaml_name}: "
        f"{len(batch_circuits)} circuits × {len(batch_runners)} runners = "
        f"{len(batch_circuits) * len(batch_runners)} runs"
    )