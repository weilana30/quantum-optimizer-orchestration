from itertools import product
import yaml
import pandas as pd
import copy

optimizers = {
    "qiskit_ai": {
        "type": "qiskit_ai",
        "name": "qiskit_ai",
        "optimization_levels": [3],
        "iterations_per_level": 1,
        "layout_mode": "optimize",
    },
    "qiskit_standard": {
        "type": "qiskit_standard",
        "name": "qiskit_standard",
        "optimization_levels": [3],
    },
    "tket": {
        "type": "tket",
        "name": "tket",
        "gate_set": "IBMN",
    },
}

repo_root = "/Users/andrewweiland/UCCS_REU/quantum-optimizer-orchestration"
base_output_dir = "brute_force_chains/cheap_optimizers/full_run/two_opt"

runners = []

for chain_len in range(1, 2):
    for chain in product(optimizers.keys(), repeat=chain_len):
        chain_name = "__".join(chain)
        runners.append({
            "name": chain_name,
            "type": "chain",
            "steps": [copy.deepcopy(optimizers[opt]) for opt in chain],
            "save_intermediates": True,
            "output_dir": f"{base_output_dir}/reports/chains/{chain_name}",
        })

circuits = []
circuit_df = pd.read_csv(
    f"{repo_root}/brute_force_chains/cheap_optimizers/full_run/cheap_single_opts.csv"
)

for _, row in circuit_df.iterrows():
    path = row["artifact_path"]
    name = row["opt_chain"]

    circuits.append({
        "name": name,
        "path": f"{repo_root}/{path}",
        "gate_set": "IBMN",
        "tags": ["experiment", name],
    })

# Split into two batches
mid = len(circuits) // 2
batches = [
    ("batch_1", circuits[:mid]),
    ("batch_2", circuits[mid:]),
]

for batch_name, batch_circuits in batches:
    output_dir = f"{base_output_dir}/{batch_name}/reports"

    config = {
        "metadata": {
            "description": "Second optimizer batch test using qiskit_ai, qiskit_standard, and tket only.",
            "job_info": f"cheap_2_opt_{batch_name}",
            "default_output_dir": f"{repo_root}/{output_dir}",
        },
        "circuits": batch_circuits,
        "runners": runners,
        "metrics": ["depth", "two_qubit_gates", "two_qubit_depth", "total_gates"],
    }

    yaml_path = f"cheap_second_opt_{batch_name}.yaml"

    with open(yaml_path, "w") as f:
        yaml.dump(config, f, sort_keys=False)

    print(f"Wrote {yaml_path}")
    print(f"  Circuits: {len(batch_circuits)}")
    print(f"  Runners: {len(runners)}")
    print(f"  Num chains: {len(batch_circuits) * len(runners)}")