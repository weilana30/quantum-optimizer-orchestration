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
    }
}

repo_root = "/Users/andrewweiland/UCCS_REU/quantum-optimizer-orchestration"
base_output_dir = "brute_force_chains/cheap_optimizers/full_run/one_opt_qiskit_ai"

runners = []

for opt_name, opt_config in optimizers.items():
    runners.append({
        "name": opt_name,
        "type": "chain",
        "steps": [copy.deepcopy(opt_config)],
        "save_intermediates": True,
        "output_dir": f"{base_output_dir}/reports/chains/{opt_name}",
    })

circuits = []

circuit_df = pd.read_csv(
    f"{repo_root}/dataset_analysis/circuits.csv"
)
circuit_df = circuit_df[circuit_df["source"] == "local"]
for _, row in circuit_df.iterrows():
    path = row["qasm_path"]
    name = row["name"]

    circuits.append({
        "name": name,
        "path": f"{repo_root}/{path}",
        "gate_set": "IBMN",
        "tags": ["experiment", name],
    })

output_dir = f"{base_output_dir}/reports"

config = {
    "metadata": {
        "description": "Single optimizer run using qiskit_ai only.",
        "job_info": "one_opt_qiskit_ai",
        "default_output_dir": f"{repo_root}/{output_dir}",
    },
    "circuits": circuits,
    "runners": runners,
    "metrics": ["depth", "two_qubit_gates", "two_qubit_depth", "total_gates"],
}

yaml_path = "one_opt_qiskit_ai.yaml"

with open(yaml_path, "w") as f:
    yaml.dump(config, f, sort_keys=False)

print(f"Wrote {yaml_path}")
print(f"  Circuits: {len(circuits)}")
print(f"  Runners: {len(runners)}")
print(f"  Num runs: {len(circuits) * len(runners)}")