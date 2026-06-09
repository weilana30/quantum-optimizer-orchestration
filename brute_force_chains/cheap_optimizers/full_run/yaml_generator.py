from itertools import product
import yaml
import pandas as pd

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
runners = []

for chain_len in range(1, 4):
    for chain in product(optimizers.keys(), repeat=chain_len):
        runners.append({
            "name": "__".join(chain),
            "type": "chain",
            "steps": [optimizers[opt] for opt in chain],
            "save_intermediates": True,
            "output_dir": f"brute_force_chains/cheap_optimizers/initial_test/reports/chains/{'__'.join(chain)}",
        })

circuits = []
circuit_df = pd.read_csv("/Users/andrewweiland/UCCS_REU/quantum-optimizer-orchestration/dataset_analysis/circuits.csv")
local = circuit_df[circuit_df["source"] == "local"]
unique_paths = local["qasm_path"].unique()
for path in unique_paths:
    name = path.split("/")[-1].split(".")[0]
    circuits.append({
        "name": name,
        "path": path,
        "gate_set": "IBMN",
        "tags": ["experiment", name],
    })



config = {
    "metadata": {
        "description": "Sample chain test using qiskit_ai, qiskit_standard, and tket only.",
        "job_info": "fast_chain_sample",
        "default_output_dir": "brute_force_chains/cheap_optimizers/initial_test/reports/",
    },
    "circuits": [
        {
            "name": "qft_8",
            "path": f"{repo_root}/benchmarks/ai_transpile/qasm/qft_8.qasm",
            "gate_set": "IBMN",
            "tags": ["sample", "qft"],
        }
    ],
    "runners": runners,
    "metrics": ["depth", "two_qubit_gates", "two_qubit_depth", "total_gates"],
}

with open("cheap_optimizer_test.yaml", "w") as f:
    yaml.dump(config, f, sort_keys=False)