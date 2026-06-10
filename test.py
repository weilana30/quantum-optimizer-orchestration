from benchmarks.ai_transpile.transpilers import transpile_with_qiskit_ai
import time
from qiskit import qasm2

path = "benchmarks/ai_transpile/qasm/feynman/hwb12.qasm"
qc = qasm2.load(path)

start = time.time()
results = transpile_with_qiskit_ai(qc)
print("seconds:", time.time() - start)

for r in results:
    print(r.label, r.circuit.depth(), r.circuit.num_nonlocal_gates(), r.circuit.size())