---
category: sop
conditions:
  domain: hccl
  hardware:
  - Ascend 910B
  - Ascend 910C
domain_path: hccl/benchmark/hccl-test-suite/mpi-allreduce-and-alltoallv
evidence_strength: software_contract
id: unit-hccl-test-mpich-compilation-and-collective-bandwidth-benchmark-sop
inherits_from:
- unit-ascend-hccl-roce-hccs-topology-and-ranktable-axioms
provenance:
  evidence_refs:
  - https://github.com/Ascend/agent-skills/blob/942f0138aa9c78467507babb09aa9c64c5f534dc/skills/hccl-test/SKILL.md#L1-L260
  - https://github.com/Ascend/agent-skills/blob/942f0138aa9c78467507babb09aa9c64c5f534dc/skills/hccl-test/references/parameters.md#L1-L185
  - https://github.com/Ascend/agent-skills/blob/942f0138aa9c78467507babb09aa9c64c5f534dc/skills/hccl-test/scripts/setup-hccl-env.sh#L1-L95
  source_locator: Ascend/agent-skills@942f0138aa9c78467507babb09aa9c64c5f534dc:skills/hccl-test/SKILL.md
  source_type: source_contract
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
status: active
tier: L2
title: Ascend hccl-test MPICH Compilation, Multi-Node Collective Launch & Algorithm/Protocol
  Matrix SOP
triggers:
  error_signatures:
  - check_err
  - all_reduce_test failed
  - HCCL_ALGO
  keywords:
  - hccl-test
  - mpich
  - mpirun
  - all-reduce-test
  - alltoallv-test
  - hccl-algo
  - busbw
  - roce
  - hccs
---
## Preconditions & Trigger

Apply this SOP when compiling and running the CANN `hccl_test` benchmark suite (`all_reduce_test`, `all_gather_test`, `reduce_scatter_test`, `alltoall_test`, `alltoallv_test`, `broadcast_test`) across single-node HCCS or multi-node RoCE Ascend NPU clusters to baseline collective bus bandwidth (`busbw`) before launching distributed training or inference.

## Compilation & Environment Setup (`setup-hccl-env.sh`)

1. **Source CANN Toolkit & Locate `hccl_test`**:
   - Source the CANN environment (`source /usr/local/Ascend/ascend-toolkit/set_env.sh` where `ASCEND_DIR=/usr/local/Ascend/ascend-toolkit/latest`).
   - Verify `$ASCEND_DIR/tools/hccl_test` exists and contains `Makefile` and `common/src`.
2. **Compile MPICH & Build `hccl_test` Binaries**:
   - Build MPICH (`mpich-3.2.1` or system `mpich`/`openmpi` matching cluster launch tooling) into `MPI_HOME=/usr/local/mpich` and export:
     ```bash
     export MPI_HOME=/usr/local/mpich
     export ASCEND_DIR=/usr/local/Ascend/ascend-toolkit/latest
     export PATH=$MPI_HOME/bin:$PATH
     export LD_LIBRARY_PATH=$MPI_HOME/lib:$ASCEND_DIR/lib64:$LD_LIBRARY_PATH
     ```
   - Compile the benchmark suite from `$ASCEND_DIR/tools/hccl_test`:
     ```bash
     cd "$ASCEND_DIR/tools/hccl_test"
     make MPI_HOME=$MPI_HOME ASCEND_DIR=$ASCEND_DIR
     ```
   - Confirm the compiled binaries are produced under `$ASCEND_DIR/tools/hccl_test/bin/` (`all_reduce_test`, `all_gather_test`, `reduce_scatter_test`, `alltoall_test`, `alltoallv_test`, `broadcast_test`, `reduce_test`, `scatter_test`).

## CLI Parameter Semantics & Launch Equations

1. **Rank-to-Device Mapping (`-n` vs. `-p`)**:
   - In `mpirun -n <total_mpi_procs> ... -p <npus_per_node>`:
     - `-p <npus_per_node>` tells `hccl_test` how many local NPU devices each node uses (`device_id = local_rank % npus_per_node`).
     - For a single node with 8 NPUs: `mpirun -n 8 ./bin/all_reduce_test -b 8K -e 1G -f 2 -d fp32 -o sum -p 8`.
     - For $M$ nodes with 8 NPUs per node using a hostfile (`each line: <host_ip>:8`): `mpirun -f hostfile -n $((M * 8)) ./bin/all_reduce_test -b 8K -e 1G -f 2 -d bf16 -o sum -p 8 -c 1`.
   - **Key CLI Flags**:
     - `-b <minbytes>` / `-e <maxbytes>`: Message payload sweep bounds (e.g., `-b 8K -e 4G`).
     - `-f <stepfactor>`: Multiplicative step factor between message sizes (typically `-f 2`).
     - `-d <datatype>`: `int8`, `int16`, `int32`, `int64`, `fp16`, `fp32`, `bf16` (note: `bf16` is supported only on Ascend 910B / 910C / 950).
     - `-o <op>`: Reduction operator (`sum`, `prod`, `min`, `max`; omit for `all_gather_test`, `alltoall_test`, `alltoallv_test`, `broadcast_test`).
     - `-w <warmup_iters>` / `-n <iters>`: Warmup iterations (default `5`) and timed measurement iterations (default `20`).
     - `-c <check>`: Result verification (`-c 1` enables host-side numerical verification against expected reduction outputs; `-c 0` disables verification for pure timing runs).

## Collective Algorithm Matrix (`HCCL_ALGO`) & Bus Bandwidth Interpretation

1. **Select Optimal Algorithm per Topology**:
   - Control algorithm selection per collective stage (e.g., `level0` intra-node HCCS, `level1` inter-node RoCE) via `HCCL_ALGO` or `HCCL_INTRA_ROCE_ENABLE`:
     - **`HD` (Halving-Doubling)**: Lowest latency for small messages (`< 1 MB`) where step count $\log_2(N)$ dominates.
     - **`RING`**: Maximizes sustained bandwidth for large messages (`>= 64 MB`) over RoCE/HCCS rings.
     - **`MESH`**: Optimal on full-mesh HCCS interconnects (e.g., 8-NPU A2 full-mesh or A3 superpod intra-node).
     - **`NB` (Non-Blocking) / `PIPELINE`**: Overlaps level0 HCCS and level1 RoCE phases across multi-node hierarchies.
     - **`PAIRWISE`**: Mandatory for large-scale `alltoall_test` / `alltoallv_test` MoE expert dispatch benchmarking.
2. **Bus Bandwidth (`busbw`) Acceptance Criteria**:
   - Inspect the `Transitbw(GB/s)` / `Busbw(GB/s)` columns in the output table. On healthy 8-NPU Ascend 910B/910C nodes, intra-node HCCS `all_reduce_test` at `1G` payload should achieve consistent `Busbw` without drops across consecutive `-n 20` iterations, and `-c 1` must report `check_err: 0`.

## Verification

- Run `mpirun -n 8 "$ASCEND_DIR/tools/hccl_test/bin/all_reduce_test" -b 1M -e 512M -f 2 -d bf16 -o sum -p 8 -c 1` and verify exit code `0` with zero numerical check errors across all message sizes.
