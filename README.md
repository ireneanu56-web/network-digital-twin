# Network Digital Twin MVP

## Intelligent Fault Detection

The project combines a statistical detector with three lightweight machine-learning
experiments. All models use the existing `data/network_data.csv` telemetry and the
fault labels `NORMAL`, `LATENCY_FAULT`, and `PACKET_LOSS_FAULT`.

- **Statistical threshold baseline:** `monitoring/fault_detector.py` derives
  explainable thresholds from baseline samples and applies ordered rules to classify
  each telemetry row.
- **GRU temporal fault detector:** `models/gru_detector.py` classifies short,
  three-sample telemetry windows. Windows are built separately inside each
  train/validation/test split to avoid crossing split boundaries.
- **Autoencoder anomaly detector:** `models/autoencoder_detector.py` learns to
  reconstruct normal training telemetry. A threshold from normal validation
  reconstruction errors distinguishes `NORMAL` from `ANOMALY`; it does not identify
  the specific fault category.
- **GNN topology-aware fault predictor:** `models/gnn_detector.py` applies message
  passing on the existing Mininet device/link graph and predicts the network-wide
  fault label. The current CSV has network-wide measurements only, not telemetry
  attributed to individual links or devices, so this MVP cannot localize a fault
  within the graph.

Preprocessing imputes missing feature values and fits scaling using training rows
only. Deterministic chronological holdouts are made within each label so adjacent
time-series rows are less likely to leak across train, validation, and test sets.
Model seeds are fixed. Evaluation reports held-out test metrics and confusion
matrices. Autoencoder metrics are binary (normal vs anomaly) and should not be
compared as if they were three-class fault-identification results.

The current dataset has only 59 labeled samples, and some older baseline rows lack
throughput and retransmission measurements. Results are preliminary MVP experiments;
they are not evidence of publication-level performance or broad generalization.

Run the ML workflow from the repository root:

```bash
python -m models.train_models
python -m models.evaluate_models
streamlit run dashboard/app.py
```

Training saves the models under `models/saved/`. Evaluation appends its comparison
metrics to `results/ml_results.csv` and creates timestamped figures under
`results/plots/`.

## Performance Optimization and Self-Healing

The self-healing MVP uses the existing redundant Mininet topology:
`h1-s1-s2-s4-h4` and `h1-s1-s3-s4-h4`. The controlled experiment measures baseline
traffic, applies a 50 ms `tc/netem` delay to the active `s1-s2` link, measures the
fault condition, and attempts recovery through the alternate `s1-s3-s4` branch.

1. The experiment reuses the statistical detector to classify measured fault
   telemetry against baseline thresholds.
2. The injected link is identified explicitly by the controlled experiment. The
   existing dataset contains network-wide metrics, not per-link telemetry, so
   automatic physical-link localization is not available in this MVP.
3. The digital-twin NetworkX graph removes the reported faulty link for route
   search.
4. NetworkX finds the shortest available alternate path.
5. The program resolves live Mininet interface port numbers and installs
   host-pair-specific OpenFlow rules with `ovs-ofctl` on the involved OVS switches.
6. Bidirectional traffic for the selected source/destination pair follows the
   alternate path; rerouting is reported as successful only after flow installation
   succeeds.
7. Ping and iperf3 metrics are measured before injection, during the fault, and
   after the route change.
8. Throughput change, RTT change, packet-loss reduction, decision time, and total
   recovery time are reported from measured values. No improvement is assumed.

This is a rule-based self-healing MVP, not reinforcement learning. The experiment
starts and stops its own Mininet instance, cleans up the injected `tc` qdisc,
removes the host-pair OpenFlow rules, terminates its iperf3 server, and stops
Mininet on completion or Ctrl+C. Stop any manually running Mininet network first,
because this demo uses the same switch names.

Run from the repository root on Linux/WSL with Mininet, Open vSwitch, `iperf3`, and
`tc` installed:

```bash
sudo apt install mininet openvswitch-switch iperf3 iproute2 python3-networkx python3-pandas python3-matplotlib
sudo python3 -m optimization.evaluate_optimization
python3 -m optimization.plot_results
streamlit run dashboard/app.py
```

The Mininet experiment needs these system packages available to the same Python
interpreter used by `sudo python3`; `networkx` is also included in
`requirements.txt`.

Each successful experiment appends one row to
`results/optimization_results.csv`. Plot generation reads only that file and writes
four figures into `results/plots/`. The experiment reports an error if Mininet's
Python modules are unavailable to the selected `python3` interpreter.
