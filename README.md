# Graph Attention Network for Multi-Agent Estimation

This repository provides a framework for **supervised learning** of multi-agent quantity estimation based on **graph-structured representations** using a Graph Attention Network (GAT). It includes:

* Data collection via environment interaction
* HDF5-formatted dataset with node-wise graph information
* Supervised training with GAT to predict the number of agents required for success

---

## 📁 Project Structure

```
.
├── collecting_data/
│   └── test_driver.py        # Script to generate and save graph data in HDF5 format
│
├── training/
│   ├── train_gat.py          # Supervised training pipeline using GAT
│   └── gat_model.py          # GAT architecture

```

---

## 📦 Dependencies

```bash
pip install torch numpy h5py tensorboard
```

---

## 📊 Data Collection

To collect graph-based training data, run the following script:

```bash
python collecting_data/test_driver.py
```

This will generate `.h5` files in the output directory (typically under `data/hdf5_data/`). Each file stores multiple episodes with graph data, structured as:

```python
{
    "global_node_features": worker.global_node_features,        # shape: [N, 2] - node coordinates
    "adjacency_matrix": worker.adjacency_matrix,                # shape: [N, N] - binary graph connectivity (1 = no edge)
    "start_point_index": worker.start_point_index,              # int - index of the start node
    "success_indicator": worker.success_indicator,              # bool - whether the episode succeeded
    "n_agents": worker.perf_metrics['n_agents']                 # float - target label (number of agents)
}
```

---

## 🧠 Supervised Training

Once the dataset is collected, you can train the GAT model by running:

```bash
python training/train_gat.py
```

### Key Arguments

| Argument                 | Description                          | Default          |
| ------------------------ | ------------------------------------ | ---------------- |
| `--train_data_directory` | Path to HDF5 training data           | `data/hdf5_data` |
| `--val_data_directory`   | Path to HDF5 validation data         | `data/hdf5_eval` |
| `--hidden_dim`           | Hidden dimension size for GAT layers | `512`            |
| `--num_heads`            | Number of attention heads            | `16`             |
| `--batch_size`           | Training batch size                  | `64`             |
| `--num_epochs`           | Number of training epochs            | `200`            |

To monitor training progress, use TensorBoard:

```bash
tensorboard --logdir logs
```

---

## 💾 Checkpoints

The best-performing model (based on validation loss) will be saved automatically to the `checkpoints/` directory.

---

## 📌 Notes

* The adjacency matrix uses **1 to represent no connection** (i.e., masked edges).
* The model predicts the number of agents needed **at the `start_point_index`** node.
* Padding is applied so all samples have a consistent node size (default: 360 nodes).

---

