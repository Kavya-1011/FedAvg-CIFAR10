# Federated Averaging (FedAvg) on CIFAR-10

A from-scratch PyTorch implementation of the Federated Averaging algorithm 
(McMahan et al., *Communication-Efficient Learning of Deep Networks from 
Decentralized Data*), benchmarked against centralized training to quantify 
the accuracy trade-off of decentralized learning.

## What this implements
- CNN image classifier trained via FedAvg across 10 simulated clients
- Client-side local training (`client_update`) and server-side weight 
  aggregation (`server_aggregate`)
- A federated training loop: per round, 5 of 10 clients are randomly 
  selected, train locally for 2 epochs (SGD, lr=0.01), and their weights 
  are averaged into the global model
- A centralized baseline: the same CNN architecture trained on the full, 
  unpartitioned dataset for direct comparison

## Data partitioning
Client data is split **IID**: the full CIFAR-10 training set is shuffled, 
then divided into 10 equal-sized shards, so each client sees a 
representative, class-balanced sample. No label-based skew or 
heterogeneity is introduced.

## Results

| Setup | Accuracy (start → end) |
|---|---|
| Federated (FedAvg, 20 rounds) | 33.43% → 59.2% |
| Centralized (20 epochs, full dataset) | 46.26% → 70.73% |

**Gap: ~11.5 percentage points.**

Since the client split is IID, this gap isn't caused by data heterogeneity 
across clients — it reflects the inherent cost of federated training 
dynamics: fewer effective gradient updates per round compared to 
continuous centralized training, and drift introduced by periodically 
averaging independently-trained client weights.

## Limitations & next steps
This experiment doesn't isolate how much *additional* gap would appear 
under non-IID conditions, which is the more realistic federated learning 
setting. A natural extension is to repartition client data by label 
(or via a Dirichlet allocation) and re-run the same simulation to measure 
that effect directly.

## Stack
PyTorch, NumPy, torchvision (CIFAR-10)
