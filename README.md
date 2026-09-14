# A BLE-Based Fine-Grained Attendance System via Relational RSSI Modeling and Discrete Location Classification

This repository contains the dataset, code, and supplementary materials for the research paper on improving Bluetooth Low Energy (BLE) indoor positioning for classroom attendance systems.

## Overview

Traditional student absence recording is time-consuming, and computer vision approaches often struggle with occlusion, lighting, and camera availability in dense classrooms. This research proposes an alternative smart attendance system using BLE Received Signal Strength Indicator (RSSI) fingerprinting.

To overcome the instability of raw RSSI signals caused by multipath propagation, we introduce a novel approach utilizing **Relational RSSI Features** combined with **Temporal Smoothing**. Rather than treating localization as a continuous spatial problem, this study formulates it as a discrete classification task across multiple spatial granularities (from individual workstations to broad classroom zones).

## Key Contributions

1. **Temporal Smoothing**: Implementing a one-dimensional Kalman Filter (KF) to stabilize raw RSSI fluctuations.
2. **Relational RSSI Features**: 
   - *Beacon Ranking Feature*: Captures the relative ordering of beacon signal strengths.
   - *Pairwise Difference Feature*: Captures the exact signal margins between beacon pairs, inspired by the LANDMARC relative fingerprinting principle.
3. **Granularity Analysis**: Evaluating the system's performance across 40, 20, 8, and 4 spatial labels to understand the trade-offs between spatial resolution and classification reliability.

## Dataset

The study employs a self-collected dataset gathered in a 9.1 m × 8.5 m classroom containing 40 student workstations.
- **Infrastructure**: 4 BLE beacons (UGREEN Bluetooth 6.0) mounted above specific workstations.
- **Data Points**: 200 distinct capture points (5 positions per workstation).
- **Size**: 31,972 valid, time-consistent RSSI samples after preprocessing and staleness cutoff filtering.

## Methodology

1. **Preprocessing**: Forward-fill snapshot approach with a 3-second staleness cutoff to synchronize asynchronous BLE readings.
2. **Feature Engineering**: Applying KF, extracting Rank features, and calculating Diff features.
3. **Modeling**: Training Weighted K-Nearest Neighbors (WKNN) and Random Forest (RF) classifiers. Data is standardized using `StandardScaler`.
4. **Evaluation**: Models are evaluated on accuracy, precision, recall, and F1-score across four distinct label groupings (40, 20, 8, and 4 labels).


## Prerequisites

- Python 3.8+
- scikit-learn
- pandas
- numpy
- matplotlib / seaborn (for visualization)

## Citation

If you use this dataset or code in your research, please cite our paper:

```bibtex
@article{ble_attendance_2026,
  title={A BLE-Based Fine-Grained Attendance System via Relational RSSI Modeling and Discrete Location Classification},
  % author={[Place Holder]},
  % journal={[Place Holder]},
  year={2026}
}
```
