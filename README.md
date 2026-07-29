# WiFi CSI-Based People Detection and Counting

Master's thesis project — **Electrical and Microsystems Engineering, OTH Regensburg**

A device-free occupancy sensing system built from two ESP32-S3 microcontrollers (≈ €20 total)
and a Raspberry Pi 5. The system extracts 802.11n Channel State Information (CSI) from
commodity WiFi hardware and infers room occupancy without cameras, wearables or user interaction.

---

## System overview

```
ESP32-S3 (TX)  --ESP-NOW packets, 2.4 GHz ch.11, 30 Hz-->  ESP32-S3 (RX)
                                                                 |
                                                    192 subcarrier CSI over UART
                                                                 v
                                                        Raspberry Pi 5
                                          windowing -> features -> Random Forest
                                                                 v
                                                     occupancy estimate (0-3)
```

Full pipeline — acquisition, feature extraction, training and real-time inference —
runs on the Raspberry Pi with a **0.53 % CPU duty cycle**.

## Hardware

| Component | Role |
|---|---|
| 2 × ESP32-S3-DevKitM-1 | CSI transmitter and receiver |
| Raspberry Pi 5 | Data logging, training, real-time inference |
| Windows laptop | Firmware development, offline analysis |

## Software

ESP-IDF v6.0.1 · Python 3.13 · NumPy · pandas · scikit-learn · Matplotlib · pySerial

The ESP-CSI reference firmware was ported from ESP-IDF v5 to v6
(`WIFI_BW_HT40` → `WIFI_BW40`, `WIFI_BW_HT20` → `WIFI_BW20`).

---

## Dataset

| Environment | Description | Sessions |
|---|---|---|
| Bedroom | 16–19 m², furnished, laminate | 42 |
| Kitchen | 15–25 m², appliances, laminate | 30 |
| Classroom | 50–80 m², ~60 seats, laminate | 26 |

**~120 sessions · 4 occupancy classes (0–3 persons) · 4 subjects · multiple days ·
~200 000 CSI packets.** Four representative recordings are included in `data_sample/`;
the full dataset is available on request.

---

## Key results

Evaluation uses **session-grouped cross-validation** (no window from a recording appears in
both training and test sets) and **imbalance-robust metrics** with explicit majority baselines.

### Classification performance

| Task | Baseline | Best model | Balanced accuracy | ROC-AUC |
|---|---|---|---|---|
| Presence (0 vs 1+) | 50.0 % | SVM (RBF) | **86.2 % ± 4.4** | **96.2 % ± 2.6** |
| Occupancy level (0/1/2+) | 33.3 % | Random Forest | **67.1 % ± 9.3** | — |
| Exact count (0–3) | 25.0 % | Random Forest | **52.4 % ± 6.8** | — |
| Count within ±1 person | — | Random Forest | **84.3 %** | — |

All results significant against baseline (p = 0.0001 – 0.011).

### Generalisation

| Study | Result |
|---|---|
| Unseen **person** (leave-one-subject-out, 4 subjects) | 80.9 % ± 1.1 — only **1.4 pp** below same-subject reference |
| Unseen **environment** (leave-one-room-out) | 69.0 – 96.0 %, i.e. **15–20 pp** degradation |
| Multi-environment training | 84.1 % ± 7.6 (best overall stability) |
| Few-shot adaptation | 2 min of calibration data in a new room: AUC 69.8 % → 76.8 % |

**The system is person-independent but environment-dependent.**

### Sensor and configuration analysis

- **RSSI performs comparably to CSI** on this single-link setup (89.2 % vs 88.0 % presence);
  CSI's advantage appears only for exact counting (+3.7 pp).
- **8 subcarriers ≈ 166 subcarriers** and **5 Hz ≈ 30 Hz** for presence detection —
  a ~125× reduction in data rate with no measurable accuracy loss.
- Occupancy information for a single TX–RX link therefore lies predominantly in
  **temporal power dynamics**, not per-subcarrier frequency diversity.

### Embedded performance (Raspberry Pi 5)

| Metric | Value |
|---|---|
| Feature extraction | 1.95 ms |
| Model inference | 8.61 ms |
| Total per window | 10.6 ms |
| CPU duty cycle (2 s interval) | 0.53 % |
| Throughput | 95 windows/s (190× real-time) |
| Peak memory | 129 MB |

---

## Repository structure

```
code/          Python pipeline: logging, features, ML, evaluation, figures
firmware/      Ported ESP-CSI application code (TX and RX)
figures/       Thesis figures (PNG + PDF, 300 dpi)
docs/          Full result logs and data-collection campaign log
data_sample/   Example recordings, one per occupancy class
```

## Reproducing the analysis

```bash
pip install numpy pandas matplotlib scikit-learn scipy seaborn pyserial joblib

python code/csi_features.py          # raw CSV -> feature matrix
python code/csi_ml.py                # main classification results
python code/csi_stats_balanced.py    # imbalance-robust metrics + significance
python code/csi_crossroom.py         # cross-environment transfer
python code/csi_loso.py              # leave-one-subject-out
python code/csi_fewshot.py           # few-shot adaptation
python code/csi_rssi_compare.py      # CSI vs RSSI comparison
python code/csi_sensitivity.py       # parameter sensitivity study
```

Data acquisition (requires hardware):

```bash
python code/csi_logger.py --port COM8 --baud 921600 --label 0 --duration 60 --tag room1 --note "empty"
```

---

## Author

Venkata Sasi Kiran Reddy Kotapati
M.Sc. Electrical and Microsystems Engineering, OTH Regensburg, 2026

## Acknowledgements

Built on [Espressif ESP-CSI](https://github.com/espressif/esp-csi).