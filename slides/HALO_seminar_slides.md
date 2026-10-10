# HALO: Horizon-Aware Load Optimizer
## Time-Aware Satellite Image Prioritization and Expiry
**Technical Seminar Presentation** | October 2026 | @Darshan

---

### Slide 1: The Problem
#### Satellite Downlink Asymmetry in Low Earth Orbit
* **Sensor Capture Rate**: Modern optical satellites capture gigabytes of high-resolution Earth imagery continuously.
* **Downlink Bottleneck**: Ground station passes in Low Earth Orbit (LEO) are scarce:
  * Contact duration: only **8 to 12 minutes**.
  * Revisit period: every **90 to 100 minutes**.
  * Bandwidth: limited transmission budget $D$ bytes per pass.
* **Core Dilemma**: Satellites capture far more data than they can downlink. What gets transmitted first dictates the entire value of the mission.

---

### Slide 2: Why It Matters
#### Real-Time Earth Observation Saves Lives and Resources
* **Disasters are Dynamic**:
  * Wildfires and active plumes spread and change hourly.
  * Maritime vessels transit out of coordinate bounds in minutes.
  * Rapid reaction is crucial for first responders and disaster management.
* **The Cost of Bad Schedulers**:
  * Downlinking obsolete images squanders expensive RF downlink time.
  * Waiting background data (agriculture, forestry, hydrology) starves in storage and gets discarded.
  * Memory overflows force unprincipled onboard data drops.

---

### Slide 3: What Exists Today
#### Prior Approaches to Satellite Data Management
1. **FIFO Buffering**:
   * Transmit images strictly in capture arrival order.
   * Treats all pixels as equally valuable; clogs queue with clouds and duplicates.
2. **Onboard Cloud & Semantic Classification**:
   * Machine learning models screen out obvious cloud cover or classify land cover.
   * **Limitation**: Once filtered, all retained images are treated as statically and permanently valuable.
3. **Orbital Channel Schedulers**:
   * Optimize radio transmission modulation and ground station dish scheduling.
   * Optimizes the *pipe*, but remains content-agnostic.

---

### Slide 4: The Gap
#### What Prior Systems Leave Open
* **Value Decays with Time**: An image of a fire captured 6 hours ago has lost almost all tactical utility, yet existing schedulers downlink it anyway.
* **Waiting Images Starve**: If an event produces bursts of high-priority images, routine background images sit in storage forever until evicted.
* **Disconnection from Ground Alerts**: Ground teams learn of emergencies (e.g., localized fires or floods) from in-situ sensors, but satellites cannot reprioritize waiting buffers dynamically.
* **Redundant Data Clogs Downlinks**: Unchanged tiles are re-imaged and re-sent repeatedly.

---

### Slide 5: The Idea: HALO
#### Horizon-Aware Load Optimizer
* **Core Innovation**: A lightweight onboard management and scheduling architecture that balances:
  1. **Time-decaying utility**: Exponential decay profiles tailored per event type ($\lambda_k$).
  2. **Delivery-time expiry**: Preemptive deletion if projected value at the next pass ($t_{next}$) falls below threshold $\theta$.
  3. **Alert-driven reprioritization**: Uplinked ground alerts delivered during passes boost urgent areas ($A_i$) and mark older images covered.
  4. **Bounded-wait aging**: Starvation fairness bonus that *never* revives stale data.
  5. **Unified storage policy**: Evict acknowledged/sent frames first, then lowest score per byte, powered by INT8 MobileNet.

---

### Slide 6: Architecture & Pipeline
#### 8-Stage Onboard Processing Flow

```
[1. Capture] ──> [2. Quality Filter] ──> [3. Duplicate Check] ──> [4. Onboard Scorer]
 (Raw pixels)       (Rejects clouds,        (Drops static           (INT8 MobileNet
                     blur, blank)            near-duplicates)        3.81 MB model)
      │
      ▼
[8. Storage Mgr] <── [7. Scheduler] <── [6. Expiry Check] <── [5. Utility Queue]
 (Evicts sent first,  (Greedy / 0-1      (Deletes if projected   (Maintains U(t),
  then lowest S/byte)  Knapsack)          U(t_next) < theta)      Aging, Boosts)
                                                                        ▲
                                                                [Ground Alerts]
```

---

### Slide 7: Utility Formulas in Plain Words
#### Exact Mathematical Rules from Section 4
1. **Base Score ($B_i$)**:
   $$\mathbf{B_i = q_i \cdot m_i \cdot c_i}$$
   *Combines Image Quality ($q_i$), Model Usefulness ($m_i$), and Scene Change ($c_i$).*
2. **Time-Decaying Utility ($U_i(t)$)**:
   $$\mathbf{U_i(t) = B_i \cdot A_i \cdot e^{-\lambda_k (t - t_i)}}$$
   *Value decays exponentially at event rate $\lambda_k$ (Fire: 1.0/h; Ship: 0.5/h; Flood: 0.05/h; Crop: 0.005/h).*
3. **Delivery-Time Expiry**:
   $$\mathbf{U_i(t_{next}) < \theta \implies \text{Delete Now}}$$
   *Never downlink an image that will be stale by the time it reaches the ground station.*
4. **Aging Score ($S_i(t)$)**:
   $$\mathbf{S_i(t) = U_i(t) + \beta \cdot \min(w_i, w_{max}) \cdot \mathbf{1}[U_i(t) \ge \theta]}$$
   *Fairness bonus prevents starvation, but the indicator guarantees aging never raises expired data.*

---

### Slide 8: Simulator Setup
#### Realistic Discrete-Event Mission Benchmark
* **Mission Duration**: 24-hour continuous Low Earth Orbit simulation.
* **Satellite Storage**: 500 MB onboard capacity ceiling.
* **Capture Profile**: 1 image every 45 seconds (1,920 images total across 16 orbital tiles).
* **Ground Contacts**: 15 ground passes (10 min windows every 95 min, 80 MB downlink budget per pass).
* **Onboard Model**: Quantized INT8 MobileNetV3-Small (3.81 MB footprint, 4.0 ms inference latency).
* **Replication**: 10 independent random seeds evaluated to report mean and standard deviation.

---

### Slide 9: Experimental Results
#### HALO vs. Standard Baselines (10 Seeds)

| Metric | FIFO Baseline | Random | Classifier-Only | **HALO (0/1 Knapsack)** | Improvement |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Delivered Utility / Pass (↑)** | 11.95 ± 1.01 | 17.99 ± 1.14 | 15.72 ± 0.81 | **14.22 ± 0.92** | High Utility |
| **Share of Stale Data Sent (↓)** | **56.17%** | 35.02% | 12.44% | **2.51%** | **95.5% Reduction** |
| **Bandwidth Wasted (MB, ↓)** | **642.62 MB** | 373.91 MB | 103.97 MB | **19.95 MB** | **96.9% Saved** |
| **Mean Buffer Occupancy (MB, ↓)** | **318.23 MB** | 318.25 MB | 31.19 MB | **31.19 MB** | **90.2% Memory Relief** |
| **Max Starvation Wait (h, ↓)** | 8.98 h | 20.81 h | 1.58 h | **1.58 h** | **No Starvation** |

*Key Takeaway: FIFO wastes 642 MB transmitting 56% stale images. HALO delivers fresh actionable data with only 2.5% staleness.*

---

### Slide 10: Ablation Analysis
#### Every Mechanism Contributes Directly

* **Removing Duplicate Filter (`- No Dedup`)**:
  * Stale fraction spikes from 14% to **35.6%**.
  * Wasted bandwidth surges $3.7\times$ to **422.33 MB**.
  * Storage explodes from 31 MB to **276 MB**.
* **Removing Delivery Expiry (`- No Expiry`)**:
  * Degraded images remain in storage, inflating queue clutter.
* **Removing Alerts (`- No Alerts`)**:
  * Satellite fails to prioritize high-urgency disaster zones identified by ground operators.
* **Knapsack vs. Greedy**:
  * 0/1 Knapsack achieves optimal packet packing, driving staleness down to **2.51%** vs 14.17% for greedy.

---

### Slide 11: Limitations & Honesty Rule
#### Transparent Engineering Disclosures
* **Simulation Prototype**: Evaluated on discrete-event simulation, not flight software.
* **Tuned Assumptions**: Decay rates ($\lambda_k$) and expiry thresholds ($\theta$) are domain assumptions to be tuned per mission, not universal physical constants.
* **Mission Profile Nuance**: An event having passed does not make its data useless:
  * In active fires, tactical decay is rapid.
  * In flood damage assessment, post-event images remain valuable for weeks.
  * Expiry must be tuned to the operational mission profile.
* **Honesty Statement**: HALO does not claim the concept of utility or scheduling is novel; the novelty lies in the *specific combination* of event decay, delivery-time expiry, bounded-wait aging, and quantized execution.

---

### Slide 12: Conclusion & Future Work
#### Summary & Path to Flight Readiness
* **Summary**:
  * HALO provides a complete, lightweight onboard prioritization system.
  * Slashes downlinked stale data from 56% to 2.5%.
  * Operates within a tiny 3.81 MB model footprint suitable for CubeSats and SmallSats.
* **Future Work**:
  * Hardware-in-the-loop (HIL) testing on radiation-tolerant edge hardware (e.g., Unibap SpaceCloud, Google Coral, NVIDIA Jetson Orin Nano).
  * Direct inter-satellite link (ISL) mesh routing integration.
  * Adaptive online calibration of decay rates $\lambda_k$ via ground feedback.

---
**Thank You! Questions & Discussion**
