# cfd-thermal-digital-twin
Real-time thermal surrogate model combining 3D CFD simulations (SimScale) and SVR machine learning for data center cooling optimization.
# 🌡️ Real-Time Thermal Surrogate Model Using 3D CFD & Machine Learning

[![Python](https://shields.io)](https://python.org)
[![Scikit-Learn](https://shields.io)](https://scikit-learn.org)
[![SciPy](https://shields.io)](https://scipy.org)
[![SimScale](https://shields.io)](https://simscale.com)

## 📌 Project Overview
This project develops a **real-time thermal surrogate model** for data center and electronic cabinet cooling systems by integrating high-fidelity **3D Computational Fluid Dynamics (CFD)** simulations with **Machine Learning**. 

By replacing computationally expensive 3D CFD runs (~10 minutes per simulation) with a trained surrogate model capable of **millisecond-level inference**, this framework provides a scalable, physics-informed foundation for **Digital Twin** and real-time thermal management applications.

---

## 🛠️ Engineering & Data Science Workflow

### 1. 3D Thermal & Fluid Flow Analysis (SimScale)
* **Turbulence Modeling:** Conducted 3D steady-state forced convection simulations using the **k-ω SST** turbulence model on SimScale.
* **Domain & Validation:** Simulated data center / electronic cabinet cooling scenarios and verified critical thermal and aerodynamic boundary conditions.

### 2. Design of Experiments (DoE) & Data Calibration
* **Parameter Space:** Defined the operational envelope across air inlet velocity (**0.1 – 2.5 m/s**) and thermal heat load (**1000 – 5000 W**).
* **Sampling Strategy:** Applied **Latin Hypercube Sampling (LHS)** via `SciPy` to uniformly sample the design space, generating a calibrated, physics-based training dataset.

### 3. Machine Learning & Model Validation (Python)
* **Algorithm Benchmarking:** Trained and compared **Ridge Regression**, **Gaussian Process Regression (GPR)**, and **Support Vector Regression (SVR with RBF kernel)** using `scikit-learn`.
* **Validation Strategy:** Implemented **Leave-One-Out Cross-Validation (LOOCV)** to rigorously evaluate generalization accuracy on limited, high-cost simulation datasets.

---

## 📊 Performance & Optimization Results

| Evaluation Metric | Result / Benchmark |
| :--- | :--- |
| **Optimal Algorithm** | Support Vector Regression (**SVR - RBF Kernel**) |
| **Mean Absolute Error (MAE)** | **±0.15 °C** |
| **Coefficient of Determination (R²)** | **87.3%** |
| **3D CFD Computation Time** | ~10 minutes / simulation |
| **Surrogate Model Inference Time** | **Milliseconds** (~60,000x speedup) |

---

## 💻 Tools & Libraries
* **3D CFD Simulation:** SimScale (`k-ω SST` Forced Convection)
* **Programming Language:** Python
* **Data Manipulation & DoE:** `Pandas`, `SciPy` (Latin Hypercube Sampling)
* **Machine Learning & Metrics:** `scikit-learn` (SVR, GPR, Ridge, LOOCV)
* **Data Visualization:** `Matplotlib`

---

## 📂 Repository Structure
```text
├── data/               # Calibrated LHS simulation dataset (.csv)
├── notebooks/          # Jupyter Notebooks for LHS sampling, model benchmarking & LOOCV
├── src/                # Python scripts for training and real-time thermal inference
├── results/            # 3D CFD temperature/velocity contours & ML parity plots
└── README.md           # Project documentation
```

---

## 🚀 Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd DEPO_ADIN
   ```
2. **Install required packages:**
   ```bash
   pip install pandas scipy scikit-learn matplotlib
   ```
3. **Explore the workflow:**
   Run the scripts in `src/` or open the notebooks in `notebooks/` to inspect the LHS dataset, LOOCV metrics, and real-time temperature predictions.
