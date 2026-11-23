# Real-Time Thermal Surrogate Model Using 3D CFD & Machine Learning

Real-time thermal surrogate model combining 3D CFD simulations and SVR machine learning for data center thermal optimization.

## Surrogate Model Validation & Thermal Map
![Surrogate Model Results](results/surrogate_model_sonuc.png)

## Performance Metrics
- **Algorithm:** SVR (RBF Kernel)
- **MAE:** +- 0.1537 C
- **R2 Score:** 87.3%
- **Speedup:** ~60,000x (~10 min CFD to ms)

## Quick Start
```bash
git clone https://github.comsuleozturk-eng/cfd-thermal-digital-twin.gitsuleozturk-eng/cfd-thermal-digital-twin.git
cd cfd-thermal-digital-twin
pip install -r requirements.txt
python src/train.py
python src/predict.py --velocity 1.5 --power 3000
```
