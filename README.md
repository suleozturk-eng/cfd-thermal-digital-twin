# Real-Time Thermal Surrogate Model Using 3D CFD & Machine Learning

Real-time thermal surrogate model combining 3D CFD simulations (SimScale) and SVR machine learning for data center cooling optimization.

## Quick Start
```bash
git clone https://github.com
cd cfd-thermal-digital-twin
pip install -r requirements.txt
python src/train.py
python src/predict.py --velocity 1.5 --power 3000
python -m streamlit run app.py
```
