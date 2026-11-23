import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.linear_model import Ridge

df = pd.read_csv('data/simscale_kalibre_veri.csv')
X = df[['Hava_Hizi_ms', 'Isi_Gucu_W']]
y = df['Maks_Sicaklik_C']

modeller = {
    'Ridge Regresyon': make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    'SVR (RBF)': make_pipeline(StandardScaler(), SVR(kernel='rbf', C=10, gamma='scale')),
    'Gaussian Process': make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=C(1.0) * RBF(1.0), n_restarts_optimizer=10))
}

loo = LeaveOneOut()
en_iyi_isim = ''
en_dusuk_hata = float('inf')

print('--- MODEL PERFORMANSLARI (Ortalama Mutlak Hata - MAE) ---')
for isim, model in modeller.items():
    skorlar = cross_val_score(model, X, y, cv=loo, scoring='neg_mean_absolute_error')
    mae = -np.mean(skorlar)
    print(f'{isim}: {mae:.4f} °C hata')
    if mae < en_dusuk_hata:
        en_dusuk_hata = mae
        en_iyi_isim = isim

print(f'\nSecilen En Iyi Model: {en_iyi_isim}')

secilen_model = modeller[en_iyi_isim]
tahminler = cross_val_predict(secilen_model, X, y, cv=loo)
secilen_model.fit(X, y)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

ax1.scatter(y, tahminler, color='royalblue', edgecolors='black', label='Veri Noktalari')
ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Ideal Dogru (y=x)')
ax1.set_xlabel('CFD Kalibre Sicaklik (°C)')
ax1.set_ylabel('ML Tahmini (°C)')
ax1.set_title(f'Model Dogrulama ({en_iyi_isim} - MAE: {en_dusuk_hata:.3f}°C)')
ax1.legend()
ax1.grid(True)

hiz_grid = np.linspace(0.1, 2.5, 100)
guc_grid = np.linspace(1000, 5000, 100)
H, G = np.meshgrid(hiz_grid, guc_grid)
izgara = pd.DataFrame({'Hava_Hizi_ms': H.ravel(), 'Isi_Gucu_W': G.ravel()})
Z = secilen_model.predict(izgara).reshape(H.shape)

kontur = ax2.contourf(H, G, Z, levels=20, cmap='coolwarm')
plt.colorbar(kontur, ax=ax2, label='Tahmin Edilen Maks. Sicaklik (°C)')
ax2.scatter(X['Hava_Hizi_ms'], X['Isi_Gucu_W'], c='black', s=20, label='LHS Ornekleri')
ax2.scatter([2.5, 0.2], [1000, 4500], c='yellow', edgecolors='black', s=80, marker='*', label='Gercek CFD Benchmark')
ax2.set_xlabel('Hava Giris Hizi (m/s)')
ax2.set_ylabel('Isi Gucu (W)')
ax2.set_title('Termal Surrogate Model Tahmin Haritasi')
ax2.legend()

plt.tight_layout()
import joblib; os.makedirs('models', exist_ok=True); joblib.dump(secilen_model, 'models/svr_thermal_surrogate.pkl'); os.makedirs('results', exist_ok=True); plt.savefig('results/surrogate_model_sonuc.png', dpi=300)
plt.show()
