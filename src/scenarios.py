import pandas as pd
from scipy.stats import qmc

# Örnek: 3 değişken için (Giriş Hızı, Güç, Giriş Sıcaklığı)
# Alt ve üst sınırları kendi sisteminize göre güncelleyin
alt_sinirlar = [0.5, 500, 18.0]   # [Min Hız (m/s), Min Güç (W), Min Sıcaklık (°C)]
ust_sinirlar = [5.0, 3000, 30.0]  # [Max Hız (m/s), Max Güç (W), Max Sıcaklık (°C)]

sampler = qmc.LatinHypercube(d=len(alt_sinirlar), seed=42)
ornekler = sampler.random(n=30)
olcekli_veri = qmc.scale(ornekler, alt_sinirlar, ust_sinirlar)

df = pd.DataFrame(olcekli_veri, columns=["Giris_Hizi", "Guc_W", "Giris_Sicakligi"])
df = df.round(2) # Sayıları okumayı kolaylaştırmak için yuvarlama
df.to_csv("simscale_senaryolar.csv", index=False)
print(df.head())
