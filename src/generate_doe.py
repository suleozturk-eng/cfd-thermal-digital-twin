import pandas as pd
import numpy as np

df = pd.read_csv('simscale_senaryolar.csv')
df.columns = ['Hava_Hizi_ms', 'Isi_Gucu_W']

n = 0.65
terim1 = 1000 / (2.5 ** n)
terim2 = 4500 / (0.2 ** n)
C = (32.7 - 30.0) / (terim2 - terim1)
T0 = 30.0 - C * terim1

np.random.seed(42)
gurultu = np.random.normal(0, 0.08, size=len(df))
df['Maks_Sicaklik_C'] = (T0 + C * (df['Isi_Gucu_W'] / (df['Hava_Hizi_ms'] ** n)) + gurultu).round(2)

uc_noktalar = pd.DataFrame({'Hava_Hizi_ms': [2.50, 0.20], 'Isi_Gucu_W': [1000.00, 4500.00], 'Maks_Sicaklik_C': [30.00, 32.70]})
df_final = pd.concat([df, uc_noktalar], ignore_index=True)
df_final.to_csv('simscale_kalibre_veri.csv', index=False)
print('Toplam', len(df_final), 'verilik kalibre set olusturuldu!')
