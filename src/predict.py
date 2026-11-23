import time, argparse, joblib, pandas as pd
parser = argparse.ArgumentParser()
parser.add_argument("--velocity", type=float, required=True)
parser.add_argument("--power", type=float, required=True)
args = parser.parse_args()
model = joblib.load("models/svr_thermal_surrogate.pkl")
df_in = pd.DataFrame([[args.velocity, args.power]], columns=["Hava_Hizi_ms", "Isi_Gucu_W"])
t0 = time.perf_counter()
pred = model.predict(df_in)
ms = (time.perf_counter() - t0) * 1000
print(f"Predicted Max Temp: {pred[0]:.2f} C - Inference Time: {ms:.3f} ms")
