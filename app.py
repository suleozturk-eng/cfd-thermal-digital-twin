import time, joblib, numpy as np, pandas as pd, streamlit as st, matplotlib.pyplot as plt
st.set_page_config(page_title="CFD Thermal Digital Twin", page_icon="🌡️", layout="wide")
st.title("🌡️ Real-Time Thermal Digital Twin (3D CFD + SVR)")
model = joblib.load("models/svr_thermal_surrogate.pkl")
col1, col2 = st.columns([1, 1.5])
with col1:
    velocity = st.slider("Air Inlet Velocity (m/s)", 0.1, 2.5, 1.5, 0.05)
    power = st.slider("Thermal Heat Load (W)", 1000, 5000, 3000, 50)
    df_in = pd.DataFrame([[velocity, power]], columns=["Hava_Hizi_ms", "Isi_Gucu_W"])
    t0 = time.perf_counter()
    pred_temp = model.predict(df_in)[0]
    inf_ms = (time.perf_counter() - t0) * 1000
    st.metric("Predicted Max Temperature", f"{pred_temp:.2f} °C")
    st.metric("Inference Runtime", f"{inf_ms:.2f} ms", "~60,000x faster than CFD")
    if pred_temp > 45.0: st.error("⚠️ Critical Thermal Alert ( °C)")
    else: st.success("✅ Safe Operating Regime")
with col2:
    hiz_grid = np.linspace(0.1, 2.5, 60)
    guc_grid = np.linspace(1000, 5000, 60)
    H, G = np.meshgrid(hiz_grid, guc_grid)
    Z = model.predict(pd.DataFrame({"Hava_Hizi_ms": H.ravel(), "Isi_Gucu_W": G.ravel()})).reshape(H.shape)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    kontur = ax.contourf(H, G, Z, levels=20, cmap="coolwarm")
    plt.colorbar(kontur, ax=ax, label="Max Temp (°C)")
    ax.scatter([velocity], [power], color="lime", edgecolors="black", s=180, marker="*", zorder=5, label="Operating Point")
    ax.set_xlabel("Inlet Velocity (m/s)")
    ax.set_ylabel("Heat Load (W)")
    ax.legend()
    st.pyplot(fig)
    if st.button("💾 Save Current Map to results/dashboard_demo.png"):
        fig.savefig("results/dashboard_demo.png", dpi=300, bbox_inches="tight")
        st.toast("Saved to results/dashboard_demo.png!")
