import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model_klasifikasi_lemon.joblib")

st.set_page_config(
    page_title="Klasifikasi Lemon",
    page_icon="🍋"
)

st.title("🍋 Klasifikasi Lemon")
st.markdown("Aplikasi untuk mengklasifikasikan lemon menjadi kategori **Bagus**, **Sedang**, atau **Jelek**.")

diameter = st.slider("Diameter", 40.0, 75.0, 50.0)
berat = st.slider("Berat", 60.0, 150.0, 100.0)
tebal_kulit = st.slider("Tebal Kulit", 3.0, 7.0, 3.5)
kadar_gula = st.slider("Kadar Gula", 5.0, 10.0, 6.0)

asal_daerah = st.selectbox("Asal Daerah", ["California", "Malang", "Medan"], index=0)
warna = st.selectbox("Warna", ["Hijau pekat", "Kuning kehijauan", "Kuning cerah"], index=1)
musim_panen = st.selectbox("Musim Panen", ["Puncak", "Akhir", "Awal"], index=0)

if st.button("Prediksi", type="primary"):
  
    data = pd.DataFrame([[diameter, berat, tebal_kulit, kadar_gula, asal_daerah, warna, musim_panen]],
                        columns=["diameter", "berat", "tebal_kulit", "kadar_gula", "asal_daerah", "warna", "musim_panen"])
    
    prediksi = model.predict(data)[0]
    presentase = max(model.predict_proba(data)[0])
    
    st.success(f"Hasil prediksi: **{prediksi}** dengan tingkat keyakinan **{presentase*100:.2f}%**")
    st.balloons()

st.divider()
st.caption("Dibuat dengan 🍋 oleh **Saffa Dhiya**")
