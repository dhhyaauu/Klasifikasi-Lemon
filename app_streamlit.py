import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model_klasifikasi_lemon.joblib")

st.set_page_config(
	page_title="Klasifikasi Lemon",
	page_icon=":lemon:"
)

st.title(":lemon: Klasifikasi Lemon")
st.markdown("Aplikasi untuk klasifikasi Lemon bagus, sedang atau jelek")

diameter = st.slider("Diameter", 5.00, 10.0, 6.0)
berat = st.slider("Berat", 50.0, 150.0, 110.0
tebal_kulit = st.slider("Tebal Kulit", 3.0, 7.0, 3.5)
kadar_gula = st.slider("Kadar Gula", 5.0, 10.0, 6.0)
asal_daerah = st.pills("Asal Daerah", ["California", "Malang", "Medan" ="California"]
warna = st.pills("Warna", ["Hijau pekat","Kuning kehijauan","Kuning cerah" default = "Kuning kehijauan"]
musim_panen = st.pills("Musim Panen", ["Puncak","Akhir","Awal"], default="Puncak")

if st.button("Prediksi", type="primary"):
	data = pd.DataFrame([[diameter,berat,tebal_kulit,kadar_gula,asal_daerah,warna,musim_panen]], columns=["diameter","berat","tebal_kulit","kadar_gula","asal_daerah","warna","musim_panen"])
	prediksi = model.predict(data)[0]
	presentase = max(model.predict_proba(data)[0])
	st.success(f"Prediksi **{prediksi}** dengan tingkat keyakinan **{presentase*100:.2f}%**")
	st.balloons()

st.divider()
st.captions(Dibuat dengan :lemon: oleh **Saffa Dhiya**")


