import streamlit as st
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
st.set_page_config(
    page_title="Segmentasi Pelanggan UMKM",
    page_icon="🛍️",
    layout="wide"
)
# Cari folder utama proyek berdasarkan lokasi file ini
ROOT_DIR = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT_DIR / "artifacts"


@st.cache_resource
def muat_model():
    model = joblib.load(ARTIFACT_DIR / "model.joblib")

    with open(
        ARTIFACT_DIR / "metadata.json",
        "r",
        encoding="utf-8"
    ) as file:
        metadata = json.load(file)

    return model, metadata


model, metadata = muat_model()
st.sidebar.title("Dashboard UMKM")
st.sidebar.caption("Bantu susun promosi berdasarkan perilaku pelanggan.")

menu = st.sidebar.radio(
    "Pilih halaman",
    ["Segmentasi Pelanggan", "Profil Segmen"]
)
if menu == "Segmentasi Pelanggan":
    st.title("Segmentasi Pelanggan UMKM")
    st.write(
        "Kenali kelompok pelanggan berdasarkan perilaku belanja "
        "untuk membantu menentukan strategi pemasaran."
    )

    st.subheader("Data Perilaku Pelanggan")
    st.caption("Gunakan data transaksi untuk periode enam bulan yang sama.")

    with st.form("form_pelanggan"):
        kiri, kanan = st.columns(2)

        with kiri:
            jumlah_transaksi = st.number_input(
                "Jumlah transaksi dalam 6 bulan",
                min_value=1,
                value=10,
                step=1
            )

            total_belanja = st.number_input(
                "Total belanja dalam 6 bulan (Rp)",
                min_value=0,
                value=500000,
                step=10000
            )

            hari_terakhir = st.number_input(
                "Hari sejak transaksi terakhir",
                min_value=0,
                value=15,
                step=1
            )

        with kanan:
            persen_promo = st.slider(
                "Persentase transaksi menggunakan promo",
                min_value=0,
                max_value=100,
                value=30
            )

            rating = st.number_input(
                "Rating rata-rata",
                min_value=1.0,
                max_value=5.0,
                value=4.0,
                step=0.1
            )

            rating_kosong = st.checkbox(
                "Rating belum tersedia",
                help="Jika dicentang, model akan mengisi rating dengan median data latih."
            )

        tombol_prediksi = st.form_submit_button("Tentukan Segmen")

    if tombol_prediksi:
        data_input = pd.DataFrame([{
            "jumlah_transaksi_6bln": jumlah_transaksi,
            "total_belanja_rp": total_belanja,
            "hari_sejak_transaksi_terakhir": hari_terakhir,
            "persen_pakai_promo": persen_promo,
            "rating_rata": np.nan if rating_kosong else rating
        }])

        # Pastikan fitur sesuai dengan model
        data_input = data_input[metadata["fitur"]]

        # Pipeline otomatis melakukan preprocessing dan prediksi
        cluster = int(model.predict(data_input)[0])

        nama_segmen = metadata["nama_segmen"][str(cluster)]
        rekomendasi = metadata["rekomendasi_segmen"][str(cluster)]

        st.subheader("Hasil Segmentasi")
        st.success(f"Segmen pelanggan: {nama_segmen}")

        st.write("**Usulan tindakan pemasaran:**")
        st.write(rekomendasi)

        if rating_kosong:
            st.caption(
                "Rating kosong diisi otomatis menggunakan "
                "median rating dari data latih."
            )
elif menu == "Profil Segmen":
    st.divider()
    st.subheader("Profil Kelompok Pelanggan")
    st.caption(
        "Ringkasan dari 8.000 pelanggan data pengembangan. "
        "Nilai perilaku merupakan median setiap kelompok."
    )

    profil = pd.read_csv(ARTIFACT_DIR / "profil_cluster.csv")

    # Hubungkan nomor cluster dengan nama segmennya
    profil["Segmen"] = profil["Cluster"].astype(str).map(
        metadata["nama_segmen"]
    )

    kolom_tampilan = {
        "Segmen": "Segmen",
        "Jumlah Pelanggan": "Jumlah pelanggan",
        "Persentase (%)": "Proporsi (%)",
        "jumlah_transaksi_6bln": "Median transaksi",
        "total_belanja_rp": "Median belanja (Rp)",
        "hari_sejak_transaksi_terakhir": "Median hari sejak transaksi",
        "persen_pakai_promo": "Median penggunaan promo (%)",
        "rating_rata": "Median rating"
    }

    tabel_profil = profil[list(kolom_tampilan)].rename(
        columns=kolom_tampilan
    )

    st.dataframe(
        tabel_profil,
        hide_index=True
    )

    st.caption(
        "Median rating dihitung dari rating yang tersedia. "
        "Profil ini menjadi referensi, bukan batas mutlak penentuan segmen."
    )
    st.subheader("Usulan Pemasaran per Segmen")

    for cluster_id, nama in metadata["nama_segmen"].items():
        st.write(f"**{nama}**")
        st.write(metadata["rekomendasi_segmen"][cluster_id])

    st.caption(
        "Usulan ini perlu diuji dengan mempertimbangkan biaya promo "
        "dan laba setelah diskon."
    )