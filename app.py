from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model_harga_rumah.joblib"

st.set_page_config(
    page_title="Prediksi Harga Rumah",
    page_icon="🏠",
    layout="centered",
)

st.markdown(
    """
    <style>
    /* Keep the original navy and cyan colors, with a quieter background pattern. */
    .stApp {
        background-color: #0f172a;
        background-image:
            radial-gradient(circle at 1px 1px, rgba(103, 232, 249, 0.10) 1px, transparent 1.4px),
            radial-gradient(ellipse at 12% 8%, rgba(34, 211, 238, 0.10), transparent 30%),
            radial-gradient(ellipse at 92% 22%, rgba(59, 130, 246, 0.08), transparent 28%),
            radial-gradient(ellipse at 68% 92%, rgba(14, 165, 233, 0.06), transparent 30%);
        background-size: 30px 30px, auto, auto, auto;
        background-attachment: fixed;
        color: #e2e8f0;
    }
    [data-testid="stHeader"] {
        background: rgba(15, 23, 42, 0.94);
        height: 3.25rem;
        border-bottom: 1px solid rgba(103, 232, 249, 0.07);
    }
    [data-testid="stAppViewContainer"] { background: transparent; }
    /* Keep the first section clear of Streamlit's fixed top header. */
    .block-container {
        padding-top: 3.5rem !important;
        padding-bottom: 3rem;
        max-width: 920px;
    }
    h1, h2, h3, p, label { color: #e2e8f0; }
    .eyebrow {
        display: inline-block;
        color: #67e8f9;
        font-size: 0.73rem;
        font-weight: 700;
        letter-spacing: 0.16em;
        padding: 0.38rem 0.68rem;
        margin-bottom: 0.55rem;
        border: 1px solid rgba(103, 232, 249, 0.24);
        border-radius: 999px;
        background: rgba(8, 145, 178, 0.10);
    }
    .main-title { color: #e2e8f0; margin-bottom: 0; letter-spacing: -0.035em; }
    .subtext { color: #94a3b8; margin-top: 0.2rem; }
    [data-testid="stForm"] {
        background: rgba(15, 23, 42, 0.62);
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 16px;
        padding: 1.05rem 1rem 0.9rem;
        box-shadow: 0 16px 40px rgba(2, 6, 23, 0.12);
    }
    .price-card {
        margin-top: 1rem;
        padding: 1.35rem 1.45rem 1.3rem;
        border: 1px solid rgba(34, 211, 238, 0.52);
        border-radius: 16px;
        background: linear-gradient(135deg, rgba(8, 145, 178, 0.13), rgba(30, 41, 59, 0.84) 58%, rgba(15, 23, 42, 0.94));
        box-shadow: 0 14px 34px rgba(2, 6, 23, 0.18);
    }
    .price-label {
        color: #a5f3fc;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .price-value {
        color: #f8fafc;
        font-size: clamp(1.8rem, 4vw, 2.45rem);
        font-weight: 750;
        letter-spacing: -0.035em;
        line-height: 1.2;
    }
    .price-note { color: #94a3b8; font-size: 0.85rem; margin-top: 0.6rem; }
    [data-testid="stFormSubmitButton"] button {
        background: #0891b2;
        color: white;
        border: 1px solid rgba(103, 232, 249, 0.22);
        border-radius: 10px;
        min-height: 2.75rem;
        transition: background 160ms ease, transform 160ms ease;
    }
    [data-testid="stFormSubmitButton"] button:hover {
        background: #06b6d4;
        color: #0f172a;
        border-color: #67e8f9;
        transform: translateY(-1px);
    }
    div.stButton > button { background: #0891b2; color: white; border: 0; }
    div.stButton > button:hover { background: #06b6d4; color: #0f172a; }
    [data-testid="stMetric"] { background: rgba(15, 23, 42, 0.45); border-radius: 12px; padding: 0.6rem; }
    @media (max-width: 640px) {
        .block-container { padding-top: 3.1rem !important; padding-left: 1rem; padding-right: 1rem; }
        [data-testid="stForm"] { padding: 0.85rem 0.75rem; }
        .price-card { padding: 1.1rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "model_harga_rumah.joblib belum ada. Jalankan cell deployment terakhir "
            "pada notebook v8_another_tuned, lalu letakkan artefak di folder yang sama dengan app.py."
        )

    paket = joblib.load(MODEL_PATH)
    required_keys = {"model", "kolom", "kecamatan", "mae_test_rupiah"}
    missing_keys = required_keys - set(paket.keys())
    if missing_keys:
        raise ValueError(f"Format artefak model tidak sesuai. Key yang hilang: {sorted(missing_keys)}")

    # Fitur ini harus sesuai dengan hasil feature selection pada notebook another_tuned.
    expected_features = {
        "log_LT", "LB", "log_LB", "LT", "total_kamar", "KM", "Kecamatan", "KT"
    }
    if not set(paket["kolom"]).issubset(expected_features):
        raise ValueError(
            "Artefak model tampaknya berasal dari versi notebook yang berbeda. "
            "Buat ulang model_harga_rumah.joblib dari notebook v8_another_tuned yang dipakai."
        )
    return paket


def buat_fitur(x):
    """Samakan feature engineering dengan notebook v8_another_tuned."""
    x = x.copy()
    for kolom in ["KT", "KM", "Garasi", "LT", "LB"]:
        if kolom not in x.columns:
            x[kolom] = np.nan
        x[kolom] = pd.to_numeric(x[kolom], errors="coerce")

    x["Garasi_hilang"] = x["Garasi"].isna().astype(int)
    x["LB_hilang"] = x["LB"].isna().astype(int)
    x["LT_hilang"] = x["LT"].isna().astype(int)
    x["total_kamar"] = x["KT"] + x["KM"]
    x["rasio_LB_LT"] = x["LB"] / x["LT"].replace(0, np.nan)
    x["log_LT"] = np.log1p(x["LT"].clip(lower=0))
    x["log_LB"] = np.log1p(x["LB"].clip(lower=0))
    return x


def rupiah(nilai):
    return "Rp " + f"{int(round(nilai)):,}".replace(",", ".")


def nilai_fitur(nilai, unknown, batas_bawah=None, batas_atas=None):
    if unknown:
        return np.nan, None
    if batas_bawah is not None and nilai < batas_bawah:
        return np.nan, f"Nilai {nilai:g} berada di bawah batas minimum {batas_bawah}; diperlakukan sebagai data tidak diketahui."
    if batas_atas is not None and nilai > batas_atas:
        return np.nan, f"Nilai {nilai:g} berada di atas batas maksimum {batas_atas}; diperlakukan sebagai data tidak diketahui."
    return float(nilai), None


st.markdown('<div class="eyebrow">EARLY BIRD · DATA SCIENCE</div>', unsafe_allow_html=True)
st.markdown('<h1 class="main-title">Prediksi Harga Rumah</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtext">Estimasi harga penawaran berdasarkan spesifikasi rumah dan wilayah.</p>',
    unsafe_allow_html=True,
)

try:
    paket = load_model()
except Exception as exc:
    st.error("Model belum bisa dimuat. Periksa artefak model dan versinya.")
    st.code(str(exc))
    st.stop()

st.caption(
    f"Model aktif: {paket.get('nama_model', 'Model regresi')} · "
    f"MAE test: {rupiah(paket['mae_test_rupiah'])}"
)

with st.form("form_prediksi"):
    st.subheader("Spesifikasi rumah")
    col1, col2 = st.columns(2)

    with col1:
        kt_unknown = st.checkbox("Kamar tidur tidak diketahui")
        kt = st.number_input(
            "Kamar tidur", min_value=0, max_value=20, value=3, step=1,
            disabled=kt_unknown,
        )
        lt_unknown = st.checkbox("Luas tanah tidak diketahui")
        lt = st.number_input(
            "Luas tanah (m²)", min_value=0, max_value=100_000, value=90, step=5,
            disabled=lt_unknown,
        )

    with col2:
        km_unknown = st.checkbox("Kamar mandi tidak diketahui")
        km = st.number_input(
            "Kamar mandi", min_value=0, max_value=20, value=2, step=1,
            disabled=km_unknown,
        )
        lb_unknown = st.checkbox("Luas bangunan tidak diketahui")
        lb = st.number_input(
            "Luas bangunan (m²)", min_value=0, max_value=10_000, value=70, step=5,
            disabled=lb_unknown,
        )

    daftar_kecamatan = paket.get("kecamatan", [])
    opsi_kecamatan = daftar_kecamatan + ["Lainnya / tidak tercantum"]
    kec_dipilih = st.selectbox("Kecamatan / wilayah", opsi_kecamatan)
    submitted = st.form_submit_button("Prediksi harga", use_container_width=True)

if submitted:
    warnings = []
    kt_value, msg = nilai_fitur(kt, kt_unknown, 0, 10)
    if msg:
        warnings.append("Kamar tidur: " + msg)
    km_value, msg = nilai_fitur(km, km_unknown, 0, 10)
    if msg:
        warnings.append("Kamar mandi: " + msg)
    lt_value, msg = nilai_fitur(lt, lt_unknown, 10, 100_000)
    if msg:
        warnings.append("Luas tanah: " + msg)
    lb_value, msg = nilai_fitur(lb, lb_unknown, 10, 10_000)
    if msg:
        warnings.append("Luas bangunan: " + msg)

    kecamatan = "Lainnya" if kec_dipilih == "Lainnya / tidak tercantum" else kec_dipilih
    input_df = pd.DataFrame([{
        "KT": kt_value,
        "KM": km_value,
        # Garasi tidak dipilih sebagai fitur final pada notebook ini.
        "Garasi": np.nan,
        "LT": lt_value,
        "LB": lb_value,
        "Kecamatan": kecamatan,
    }])

    if pd.notna(lt_value) and pd.notna(lb_value) and lt_value > 0:
        area_ratio = lb_value / lt_value
        if area_ratio > 10:
            warnings.append(
                "Perbandingan luas bangunan terhadap tanah sangat tinggi. Nilai tetap diproses "
                "sesuai pipeline training, tetapi hasil perlu ditafsirkan dengan hati-hati."
            )

    fitur = buat_fitur(input_df)
    kolom_model = paket["kolom"]
    kolom_hilang = [kol for kol in kolom_model if kol not in fitur.columns]

    if kolom_hilang:
        st.error(f"Fitur berikut tidak tersedia untuk model: {kolom_hilang}")
        st.stop()

    hasil = float(paket["model"].predict(fitur[kolom_model])[0])
    jumlah_missing = int(input_df[["KT", "KM", "LT", "LB"]].isna().sum(axis=1).iloc[0])

    st.markdown(
        f"""
        <div class="price-card">
            <div class="price-label">Estimasi harga penawaran</div>
            <div class="price-value">{rupiah(hasil)}</div>
            <div class="price-note">Hasil model berdasarkan fitur yang tersedia. Bukan appraisal resmi.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for pesan in warnings:
        st.warning(pesan)

    if jumlah_missing >= 2:
        st.warning(
            f"Ada {jumlah_missing} dari 4 fitur fisik yang tidak diketahui atau berada di luar batas domain. "
            "Prediksi bisa kurang andal karena model harus mengandalkan imputasi."
        )

    if hasil >= 10_000_000_000:
        st.warning(
            "Prediksi berada pada segmen harga tinggi. Pada evaluasi notebook, model lebih sulit "
            "memprediksi rumah mahal dibandingkan listing pada rentang harga yang lebih umum."
        )

    st.info(
        "Model dilatih untuk listing dengan harga penawaran Rp100 juta–Rp100 miliar. "
        "Hasil ini merupakan estimasi, bukan harga transaksi atau appraisal resmi. "
        "Akurasi dapat lebih rendah untuk harga tinggi, lokasi yang jarang muncul, "
        "atau listing dengan informasi fisik yang tidak lengkap."
    )

with st.expander("Tentang model"):
    st.write(f"Model final: {paket.get('nama_model', 'regresi')}")
    st.write(f"MAE pada test set: {rupiah(paket['mae_test_rupiah'])}.")
    if "cv_mae_5fold_juta" in paket:
        st.write(
            f"MAE CV 5-fold pada {paket.get('cv5_n', 'n/a')} baris training: "
            f"{rupiah(paket['cv_mae_5fold_juta'] * 1e6)}."
        )
    if "rmse_test_rupiah" in paket:
        st.write(f"RMSE test: {rupiah(paket['rmse_test_rupiah'])}.")
    if "mape_test_pct" in paket and "r2_test" in paket:
        st.write(f"MAPE: {paket['mape_test_pct']:.1f}% · R²: {paket['r2_test']:.2f}.")
    st.write(
        "Jumlah garasi tidak diminta karena fitur tersebut tidak terpilih dalam feature selection "
        "final pada notebook v8_another_tuned."
    )
