# Deployment — House Price Prediction

Folder ini menyiapkan aplikasi Streamlit untuk model dari notebook `Early_Bird_v9.ipynb`.

## Penting sebelum deploy

`model_harga_rumah.joblib` harus dibuat dari cell deployment terakhir di notebook yang dipilih. Jangan menggunakan artefak dari versi notebook lain karena struktur fitur dan metadata model dapat berbeda.

## File yang harus ada di root repository

```text
house-price-prediction/
├── app.py
├── requirements.txt
├── model_harga_rumah.joblib
├── Early_Bird_v9.ipynb
└── README.md
```

Dataset CSV tidak diperlukan untuk menjalankan aplikasi setelah artefak model tersedia. Sertakan dataset hanya jika memang dibutuhkan untuk dokumentasi/reproduksi, dan pastikan boleh dipublikasikan.

## 1. Buat artefak dari notebook terpilih

1. Buka `Early_Bird_v9.ipynb` di Colab.
2. Pastikan semua cell sudah dijalankan dengan urutan terbaru.
3. Jalankan ulang cell yang menyimpan `model_harga_rumah.joblib` dan `requirements.txt`.
4. Download kedua file tersebut dari Colab. Pastikan nama file persis seperti di atas.
5. Letakkan `model_harga_rumah.joblib` di folder yang sama dengan `app.py`.

Aplikasi ini mengharapkan metadata yang disimpan notebook: `model`, `kolom`, `kecamatan`, `nama_model`, dan `mae_test_rupiah`. Ia juga memeriksa apakah daftar fitur cocok dengan feature selection notebook Tuned.

## 2. Uji lokal di Ubuntu

Buka terminal di folder proyek, lalu jalankan:

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Buka URL lokal yang muncul di terminal, biasanya `http://localhost:8501`.

Coba minimal tiga kasus:

- input normal: KT 3, KM 2, LT 90, LB 70, Kecamatan Sawangan;
- LB tidak diketahui;
- kecamatan `Lainnya / tidak tercantum`.

Pastikan hasil prediksi muncul, pesan peringatan masuk akal, dan terminal tidak menunjukkan traceback. Output prediksi harus sejalan dengan tes contoh di notebook; selisih bisa muncul jika data input atau artefak model berbeda.

## 3. Upload ke GitHub

Buat repository GitHub baru, misalnya `early-bird-house-price`, lalu letakkan `app.py`, `requirements.txt`, `model_harga_rumah.joblib`, notebook, dan README di root repository. Jangan memasukkan folder `.venv/` atau cache.

```bash
git init
git add app.py requirements.txt model_harga_rumah.joblib README.md Early_Bird_House_Price_Prediction_v8_another_tuned_rapi.ipynb
git commit -m "Prepare house price prediction app"
git branch -M main
git remote add origin https://github.com/<USERNAME>/<REPOSITORY>.git
git push -u origin main
```

Ganti `<USERNAME>` dan `<REPOSITORY>` dengan nilai milikmu. Jika repo sudah ada, ikuti workflow repo tersebut dan jangan jalankan `git init` lagi.

Model dihasilkan lewat joblib dan berukuran puluhan MB. Periksa ukuran serta status Git sebelum push. Jangan commit file sensitif, token, password, atau rahasia akun.

## 4. Deploy ke Streamlit Community Cloud

1. Buka https://share.streamlit.io/ dan login menggunakan GitHub.
2. Pilih **Create app**.
3. Pilih repository, branch `main`, dan entrypoint `app.py`.
4. Pada **Advanced settings**, pilih Python 3.12 agar sesuai dengan dependensi yang digunakan selama notebook dijalankan, bila versi tersebut tersedia di pilihan.
5. Klik **Deploy**.
6. Tunggu log build selesai, lalu buka URL yang diberikan Streamlit.

Dokumentasi resmi:
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies

## Catatan metodologi

Model final dari notebook Tuned menggunakan fitur hasil feature selection. `app.py` di folder ini membuat fitur sesuai fungsi di notebook dan memilih kolom dari metadata artefak. Jumlah garasi tidak diminta karena tidak terpilih sebagai fitur final. Harga yang tampil adalah estimasi harga penawaran, bukan appraisal resmi.
