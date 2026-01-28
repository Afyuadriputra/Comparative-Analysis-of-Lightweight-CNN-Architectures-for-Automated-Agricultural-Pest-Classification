# Comparative Analysis of Lightweight CNN Architectures for Automated Agricultural Pest Classification

Repository ini memuat implementasi dan evaluasi komparatif sistem klasifikasi citra otomatis untuk identifikasi hama pertanian. Proyek ini berfokus pada penerapan arsitektur Deep Learning yang efisien secara komputasi (Lightweight Models), yakni **MobileNetV3** dan **ShuffleNet V2**, menggunakan framework PyTorch.

Penelitian ini bertujuan untuk menghasilkan model yang tidak hanya memiliki akurasi tinggi dalam membedakan 9 jenis taksa hama, tetapi juga optimal untuk diterapkan pada lingkungan dengan sumber daya terbatas (*resource-constrained environments*).

## Daftar Isi

1.  Latar Belakang & Dataset
2.  Metodologi Eksperimen
3.  Evaluasi Kinerja
4.  Prasyarat Sistem
5.  Panduan Instalasi & Eksekusi
6.  Struktur Repositori

## Latar Belakang & Dataset

Dataset yang digunakan dalam studi ini dikurasi secara spesifik untuk tugas klasifikasi hama (*pest classification*). Data diorganisir menggunakan struktur direktori standar `ImageFolder` dan dipartisi dengan rasio 70:15:15 untuk pelatihan, validasi, dan pengujian.

### Taksonomi Kelas
Sistem dikonfigurasi untuk mengidentifikasi 9 kelas hama berikut:
1.  Fly (Lalat)
2.  Wasp (Tawon)
3.  Aphids (Kutu Daun)
4.  Armyworm (Ulat Tentara)
5.  Bollworm (Ulat Buah)
6.  Grasshopper (Belalang)
7.  Mites (Tungau)
8.  Mosquito (Nyamuk)
9.  Stem Borer (Penggerek Batang)

## Metodologi Eksperimen

Proyek ini menerapkan dua skema eksperimen terpisah untuk menguji efektivitas strategi *Transfer Learning* dan *Fine-Tuning*.

### Skema 1: The Minimalist (MobileNetV3 Small)
Pendekatan ini memanfaatkan arsitektur MobileNetV3 Small dengan strategi *feature extraction*.
* **Resolusi Input:** 160x160 piksel
* **Strategi Pelatihan:** Frozen Feature Extractor (Bobot backbone dibekukan)
* **Tujuan:** Menguji kemampuan representasi fitur pre-trained pada domain baru dengan biaya komputasi minimal.

### Skema 2: Speed Racer (ShuffleNet V2)
Pendekatan ini menggunakan arsitektur ShuffleNet V2 x0.5 dengan strategi *full fine-tuning* dan akselerasi modern.
* **Resolusi Input:** 224x224 piksel
* **Strategi Pelatihan:** End-to-End Training dengan Automatic Mixed Precision (AMP).
* **Optimisasi:** Penggunaan OneCycleLR Scheduler untuk konvergensi yang lebih cepat dan stabil.

## Evaluasi Kinerja

Berdasarkan pengujian pada *test set* yang belum pernah dilihat model sebelumnya, kedua arsitektur menunjukkan performa klasifikasi yang superior.

| Metrik Evaluasi | Skema 1 (MobileNetV3) | Skema 2 (ShuffleNet V2) |
| :--- | :--- | :--- |
| **Akurasi Global** | 98% | 98% |
| **Presisi (Macro Avg)** | 0.98 | 0.98 |
| **Recall (Macro Avg)** | 0.98 | 0.98 |
| **F1-Score (Macro Avg)** | 0.98 | 0.98 |

Hasil eksperimen menunjukkan bahwa kedua model ringan ini mampu mencapai tingkat generalisasi yang sangat tinggi tanpa memerlukan arsitektur yang berat (*heavyweight models*). Laporan klasifikasi detail, Confusion Matrix, dan kurva ROC dihasilkan secara otomatis pada akhir eksekusi program.

## Prasyarat Sistem

Untuk mereplikasi hasil eksperimen, pastikan lingkungan kerja Anda memenuhi dependensi berikut:

* **Bahasa Pemrograman:** Python 3.8 atau lebih baru
* **Deep Learning Framework:** PyTorch & Torchvision
* **Komputasi Numerik & Visualisasi:** NumPy, Matplotlib, Scikit-learn
* **Utilitas:** Tqdm

## Panduan Instalasi & Eksekusi

Ikuti langkah-langkah berikut untuk menjalankan simulasi:

1.  **Instalasi Dependensi**
    Jalankan perintah berikut pada terminal:
    ```bash
    pip install torch torchvision scikit-learn matplotlib numpy tqdm
    ```

2.  **Konfigurasi Dataset**
    Sesuaikan path direktori dataset pada skrip utama. Pastikan dataset mengikuti struktur hierarki folder per kelas.
    ```python
    # Pada file main.py
    DATA_DIR = r'path/to/your/dataset/directory'
    ```

3.  **Eksekusi Program**
    Jalankan skrip utama:
    ```bash
    python main.py
    ```

    Program akan secara berurutan melatih Skema 1 dan Skema 2, menyimpan bobot model (`.pth`), dan menampilkan visualisasi evaluasi.

## Struktur Repositori

* **Dataset Wrapper:** Implementasi kelas `ApplyTransform` untuk manajemen augmentasi data yang fleksibel pada subset train/val/test.
* **Modul Pelatihan:** Fungsi terpisah (`jalankan_skema_1`, `jalankan_skema_2`) untuk isolasi eksperimen.
* **Modul Evaluasi:** Fungsi `evaluasi_dan_plot` untuk analisis metrik komprehensif (Classification Report, CM, ROC).
