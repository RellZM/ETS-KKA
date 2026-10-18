# Visualisasi Water Jug (BFS, UCS, A*)
> **Evaluasi Tengah Semester (ETS) - Konsep Kecerdasan Artifisial (KKA)**  
> Implementasi Web Simulasi & Komparasi Algoritma Pencarian Ruang Keadaan (*State Space Search*).

---

## 1. Alur Aplikasi Web
Aplikasi berjalan dalam enam langkah, dari input masalah sampai perbandingan BFS, UCS, dan A*:
- **Input masalah:** jumlah teko (3 sampai 5), kapasitas tiap teko, dan target volume yang ingin diukur (< kapasitas salah satu teko). Isi awal ditentukan otomatis (teko 1 penuh, lainnya kosong).
- **Validasi input:** cek batasan kapasitas dan target sebelum pencarian dimulai.
- **Pilih algoritma:** BFS, UCS, A*, atau jalankan ketiganya sekaligus.
- **Eksekusi dan visualisasi:** tampilkan langkah demi langkah dengan tombol *Next*, *Prev*, *Play*, *Pause*, dan pengatur kecepatan.
- **Hasil:** jalur solusi, animasi tuang air, dan tabel metrik.
- **Perbandingan:** tabel ketiga algoritma berdampingan.

---

## 2. Aturan Input dan Validasi
- **Jumlah teko:** 3 sampai 5. Mempermudah penjelasan teka-teki multi-teko.
- **Kapasitas tiap teko:** bilangan bulat positif (> 0).
- **Isi awal:** otomatis ditentukan (Teko 1 terisi penuh sebagai wadah utama, teko lainnya kosong 0 L).
- **Target volume:** bilangan bulat positif (> 0) dan harus lebih kecil dari kapasitas teko terbesar (< max kapasitas teko).
- Total start = total goal. Kalau tidak sama, tampilkan pesan error, misalnya *"Goal 10 L, padahal air start 8 L"*.
- Start tidak boleh sama dengan goal. Kalau sama, langkahnya 0.
- **Tanpa solusi:** kalau pencarian selesai tanpa menemukan goal, tampilkan *"Tidak ada solusi"*.

---

## 3. Aturan Aksi (Operator)
- Hanya ada satu aksi, yaitu menuang dari teko $i$ ke teko $j$, dan state selalu berupa daftar isi tiap teko, misalnya $(8, 0, 0)$.
- **Syarat tuang:** $i \neq j$, teko $i$ tidak kosong, dan teko $j$ belum penuh.
- **Jumlah yang dipindah:** $\min(\text{isi teko } i, \text{ sisa ruang teko } j)$. Tuangan berhenti saat teko $i$ kosong atau teko $j$ penuh.
- **Jumlah kemungkinan aksi:** dengan $n$ teko ada $n \times (n - 1)$ tuangan dari tiap state.
- Tidak ada aksi isi dari keran atau buang air, supaya total air tetap konstan.

---

## 4. Cost per Algoritma
BFS menghitung jumlah langkah, sedangkan UCS dan A* menghitung total liter yang dipindah, sehingga jalur terbaik tiap algoritma bisa berbeda:

| Algoritma | Cost | Struktur Data | Hasil Optimal Berdasarkan |
| :--- | :--- | :--- | :--- |
| **BFS** | Tidak memakai cost; tiap tuangan = 1 langkah | Queue (FIFO) | Jumlah langkah paling sedikit |
| **UCS** | $g(n) = \text{total liter yang sudah dipindah}$ | Priority queue berdasarkan $g$ | Total liter dipindah paling sedikit |
| **A\*** | $g(n)$ sama dengan UCS, plus $h(n)$ | Priority queue berdasarkan $f = g + h$ | Total liter dipindah paling sedikit |

---

## 5. Heuristik A*
Heuristik yang dipakai adalah setengah dari total selisih isi tiap teko terhadap goal, dan nilainya tidak pernah melebihi cost sebenarnya:

$$h(n) = \frac{1}{2} \sum_{k} \left| \text{isi}_k(n) - \text{goal}_k \right|$$

- **Contoh:** state $(8, 0, 0)$ dengan goal $(4, 4, 0)$.
  - Selisih per teko: $|8 - 4| + |0 - 4| + |0 - 0| = 8$
  - $h = 8 \div 2 = 4$, artinya paling sedikit 4 liter harus berpindah sampai goal.
- Heuristik ini *admissible*. Satu tuangan sebanyak $a$ liter paling banyak mengurangi total selisih sebesar $2a$, jadi $h$ tidak pernah melebihi cost sebenarnya. Dengan begitu A* tetap optimal seperti UCS, tapi biasanya mengunjungi node lebih sedikit.

---

## 6. Aturan Struktur Data
Tiap algoritma memakai struktur data berbeda, tapi semuanya menyimpan *parent* supaya jalur solusi bisa ditelusuri mundur:
- **BFS:** *queue* dan *visited set*. State yang sudah dikunjungi tidak dimasukkan lagi.
- **UCS dan A\*:** *priority queue* dan simpanan $g$ terbaik per state. Kalau state yang sama ditemukan lewat jalur lebih murah, perbarui nilainya.
- **Tie-breaking:** kalau prioritas sama, ambil node yang masuk lebih dulu, supaya animasi selalu konsisten.
- **Parent:** simpan state asal tiap node, lalu telusuri mundur dari goal untuk mendapat jalur solusi.

---

## 7. Elemen Visualisasi
Visualisasi menampilkan lima hal supaya proses pencarian terlihat, bukan hanya hasil akhirnya:
- **Teko dan air:** gambar teko dengan level air proporsional kapasitas, penanda skala liter, dan penyorotan teko asal (*source*) dan teko tujuan (*target*) saat menuang.
- **Panel OPEN dan CLOSED:** isi *frontier* (OPEN list) dan state yang sudah dikunjungi (CLOSED list).
- **Node yang diproses:** nilai $g(n)$, $h(n)$, dan $f(n)$ untuk node evaluasi saat ini.
- **Alur transisi & jalur solusi:** urutan langkah solusi yang dapat diklik dan disorot.
- **Kontrol playback:** tombol *Play*, *Pause*, *Next*, *Prev*, *Awal*, *Akhir*, slider timeline, dan pengatur kecepatan (*speed slider*).

---

## 8. Metrik Perbandingan

| Metrik | Diukur Untuk | Yang Diharapkan |
| :--- | :--- | :--- |
| **Jumlah langkah** | BFS, UCS, A* | BFS paling sedikit |
| **Total liter dipindah (cost)** | BFS, UCS, A* | UCS dan A* paling kecil |
| **Node di-expand** | BFS, UCS, A* | A* paling sedikit |
| **Node di-generate** | BFS, UCS, A* | A* paling sedikit |
| **Ukuran maksimum OPEN** | BFS, UCS, A* | Bergantung pada kasus uji |
| **Waktu eksekusi (ms)** | BFS, UCS, A* | Bergantung pada kasus uji |

---

## 9. Mode Penggunaan Aplikasi

### 🕹️ Mode 1: Coba Sendiri (Teka-Teki Interaktif)
- **Mekanisme Persis Video Puzzle (8–5–3 Variation):**
  - Klik **Teko Asal** (berisi air) lalu klik **Teko Tujuan** untuk menuang air.
  - Air berpindah secara otomatis sesuai sisa kapasitas atau ketersediaan air.
  - Dilengkapi tombol **Undo (Batalkan Langkah)**, **Reset**, dan **Hint (Petunjuk Langkah Optimal)**.
  - Saat volume target (misal: 4 Liter) tercapai di salah satu teko, modal ucapan selamat akan muncul beserta perbandingan langkah pemain vs solusi optimal AI.

### 🤖 Mode 2: Visualisasi AI (BFS, UCS, A*)
- Menjalankan pencarian solusi optimal secara otomatis dengan playback animasi (*Play*, *Pause*, *Next*, *Prev*, slider kecepatan).
- Menampilkan narasi langkah per langkah, evaluasi node ($g, h, f$), daftar OPEN/CLOSED, serta tabel perbandingan metrik ketiga algoritma.

---

## 10. Panduan Menjalankan Program

### 1. Clone Repositori
Buka Terminal / Command Prompt / PowerShell, lalu jalankan:
```bash
git clone https://github.com/RellZM/ETS-KKA.git
cd ETS-KKA
```

### 2. (Opsional) Siapkan Virtual Environment
Disarankan menggunakan virtual environment agar dependensi terisolasi:
```bash
python -m venv venv

# Windows (PowerShell / Command Prompt):
.\venv\Scripts\activate

# macOS / Linux:
source venv/bin/activate
```

### 3. Install Dependensi
Install paket yang diperlukan yang tercantum pada `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4. Jalankan Aplikasi
Jalankan server aplikasi web Flask:
```bash
python app.py
```

### 5. Akses Melalui Browser
Buka peramban web dan kunjungi alamat lokal:
**[http://127.0.0.1:5000](http://127.0.0.1:5000)**


