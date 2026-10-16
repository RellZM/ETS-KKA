# 🛡️ Stealth Hacker Grid: AI Pathfinding Visualizer
> **Evaluasi Tengah Semester (ETS) - Konsep Kecerdasan Artifisial (KKA)**  
> Simulasi Penetrasi Jaringan Komputer Berbasis Web menggunakan Algoritma AI Search.

---

## 📌 Deskripsi Proyek

Aplikasi ini merupakan simulasi interaktif berbasis web untuk memvisualisasikan bagaimana agen kecerdasan buatan (*AI Agent / Hacker*) mencari rute penetrasi optimal menuju basis data utama (*Main Database*) di dalam topologi jaringan server.

Visualisasi ini membandingkan kinerja tiga algoritma pencarian jalur (*pathfinding*):
1. **Uninformed Search:**
   - **BFS (Breadth-First Search)**
   - **UCS (Uniform-Cost Search)**
2. **Informed Search:**
   - **A\* Search (dengan Manhattan Heuristic)**

---

## ⚙️ Representasi Grid & Nilai Bobot (Cost Map)

Ukuran grid simulasi adalah **12 × 12** sel dengan aturan:

| Tipe Sel | Nilai Grid | Bobot (Cost) | Keterangan |
| :--- | :---: | :---: | :--- |
| **Start (Hacker)** | Koordinat `(1, 1)` | - | Titik awal penetrasi jaringan (*HCK*) |
| **Goal (Main DB)** | Koordinat `(10, 10)` | - | Titik target database utama (*DB*) |
| **Server Normal** | `0` | **1** | Server standar dengan tingkat risiko deteksi minimal |
| **Firewall (Tembok)** | `1` | **∞ (Blocked)** | Lapisan keamanan absolut yang tidak dapat ditembus |
| **Sensor IDS / Honeypot** | `2` | **10** | Perangkat keamanan deteksi intrusi dengan risiko alarm sangat tinggi |

---

## 🧠 Analisis Perbandingan Algoritma

| Algoritma | Kategori | Heuristik | Karakteristik Jalur & Perilaku |
| :--- | :--- | :--- | :--- |
| **BFS** *(Breadth-First Search)* | Uninformed | Tidak ada | Mengutamakan jumlah lompatan (*hop*) paling sedikit. **Mengabaikan bobot risiko**, sehingga sering menerobos sel IDS (risiko alarm tinggi / *Terdeteksi IDS*). |
| **UCS** *(Uniform-Cost Search)* | Uninformed | Tidak ada | Memilih rute dengan total bobot akumulasi cost terendah ($g(n)$). Menemukan rute *stealth* aman dengan memutari sensor IDS, namun mengeksplorasi banyak node (*expanded nodes* tinggi). |
| **A\* Search** | Informed | Manhattan Distance | Menggabungkan fungsi biaya riil $g(n)$ dan estimasi heuristik $h(n)$ ($f(n) = g(n) + h(n)$). Menemukan rute optimal yang sama amannya dengan UCS, namun jauh lebih efisien dalam jumlah ekspansi node. |

---

## 🛠️ Prasyarat Sistem

- **Python** versi 3.8 atau yang lebih baru.
- **pip** (Python package installer).
- Web browser modern (Google Chrome, Firefox, Microsoft Edge, dll).

---

## 🚀 Panduan Instalasi & Menjalankan Program

### 1. Buka Terminal / PowerShell
Arahkan terminal ke folder proyek ini:
```bash
cd "d:\Koding-serius\KKA\ETS"
```

### 2. Install Dependensi
Install pustaka Flask yang dibutuhkan:
```bash
pip install -r requirements.txt
```
*Atau secara langsung:*
```bash
pip install flask
```

### 3. Jalankan Aplikasi
Jalankan file utama [ETS.py](ETS.py):
```bash
python ETS.py
```

Setelah dijalankan, terminal akan menampilkan log seperti berikut:
```text
 * Serving Flask app 'ETS'
 * Debug mode: on
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
```

### 4. Buka di Web Browser
Buka browser Anda dan akses tautan:
👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)** atau **[http://localhost:5000](http://localhost:5000)**

---

## 🕹️ Panduan Penggunaan Antarmuka

1. **Menggambar / Memodifikasi Peta (Toolbar Kiri):**
   - Klik tombol **Firewall (Tembok)** untuk memasang rintangan yang memblokir jalur.
   - Klik tombol **Sensor IDS (Cost: 10)** untuk meletakkan jebakan/sensor berisiko tinggi.
   - Klik tombol **Server Normal (Cost: 1)** untuk mengembalikan sel ke kondisi normal.
   - Klik tombol **Reset Map Default** untuk mengembalikan peta ke skenario default ETS.
2. **Menjalankan Algoritma:**
   - Klik salah satu tombol eksekusi: **Jalankan BFS**, **Jalankan UCS**, atau **Jalankan A\***.
   - Amati animasi warna biru muda (*node yang dieksplorasi*) dan warna hijau (*jalur akhir penetrasi*).
3. **Membaca Tabel Komparasi:**
   - Tabel di panel kanan otomatis memperbarui perbandingan:
     - **Langkah (Hop)**: Jumlah langkah dari Start ke Goal.
     - **Risk Cost**: Akumulasi total bobot rute yang dilalui.
     - **Node**: Total simpul yang dieksplorasi (*expanded nodes*).
     - **Waktu**: Durasi komputasi algoritma (dalam milidetik).
     - **Status**: Status deteksi (*Stealth Aman* vs *Terdeteksi IDS*).

---

## 📂 Struktur File

```text
ETS/
├── ETS.py              # Backend Flask (AI Search Logic) + Template Frontend Interaktif
├── requirements.txt    # Daftar dependensi Python (Flask)
└── README.md           # Dokumentasi dan panduan eksekusi proyek
```

---

## 🛑 Cara Menghentikan Server

Untuk menghentikan server Flask, kembali ke jendela terminal yang sedang menjalankan program, lalu tekan kombinasi tombol:
`CTRL + C`
