# Pundit Deculs - IDSSE v2

Kode, model beku, dependensi dan laporan untuk kandidat v2.
Data challenge tidak disertakan. Dokumen metode: Pundit_Deculs.md.

## Instalasi dan inferensi

Gunakan versi Python di ENVIRONMENT.json. Buat environment di luar folder kode:

    python -m venv "E:\MTAH2027\.venv_reproduce_v2"
    call "E:\MTAH2027\.venv_reproduce_v2\Scripts\activate.bat"
    python -m pip install -r requirements.txt
    python pundit_v2.py candidate --data-dir "E:\More Than A Hack 2027 - Initial Challenge - 13"
    python evaluation\validate_submission.py evaluation\candidates\idsse_v2\Pundit_Deculs.csv
    certutil -hashfile evaluation\candidates\idsse_v2\Pundit_Deculs.csv SHA256

Hash yang diharapkan ada di EXPECTED_SUBMISSION.sha256.
Mode candidate memakai model beku, tanpa melatih ulang atau membaca GT.
Muat bobot joblib hanya dari sumber yang dipercaya.

## Mengulang eksperimen eksternal

Laporan yang sudah selesai dipakai ulang. Untuk eksperimen baru, gunakan folder lain:

    python pundit_v2.py develop --out evaluation\experiments\idsse_v2_retrained
    python pundit_v2.py test --out evaluation\experiments\idsse_v2_retrained
    python pundit_v2.py diagnose --out evaluation\experiments\idsse_v2_retrained

Pengembangan mengunduh XML IDSSE publik ke data/external/idsse.
Bobot v1 disertakan sebagai pembanding eksternal. Model submission tidak ditimpa
oleh eksperimen di folder lain. Jangan memilih model berdasarkan GT challenge.

## Isi dan hak penggunaan

- Upstream Metrica: MIT, lihat LICENSE-METRICA.
- Data eksternal: CC BY 4.0, lihat DATA_SOURCES.md.
- Kode tim: MIT, lihat LICENSE-TEAM.
- Video, ATD, tagging, homography, koordinat turunan challenge dan GT tidak disertakan.

.gitattributes mempertahankan byte file ketika Git melakukan checkout.
.gitignore mengecualikan data dan output lokal.
Paket ini belum dipublikasikan atau dikirim oleh program pengemasan.
