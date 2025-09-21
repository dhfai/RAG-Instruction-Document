# Orchestrated RAG System - Modul Ajar Generator

Sistem RAG multi-strategi untuk pembuatan modul ajar otomatis dengan optimisasi matematis dan validasi komprehensif.

## 🎯 Fitur Utama

- **Multi-Strategy Retrieval**: 5 strategi retrieval (S1-S5) dengan optimisasi matematis
- **Multi-Strategy Generation**: 5 strategi generasi (G1-G5) dengan validasi terintegrasi
- **Cost-Aware Routing**: Optimisasi biaya dan latensi menggunakan utility functions
- **Mathematical Features**: Feature engineering komprehensif untuk chunks dan queries
- **Claim Validation**: Validasi klaim, konsistensi numerik, dan deteksi kontradiksi
- **Web Content Enhancement**: Pengayaan konten dari web scraping Google
- **FAISS Vector Search**: Database vektor untuk pencarian semantik
- **MongoDB Storage**: Penyimpanan hasil dan logging

## 🏗️ Struktur Proyek

```
multiple/
├── config/           # Konfigurasi sistem
├── core/            # Komponen inti RAG
├── strategies/      # Implementasi strategi S1-S5 dan G1-G5
├── models/          # Schema data dan models
├── storage/         # Database dan penyimpanan
├── utils/          # Utilities dan web scraping
├── data/           # Data lokal modul ajar
├── templates/      # Template modul ajar
└── main.py         # API server utama
```

## 📋 Strategi yang Tersedia

### Retrieval Strategies
- **S1**: Single-pass Dense - Retrieval sederhana dengan semantic similarity
- **S2**: Hybrid BM25+Dense - Kombinasi keyword dan semantic matching
- **S3**: Multi-hop Iterative - Retrieval bertahap untuk query kompleks
- **S4**: Clustered Selection - Seleksi dokumen beragam dari berbagai cluster
- **S5**: Query Rewrite/Decomposition - Dekomposisi query untuk handling ambiguitas

### Generation Strategies
- **G1**: Fusion-in-Decoder - Fusi multi-dokumen dalam decoder
- **G2**: Rerank-then-Generate - Reranking sebelum generasi
- **G3**: Chain-of-Thought - Reasoning eksplisit step-by-step
- **G4**: Evidence-Aware - Generasi dengan sitasi eksplisit
- **G5**: Validator-Augmented - Generasi dengan validasi terintegrasi

## 🚀 Instalasi dan Setup

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Setup Environment**
```bash
# Copy dan edit .env file
copy .env.example .env
# Isi OPENAI_API_KEY dan konfigurasi lainnya
```

3. **Setup MongoDB**
```bash
# Pastikan MongoDB running di localhost:27017
# Atau update MONGODB_URL di .env
```

4. **Siapkan Data**
```bash
# Letakkan file modul ajar (.docx, .pdf, .txt) di folder data/modul_ajar/
# Sistem akan otomatis memproses dan membuat index
```

5. **Jalankan Server**
```bash
python main.py
```

Server akan running di `http://localhost:8000`

## 📚 API Endpoints

### Utama
- `POST /generate` - Generate modul ajar
- `GET /health` - Health check sistem
- `GET /history` - Riwayat generasi

### Informasi
- `GET /strategies` - Info strategi yang tersedia
- `GET /template` - Struktur template modul ajar
- `GET /stats` - Statistik sistem

## 💡 Cara Penggunaan

### Generate Modul Ajar
```python
import requests

request_data = {
    "nama_guru": "Siti Aisyah",
    "nama_sekolah": "SDN 1 Jakarta",
    "mata_pelajaran": "Matematika",
    "topik": "Pecahan",
    "sub_topik": "Penjumlahan Pecahan",
    "alokasi_waktu": "2 x 35 menit",
    "kelas": "5",
    "fase": "C",
    "llm_used": "openai"
}

response = requests.post("http://localhost:8000/generate", json=request_data)
result = response.json()
```

### Response Structure
```json
{
    "request_id": "uuid",
    "generated_sections": {
        "identitas": {...},
        "tujuan_pembelajaran": {...},
        "profil_pelajar_pancasila": {...},
        // ... sections lainnya
    },
    "overall_quality_score": 0.85,
    "total_cost": 0.15,
    "processing_time": 12.5,
    "strategies_used": {...},
    "created_at": "2024-01-01T10:00:00"
}
```

## 🔧 Konfigurasi

### Settings Utama (`config/settings.py`)
- Database connections
- Model configurations
- Strategy parameters
- Cost optimization weights
- Feature engineering weights

### Environment Variables (`.env`)
- `OPENAI_API_KEY`: API key OpenAI
- `MONGODB_URL`: URL MongoDB
- `LOG_LEVEL`: Level logging

## 📊 Monitoring dan Logging

Sistem menggunakan logging yang cantik dengan:
- Console output berwarna dengan Rich
- File logging dengan rotasi
- Error tracking
- Performance metrics
- Request/response logging

Log files tersimpan di folder `logs/`:
- `app.log` - Log utama aplikasi
- `error.log` - Log khusus error

## 🎨 Mathematical Foundations

Sistem menggunakan pendekatan matematis untuk:

### Feature Extraction
- Entropy calculation untuk information content
- Lexical diversity measurement
- Dependency scoring dengan weighted combination
- Query ambiguity using entropy

### Utility Functions
- Strategy utility dengan weighted features
- Cost-aware expected utility calculation
- Confidence-based fallback mechanisms
- Probability distribution untuk strategy selection

### Validation
- Faithfulness scoring dengan cosine similarity
- Numeric consistency checking
- Contradiction detection using NLI
- Overall confidence calculation

## 🔄 Workflow

1. **Content Retrieval**: Mencari konten relevan dari database lokal + web
2. **Feature Extraction**: Ekstraksi fitur matematis dari query dan chunks
3. **Strategy Selection**: Pemilihan strategi optimal berdasarkan utility
4. **Content Generation**: Generasi konten menggunakan strategi terpilih
5. **Validation**: Validasi klaim dan konsistensi konten
6. **Synthesis**: Penggabungan dan finalisasi output

## 📈 Performance

- **Latency**: ~10-30 detik per modul ajar (tergantung kompleksitas)
- **Quality**: Score rata-rata 0.75-0.90 dengan validasi
- **Cost**: Optimisasi token usage dan API calls
- **Scalability**: Horizontal scaling ready dengan async/await

## 🛠️ Development

### Adding New Strategies
1. Inherit dari `BaseRetrievalStrategy` atau `BaseGenerationStrategy`
2. Implement method `retrieve()` atau `generate()`
3. Add ke factory classes
4. Update utility calculations di `RouterEngine`

### Customizing Templates
Edit file di `templates/template.json` untuk mengubah struktur modul ajar.

## 📝 License

Proyek ini untuk keperluan pendidikan dan pengembangan sistem RAG.

## 👨‍💻 Support

Untuk pertanyaan dan support, silakan buka issue di repository ini.
