# IMDB Sentiment Analysis API 🚀

Bu proje, **IMDB film yorumları** veri seti kullanılarak eğitilmiş üç farklı model (GRU, LSTM, LoRA-BERT) ile **film yorumlarının duygu analizi** yapılmasını sağlar.  
En iyi performansı LoRA-Bert verdiğinden sadece o model API, **FastAPI** ile sunulmuştur
Modele ulaşmak için :
https://huggingface.co/UfukCem/imdb-bert-LoRA-finetuned
Docker konteynerinde çalıştırılabilir. Ayrıca basit bir UI için **Streamlit** entegrasyonu da mevcuttur.

---

## 📌 Özellikler

- **Modeller**:
  - GRU
  - LSTM
  - LoRA-BERT (fine-tuned, en iyi performans)
- **API**:
  - `/` → Servis durumu
  - `/predict` → Duygu tahmini (POST)
- **Docker** ile containerized deployment
- **Streamlit UI** ile kolay test
- **Logging** ile her tahminin detaylı kaydı
- **GPU destekli inference** (Colab veya yerel GPU ortamı ile)

---

## 🛠 Kurulum

1. Repository'i klonlayın:

```bash
git clone https://github.com/UfukCemDELICE/Task.git
cd Task
uv install -r pyproject.toml
```
Docker ile çalıştırmak için:
```bash
docker build -t imdb-sentiment-api .
docker run -p 8000:8000 imdb-sentiment-api
```
Streamlit UI ile test etmek için (Dockerla çalışırken):
```bash
uv run streamlit ui.py
```
Tarayıcı veya Postman ile API' yi test etmek için:
```bash
http://localhost:8000
```

| Model     | Accuracy | Precision | Recall | F1 Score |
| --------- | -------- | --------- | ------ | -------- |
| GRU       | 0.856    | 0.851     | 0.862  | 0.857    |
| LSTM      | 0.8545   | 0.856     | 0.852  | 0.854    |
| LoRA-BERT | 0.9015   | 0.891     | 0.913  | 0.902    |

LoRA-BERT modeli en yüksek performansa sahiptir ve API için varsayılan olarak kullanılmıştır.

**Teknolojiler**

Python 3.13

PyTorch

Transformers (HuggingFace)

PEFT / LoRA

FastAPI + Uvicorn

Docker

Streamlit (opsiyonel UI)

**Logging**

Her tahmin için aşağıdaki bilgiler kaydedilir:

Tahmin edilen duygu

Confidence (olasılık)

Tahmin süresi

Hatalar (Exception logları)