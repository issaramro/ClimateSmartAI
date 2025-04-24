# AI-Powered Climate & Drought Risk Management Platform

This project is a modular, containerized platform built to forecast climate variables, assess drought risk, and manage water availability. Designed with real-world impact in mind, the system is structured around three independent AI pipelines (IEPs) coordinated by an external ensemble endpoint (EEP). It uses MLOps principles, Docker, and cloud deployment via Azure.

---

## Project Structure

```
project_ai/
├── data_preprocessing/       # Scripts & CSVs for raw/cleaned data
├── IEP1_forecasting/         # LSTM model for climate forecasting
├── IEP2_drought_assessment/  # Classification model for drought risk (PDSI)
├── IEP3_water_availability/  # Regression model for water availability
├── EEP_interface/            # FastAPI interface to connect IEPs
├── prometheus.yml            # Prometheus config for metrics
└── docker-compose.yml        # Docker Compose file to run the platform
```

---

## Features

- **IEP1: Climate Forecasting**  
  LSTM model predicts climate variables up to a user-specified date based on historical data.

- **IEP2: Drought Risk Assessment**  
 RandomForest model and then threshold-based labeling, useing PDSI as a label, to evaluate drought severity.

- **IEP3: Water Availability Management**  
  RandomForest model and then threshold-based labeling to guide water resource planning.

- **EEP (External Ensemble Pipeline)**  
  Coordinates all IEPs, provides unified access through API or UI, and is fully containerized.

- **Monitoring with Prometheus**  
  Tracks real-time model activity and API health.

---

## 📊 Datasets on Google Drive

- 🔗 **Dataset (Google Drive):** [Access Cleaned Dataset]([https://drive.google.com/your-dataset-link](https://drive.google.com/drive/folders/19AYOqcLBMoMgqBZworMN_wgOyXXZHXM6?usp=drive_link))

---

## 💻 Local Setup

1. **Clone the Repository**

```bash
git clone https://github.com/your-username/project_ai.git
cd project_ai
```

2. **Build Docker Containers**

```bash
docker-compose build
```

3. **Run the System Locally**

```bash
docker-compose up
```

4. **Access the Interface**

Open your browser and visit:

```
http://localhost:8004  # EEP API (FastAPI docs)
http://localhost:3000  # If UI is hosted separately
```

---

## ☁️ Azure Deployment (Cloud)

> Ensure Docker images are pushed to Docker Hub and you’ve set up Azure Container Apps.

### Steps:

1. Push your images to Docker Hub

```bash
docker tag your-image username/your-image-name
docker push username/your-image-name
```

2. Go to Azure Portal → **Container Apps** → **Create App**

3. Link Docker Hub images and environment variables

4. Set up ingress for external access and monitoring

### 🔗 Live Demo (Azure)

Access the deployed system here:  
🌐 [Azure Deployment Link](https://your-azure-app-url.com)

---

## 🔍 Monitoring with Prometheus

1. Navigate to: [http://localhost:9090](http://localhost:9090)

2. Use Prometheus to track:
   - Requests per second
   - API latency
   - Individual model usage

---

## 🧪 Testing

- **Unit tests** included for each pipeline
- **Integration tests** for EEP-IEP interaction
- **End-to-end tests** (e.g., curl or Postman) to ensure full flow works

---

## ⚙️ MLOps Tools Used

- **FastAPI** for endpoints  
- **Docker & Docker Compose** for containerization  
- **Prometheus** for monitoring  
- **MLflow** for experiment tracking (optional future step)  
- **Azure** for cloud deployment

---

## 📘 Future Work

- Integrate MLflow for live experiment tracking  
- Add user authentication & authorization  
- Expand to real-time sensor data ingestion

---

## 👨‍🎓 Academic Notes

This project was developed for educational purposes in an advanced AI systems course. It meets all MLOps, testing, and deployment requirements and demonstrates practical business value in climate and water management.

---

## 📫 Contact

For feedback or questions, feel free to reach out via GitHub issues or email us at your.email@example.com
