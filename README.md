# 🌿 ClimateSmartAI: AI-Powered Climate Forecasting and Agricultural Assessment for Lebanon
Lebanon has what many countries envy: diverse climates, fertile valleys, and a rich agriculture. But despite this natural advantage, farmers are struggling due to unpredictable weather, limited access to data, and poor planning tools. That’s why we created a tool focused on supporting farmers and agribusinesses. By providing timely, data-driven insights, we help them make better decisions about planting, irrigation, and crop planning—ultimately building resilience in their work and contributing to national food sovereignty.
We believe that empowering farmers through technology is not just good business; it’s a necessity for Lebanon’s future.

This project is built to forecast climate variables, assess drought risk, and manage water availability. Designed with real-world impact in mind, the system is structured around three independent IEPs coordinated by an external ensemble endpoint (EEP). It uses MLOps principles, Docker, and cloud deployment via Azure.

---

## 🌍 Data Source (Google Earth Engine)

The raw climate data used for training and forecasting was sourced from **Google Earth Engine (GEE)**, a cloud-based platform for planetary-scale geospatial analysis.

We extracted the data of **Baalbek/Hermel, Lebanon - region**, from **1975 to 2024**, and was exported in CSV format for further preprocessing.

🔗 [Learn more about the data here]([https://earthengine.google.com/](https://developers.google.com/earth-engine/datasets/catalog/IDAHO_EPSCOR_TERRACLIMATE))

---

## Project Structure

```
project_ai/
├── data_preprocessing/       # To get cleaned data
├── IEP1_forecasting/         # LSTM model for climate forecasting
├── IEP2_drought_assessment/  # RandomForest model for drought risk
├── IEP3_water_availability/  # RandomForest model for water necessity
├── EEP_interface/            # FastAPI interface to communicate with IEPs
├── prometheus.yml            # Prometheus config for metrics
├── README.md                 # README file
└── docker-compose.yml        # Docker Compose file to run the platform
```

---

## Features

- **IEP1: Climate Forecasting**  
  LSTM model predicts climate variables up to a user-specified date based on historical data. It uses the previous 5 years to predict the next month. If the date entered by the user is a past date within the dataset, the output will be the values in the dataset itself. 

- **IEP2: Drought Risk Assessment**  
 RandomForest model and then threshold-based labeling, useing PDSI as a label, to evaluate drought severity.

- **IEP3: Water Availability Management**  
  RandomForest model and then threshold-based labeling to guide water resource planning.

- **EEP (External Ensemble Pipeline)**  
  Coordinates all IEPs, provides unified access through API and UI, and is fully containerized.

- **Monitoring with Prometheus**  
  Tracks real-time model activity and API health.

---

## 📊 Datasets on Google Drive

- 🔗 **Dataset (Google Drive):** [Access Cleaned Dataset]([https://drive.google.com/your-dataset-link](https://drive.google.com/drive/folders/19AYOqcLBMoMgqBZworMN_wgOyXXZHXM6?usp=drive_link))

---

## 💻 Local Setup

1. **Clone the Repository**

```bash
git clone git@github.com:issaramro/ClimateSmartAI.git
cd ClimateSmartAI
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
http://localhost:8004  # UI
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
- **Azure** for cloud deployment

---

## 📘 Future Work

- Integrate MLflow for live experiment tracking  
- Add a chatbot fine-tuned for this field

---

## 👨‍🎓 Academic Notes

This project was developed for educational purposes in the AI in Industry course. It meets all MLOps, testing, and deployment requirements and demonstrates practical business value in climate and water management.

---

## 🙋‍♂️ Contributors

Alaa Aoun & Issar Amro
