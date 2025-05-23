# 🌿 ClimateSmartAI: AI-Powered Climate Forecasting and Agricultural Assessment for Lebanon  
*A containerized MLOps platform for climate prediction, drought risk assessment, and water management, deployed on Azure.*  

---

## 📌 Table of Contents  
1. [Motivation](#-motivation)
2. [Project Overview](#-project-overview)  
3. [Data Source & Preprocessing](#-data-source--preprocessing)  
4. [System Architecture](#system-architecture)
5. [Results](#results)
6. [Pipeline Workflow](#-pipeline-workflow)  
7. [AI Pipelines (IEPs)](#-ai-pipelines-ieps)  
8. [Deployment (Local & Azure)](#-deployment)  
9. [CI/CD Pipeline](#-cicd-pipeline) 
10. [Monitoring & Logging](#-monitoring--logging)  
11. [Testing & Validation](#-testing--validation)  
12. [Future Work](#-future-work)
13. [Presentation](#-project-presentation) 
14. [Demo Video](#-demo-video)  
15. [Contributors](#-contributors)  

---

## 👨‍🌾 Motivation
Lebanon has what many countries envy: diverse climates, fertile valleys, and a rich agriculture. But despite this natural advantage, farmers are struggling due to unpredictable weather, limited access to data, and poor planning tools. That’s why we created a tool focused on supporting farmers and agribusinesses. By providing them with timely, data-driven insights, we help them make better decisions about planting, irrigation, and crop planning—ultimately building resilience in their work and contributing to national food.
We believe that empowering farmers through technology is not just good business; it’s a necessity for Lebanon’s future.

---

## 🌍 Project Overview  
**ClimateSmartAI** is a modular, AI-driven platform designed to:  
- **Forecast climate variables** forecasts all features using LSTM models. It uses 5 years to predict 1 month.   
- **Assess drought risk** uses Random forest and threshold-based labeling. PDSI (Palmer Drought Severity Index) is used as the label.  
- **Predict water availability** uses Random forest and threshold-based labeling. 
- **Coordinate predictions** through a unified FastAPI-based EEP (External Ensemble Pipeline).  

**Key Technologies**:  
- **MLOps**: Docker, FastAPI, Prometheus.  
- **AI Models**: LSTM (forecasting), RandomForest (classification/regression).  
- **Cloud**: Azure Container Apps.  
- **Data**: Google Earth Engine (1975–2024, Baalbek/Hermel region). 

---

## 📊 Data Source & Preprocessing  
### Data Acquisition  
- **Source**: [Google Earth Engine TerraClimate Dataset](https://developers.google.com/earth-engine/datasets/catalog/IDAHO_EPSCOR_TERRACLIMATE).  
- **Region**: Baalbek/Hermel, Lebanon (1975–2024).  
- **Variables**: Temperature, precipitation, PDSI, soil moisture.

**Full list of collected variables:**
- `aet (mm)`: Actual evapotranspiration
- `def (mm)`: Climate water deficit
- `pdsi`: Palmer Drought Severity Index
- `pet (mm)`: Reference evapotranspiration
- `pr (mm)`: Precipitation accumulation
- `ro (mm)`: Runoff
- `soil (mm)`: Soil moisture
- `srad (W/m²)`: Surface shortwave radiation
- `swe (mm)`: Snow water equivalent
- `tmmn (°C)`: Minimum temperature
- `tmmx (°C)`: Maximum temperature
- `vap (kPa)`: Vapor pressure
- `vpd (kPa)`: Vapor pressure deficit
- `vs (m/s)`: Wind speed at 10m

### Preprocessing Steps 
1. **Choosing the region**: Choosing Bekaa/Hermel area.
2. **Cleaning**: Handle missing values and scaling features (according to the scales written on the dataset website).
3. **Splitting datasets**: Split the clean dataset into training and testing CSVs.  

🔗 **Check Cleaned Datasets**: [Google Drive](https://drive.google.com/drive/folders/19AYOqcLBMoMgqBZworMN_wgOyXXZHXM6)  

---

## System Architecture  
```plaintext
project_ai/  
├── IEP1_forecasting/          # LSTM model (PyTorch)  
├── IEP2_drought_assessment/   # RandomForest (scikit-learn)  
├── IEP3_water_availability/   # RandomForest model  
├── EEP_interface/             # FastAPI orchestrator  
├── prometheus.yml             # Metrics config  
└── docker-compose.yml         # Multi-container deployment
```
---
## ✔️ Results

After running the training and testing scripts, the generated validation and testing plots can be found in the following directories:

- Validation plots: `train_val_test/train_val_test1/validation_plots/`
- Testing plots: `train_val_test/train_val_test1/testing_plots/`

This is to assess model performance.

Note: The paths mentioned above are used to check the validation and testing plots of IEP1. To check those of IEP2 or IEP3, go to `train_val_test2` or `train_val_test3`, respectively.

---
## 👨‍💻 Pipeline Workflow
1. **User Input:** Date via EEP API.
2. **IEP Execution:** Parallel model inference.
3. **Ensemble Output:** JSON response with forecasts.

---

## 🤖 AI Pipelines (IEPs)

### 1. IEP1: Climate Forecasting
- **Model:** LSTM.
- **Input:** a date.
- **Output:** If the user enters a past date, it outputs what's in the dataset. Otherwise, it outputs forecasted values of features.

### 2. IEP2: Drought Risk Assessment
- **Model:** RandomForest + PDSI thresholds.
- **Input:** a vector of features.
- **Output:** Extreme, Severe, Moderate, or no drought.

### 3. IEP3: Water Availability
- **Model:** RandomForest and threshold classification (using soil moisture, precipitation, actual and reference evapotranspiration, and snow water equivalent).
- **Input:** a vector of features.
- **Output:** No, mild, or severe irrigation needed.

---

## ☁️ Deployment 

![Interface](interface.png)

### Local Setup
```bash
git clone git@github.com:issaramro/ClimateSmartAI.git  
cd ClimateSmartAI  
docker-compose build && docker-compose up
```

-Access UI: ```http://localhost:8004```

-API Docs: ```http://localhost:8004/docs```

---
### Azure Deployment
1. **Push Images to Docker Hub:**

```bash
docker tag iep1 issaramro/iep1  
docker push issaramro/iep1

docker tag iep2 issaramro/iep2  
docker push issaramro/iep2

docker tag iep3 issaramro/iep3  
docker push issaramro/iep3

docker tag eep issaramro/eep  
docker push issaramro/eep
```

2. **Azure Setup:**
- Create Container App in Azure Portal (we used Azure Container Apps).
- Configure ingress (modify links of the requests inside the eep app.py to match the Azure internal links of the ieps, since the eep sends requests to the ieps on the running servers).
- Attach Docker Hub images.

🔗 Deployment Paused (Due to Costs): See [Demo Video](https://drive.google.com/file/d/18QmXfkfeDHtxy86cNzBuoitY77JOwpkj/view) for live walkthrough.

---

## 🔄 CI/CD Pipeline

We implemented Continuous Integration and Continuous Deployment (CI/CD) using **GitHub Actions**.

- Automated testing is triggered on every pull request and push.
- Docker images are automatically built and pushed to Docker Hub upon merging to `main`.
- Azure Container Apps are updated with the new images.
- Automated unit testing for each IEP.

---

## 📉 Monitoring & Logging
- Prometheus Dashboard: ```http://localhost:9090```
- Tracks: API latency, model inference time, error rates.
---

## 🧪 Testing & Validation
### Test Cases
- Unit Tests: pytest for each IEP.
---

## 🚀 Future Work
- MLflow Integration: Experiment tracking.
- Fine-tuned Chatbot: Answer climate/agriculture queries.
- Real-time Data: Satellite feed integration.
---

## 🎥 Project Presentation

You can access our slides here:

🔗 [View Presentation](https://docs.google.com/presentation/d/1D179VhgXpdXSF_OhqeMbrBWd0Zb-eSMWYt5yXKLWuLg/edit?usp=sharing)

The presentation covers:
- Project motivation and objectives
- System architecture and workflow
- AI models and performance metrics
- Deployment and future improvements
---

## 👥 Contributors
Alaa Aoun & Issar Amro 

Course: AI in Industry, [AUB].
