# Data Streams Quest: Build, Deploy, and Monitor AI in the Wild

**Team Programming Assignment – Operationalizing AI Across the Lifecycle**

---

## 🎯 Objective
This assignment challenges your team to transform your **individual Kafka-based air quality prediction project** into a **production-ready MLOps system**. You will collaboratively design, implement, deploy, and monitor a **real-time prediction platform** using the UCI Air Quality dataset.

Unlike the case-study assignment (strategy + lifecycle canvases), this programming track emphasizes **end-to-end engineering**:  
- Model experimentation  
- Orchestration  
- Serving  
- Monitoring  
- System-level visualization  

By the end, your team will demonstrate how to **operationalize AI**: turning a prototype into a reproducible, monitorable, and maintainable production-style system.

---

## 📌 Phases & Tasks

### Phase 1 – Interim Report (Feedback Only)
- **Model Experimentation with MLflow**
  - Run and log at least three distinct models (e.g., Random Forest, XGBoost, ARIMA/LSTM).
  - Compare results and document decision process for selecting candidate production model.
- **System Architecture Design**
  - Develop a high-level architecture diagram including Kafka, MLflow, FastAPI, monitoring, and storage.
  - Show service connections within Docker Compose.
- **Team Plan**
  - Assign roles and responsibilities.
  - Provide a project timeline with milestones.

**Deliverables:**
- Interim report (1,000–1,500 words, draft acceptable).
- System architecture diagram.
- Initial GitHub repository with MLflow experiment code and setup documentation.

> Note: Phase 1 is feedback-only. Draft content may be reused in the final report.

---

### Phase 2 – Final Deliverables (Graded)

#### 1. Model Experimentation & Registry (20 pts)
- Track at least three models in MLflow with metrics, plots, and artifacts.
- Promote one model to Production in the MLflow registry.
- Provide comparative visualizations (plots, error distributions, learning curves).

#### 2. System Architecture & Deployment (20 pts)
- Containerize all services with Docker.
- Orchestrate with `docker-compose.yml` (Kafka/Redpanda, FastAPI, MLflow, Evidently).
- Provide clear **reproduction instructions**.

#### 3. Monitoring & Visualization (20 pts)
- Implement **Evidently** for drift detection.
- Generate at least one static HTML drift report (reference vs. current datasets).
- Provide **system-level visualizations**:
  - (a) High-level architecture diagram (data flow).  
  - (b) Docker Compose service diagram (container layout).  
- Submit pollutant **time-series plots** and **predicted vs actual comparisons**.

#### 4. Final Report & Documentation (20 pts)
- Length: **2,500–4,000 words total** (includes Phase 1).
- Must include:
  - Executive summary  
  - System architecture explanation  
  - Model experimentation results (with plots)  
  - Deployment details  
  - Monitoring results + interpretation  
  - Limitations and recommendations  
- Provide an **Operations Runbook**:
  - Setup  
  - Usage  
  - Troubleshooting  
  - Rollback guidance  

#### 5. Presentation (20 pts)
- Narrated demo recording (5–10 min) of running system.  
- Slides (max 8) with **visual emphasis** (diagrams, plots, dashboards).  
- In-class **Q&A (10 min)**.

---

## 📝 Evaluation Rubric (100/105 pts)

### Report (80/85 pts)
- Content & Depth (20 pts) – Completeness of experimentation, deployment, monitoring, analysis.  
- Structure & Organization (15 pts) – Clear flow, logical narrative, use of headings.  
- Code, Repo & Appendix (15 pts) – Reproducible GitHub repo, documentation, GenAI appendix.  
- Originality & Creativity (10 pts) – Unique design/monitoring decisions.  
- Visualizations & Results (10 pts) – Clear, professional plots/diagrams.  
- Professional Standards (10 pts) – Readability, formatting, technical writing quality.  
- **Bonus (5 pts)** – Advanced Ops Extensions (see below).

### Presentation (20 pts)
- Content Delivery (10 pts) – Clear, logical explanation.  
- Engagement & Q&A (5 pts) – Team collaboration, responsiveness.  
- Visuals & Demo Quality (5 pts) – Professional demo and slides.  

---

## ⚙️ Tools (Mandatory)
- **Apache Kafka (KRaft mode)** or **Redpanda CE** – Data ingestion/streaming  
- **MLflow** – Experiment tracking & model registry  
- **FastAPI** – Serving trained model as REST API  
- **Evidently** – Data/model drift monitoring  
- **Docker & Docker Compose** – Service containerization and orchestration  

---

## 📚 Deliverables Summary
- **Interim Report:** 1,000–1,500 words (draft).  
- **Final Report:** 2,500–4,000 words total (including interim).  
- **Visualizations:** MLflow plots, pollutant time-series, monitoring dashboards, system diagrams.  
- **GitHub Repository:** Reproducible code, Dockerfiles, docker-compose.yml, monitoring scripts, README.  
- **Presentation:** 5–10 min demo video + 8-slide deck + class Q&A.  
- **Appendices:** GenAI usage documentation; optional bonus deliverables.  

---

## 🔥 Bonus Deliverables (max +5 pts)
- **Prometheus + Grafana:** Real-time metrics dashboards.  
- **Multi-partition Kafka topics:** Scalability & throughput improvements.  
- **Schema Registry (Avro/Protobuf):** Structured, versioned message formats.  

---

## 🤖 GenAI Use Policy (Tier 4)
- Generative AI tools (e.g., ChatGPT, Copilot) are allowed for **code, documentation, brainstorming**.  
- You must **verify correctness, fairness, and originality**.  
- Document all GenAI use in an **Appendix**, including:  
  - Tool name/version  
  - Prompt(s) used  
  - Output(s) incorporated  
  - Modifications/verification process  

---

## 🛠️ Support
For infrastructure issues (Kafka, Docker Compose, FastAPI), contact the TAs.  
Switching to **Redpanda** is allowed if needed.