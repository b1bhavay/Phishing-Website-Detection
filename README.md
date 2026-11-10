# Phishing Website Detection Using Machine Learning

## 1. Project Overview

Phishing Website Detection is a machine learning-based cybersecurity project designed to identify whether a website is phishing or legitimate.

The system uses website features and a trained Machine Learning model to classify websites into two categories:

- **Phishing Website (-1)**
- **Legitimate Website (1)**

The project integrates Machine Learning, FastAPI, Docker, GitHub Actions and cloud deployment to build a production-style ML application.

## 2. Problem Statement

Phishing attacks are a major cybersecurity threat. Attackers create fraudulent websites to steal sensitive information such as passwords, banking credentials and personal details.

The objective of this project is to develop a machine learning-based system that can classify websites using their extracted features and help users identify potentially malicious websites.

## 3. Dataset Description

**Dataset:** Phishing Websites Dataset

**Source:** UCI Machine Learning Repository  
**Dataset ID:** 327  
**Link:** https://archive.ics.uci.edu/dataset/327/phishing+websites

Dataset details:

- Total records: 11,055
- Original input features: 30
- Selected input features: 28
- Target variable: `Result`
- Missing values: 0
- Feature values: -1, 0 and 1

Target encoding:

| Value | Meaning |
|---|---|
| -1 | Phishing Website |
| 1 | Legitimate Website |

Two features, `favicon` and `double_slash_redirecting`, were removed during feature selection.

## 4. Data Preprocessing

The following preprocessing steps were performed:

1. Dataset loading and inspection.
2. Checking missing values.
3. Correlation analysis.
4. Feature selection.
5. Removal of two selected features.
6. Train-test splitting using stratification.
7. Feature scaling using StandardScaler.

Dataset split:

| Dataset | Records |
|---|---:|
| Training | 8,844 |
| Testing | 2,211 |
| Total | 11,055 |

Final selected features: 28.

## 5. Machine Learning Algorithms

Three machine learning algorithms were evaluated:

- Logistic Regression
- Random Forest Classifier
- Decision Tree Classifier

### Model Performance

| Algorithm | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 92.94% | 92.49% | 95.04% | 93.75% |
| Random Forest | 95.79% | 94.87% | 97.73% | 96.28% |
| Decision Tree | 94.66% | 95.43% | 94.96% | 95.20% |

### Selected Model

**Random Forest Classifier**

Random Forest was selected because it achieved the highest accuracy and F1-score among the evaluated models.

Final F1-score: **96.28%**

## 6. MLflow Experiment Tracking

MLflow was integrated to track machine learning experiments.

The following activities were performed:

- Experiment creation.
- Tracking multiple algorithm runs.
- Logging model performance metrics.
- Comparing different machine learning algorithms.
- Selecting the best-performing model.

Experiment name: `Phishing-Website-Detection`

Tracking backend: SQLite.

## 7. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data processing |
| NumPy | Numerical operations |
| Scikit-learn | Machine Learning |
| MLflow | Experiment tracking |
| FastAPI | Backend API |
| Pydantic | Input validation |
| HTML, CSS, JavaScript | Frontend |
| Pytest | Automated testing |
| Docker | Containerization |
| GitHub Actions | Continuous Integration |
| Render | Cloud deployment |

## 8. Application Features

The application provides:

- A cybersecurity-themed web interface.
- Input of 28 website feature values.
- Phishing and legitimate website classification.
- Machine learning-based predictions.
- Input validation using Pydantic.
- Security recommendations after prediction.
- Health monitoring endpoint.
- Prediction input logging for drift monitoring.

**Note:** The current application accepts extracted feature values rather than directly scanning a website URL.

## 9. API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Web application homepage |
| GET | `/health` | API health status |
| POST | `/predict` | Website classification |

### Prediction Input Format

The API accepts a JSON object containing exactly 28 feature values.

Example:

```json
{
  "values": [-1, 1, 1, 1, -1, 0, 0, 1, 1, 1, -1, -1, 0, -1, 1, 1, 0, 1, 1, 1, 1, 1, 1, -1, -1, 1, 0, 1]
}
```

Allowed values: -1, 0 and 1.

Invalid inputs are rejected with HTTP 422.

## 10. Docker and Cloud Deployment

The application was containerized using Docker.

Base image: `python:3.11-slim`

The Dockerfile installs serving dependencies and runs the FastAPI application using Uvicorn.

Cloud platform: Render

**Live Application:**

https://phishing-website-detection-g2rx.onrender.com

Health endpoint:

https://phishing-website-detection-g2rx.onrender.com/health

The deployed application was tested successfully with a phishing prediction.

## 11. Continuous Integration

GitHub Actions was configured to automate testing.

The workflow runs when code is pushed to the main branch or a pull request is created.

CI workflow activities:

1. Checkout repository.
2. Set up Python 3.11.
3. Install project dependencies.
4. Execute automated tests using Pytest.

The GitHub Actions workflow completed successfully with a green status.

## 12. Automated Testing

Four automated tests were implemented:

1. Health endpoint test.
2. Homepage response test.
3. Valid prediction test.
4. Invalid input validation test.

The tests verify API functionality and input validation.

## 13. Prediction Logging

The application records prediction information in a JSON Lines file named `prediction_logs.jsonl`.

Logged information includes:

- Prediction timestamp.
- Input website features.
- Predicted class.
- Prediction result.

These logs can support future data drift monitoring.

**Storage note:** The current Render Free instance uses local filesystem logging, which is not persistent across instance restarts.

## 14. Project Structure

```text
Phishing-Website-Detection/
│
├── data/
│   └── Phishing_Websites_Data.csv
│
├── model/
│   ├── model.joblib
│   ├── scaler.joblib
│   └── features.joblib
│
├── static/
│   └── index.html
│
├── tests/
│   └── test_app.py
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── app.py
├── Dockerfile
├── requirements.txt
├── requirements-serve.txt
└── .gitignore
```

## 15. Team Members

| Name | Roll Number |
|---|---|
| Bhavay Chandna | 24CSU034 |
| Aditya Mehta | 24CSU009 |
| Anish Sehrawat | 24CSU017 |

## 16. Future Enhancements

- Direct website URL scanning.
- Real-time feature extraction.
- Advanced phishing detection models.
- Persistent prediction log storage.
- Automated data drift detection.
- Model retraining and monitoring.
- Improved dashboard and visualization.

## 17. Conclusion

The project demonstrates how Machine Learning can be integrated with modern software engineering and deployment technologies to build a practical cybersecurity application.

The Random Forest model achieved an F1-score of 96.28%. FastAPI provides prediction services, Docker supports containerization, GitHub Actions automates testing, and Render provides cloud deployment.

The project provides a foundation for developing more advanced phishing detection systems.
