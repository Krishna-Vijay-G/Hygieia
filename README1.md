# Hygieia - AI-Powered Medical Diagnostic Platform

## Authors

<a href="https://github.com/Krishna-Vijay-G"><img src="https://github.com/Krishna-Vijay-G.png" width="40" height="40" style="border-radius: 50%;" alt="Krishna Vijay G"></a> Krishna Vijay G | <a href="https://github.com/Neyanta-Rai"><img src="https://github.com/Neyanta-Rai.png" width="40" height="40" style="border-radius: 50%;" alt="Neyanta Rai"></a> Neyanta Rai | <a href="https://github.com/Rajesh-2222"><img src="https://github.com/Rajesh-2222.png" width="40" height="40" style="border-radius: 50%;" alt="Rajesh M"></a> Rajesh M | <a href="https://github.com/raavi-12"><img src="https://github.com/raavi-12.png" width="40" height="40" style="border-radius: 50%;" alt="Raavi Rishika Chowdary"></a> Raavi Rishika Chowdary

## Overview

Hygieia is a comprehensive AI-powered medical diagnostic platform designed to assist healthcare professionals and patients with early disease detection and risk assessment. The platform integrates multiple machine learning models for diagnosing breast cancer, predicting diabetes risk, assessing heart disease probability, and analyzing skin lesions, along with an AI-powered chat assistant named Dr. Hygieia.

## Features

- **Breast Cancer Diagnosis**: Diagnostic model for breast cancer detection using Wisconsin dataset
- **Diabetes Risk Prediction**: Predictive model for early-stage diabetes risk assessment
- **Heart Disease Risk Assessment**: Predictive model for cardiovascular disease risk
- **Skin Lesion Diagnosis**: Deep learning model for dermatological analysis
- **AI Chat Assistant**: Dr. Hygieia - conversational AI for medical guidance
- **User Management**: Authentication and user profile management
- **File Upload**: Secure medical image and document upload capabilities
- **Blockchain Integration**: Secure data storage and verification
- **Responsive Web Interface**: Modern, mobile-friendly frontend

## Tech Stack

### Backend
- **Python 3.11+**
- **Flask** - Web framework
- **SQLAlchemy** - ORM for database operations
- **JWT** - Authentication tokens
- **OpenAI GPT-4o-mini** - AI chat functionality
- **Scikit-learn** - Machine learning models
- **TensorFlow/PyTorch** - Deep learning models

### Frontend
- **Next.js 14** - React framework
- **React** - UI library
- **Tailwind CSS** - Styling
- **TypeScript** - Type safety

### Database
- **PostgreSQL** (via Supabase)
- **SQLite** (local development fallback)

### Infrastructure
- **Supabase** - Backend-as-a-Service (Database, Auth, Storage)
- **DuckDNS** - Dynamic DNS for deployment

## Installation & Setup

### Prerequisites
- Python 3.11 or higher
- Node.js 18 or higher
- Git
- PostgreSQL database (or Supabase account)

### 1. Clone the Repository
```bash
git clone <repository-url>
cd hygieia
```

### 2. Backend Setup

#### Create Virtual Environment
```bash
python -m venv venv-hygieia
# On Windows:
venv-hygieia\Scripts\activate
# On macOS/Linux:
source venv-hygieia/bin/activate
```

#### Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

#### Configure Environment Variables
```bash
cp ../.env.example ../.env
# Edit .env with your actual values
```

#### Train Machine Learning Models
The project includes pre-trained models, but you can also train them from scratch:

```bash
# Navigate to backend models directory
cd backend/models

# Train Heart Disease Model
cd "Heart Risk Predictive Model"
python train-heart-prediction.ipynb  # Run the Jupyter notebook
# or
python heart_prediction_benchmarker.py

# Train Breast Cancer Diagnostic Model
cd "../BC Diagnostic Model"
python train-breast-diagnosis.py

# Train Breast Cancer Predictive Model
cd "../BC Predictive Model"
python train-breast-prediction.py

# Train Diabetes Model
cd "../Diabetes Risk Predictive Model"
python train-diabetes-prediction.py

# Train Skin Lesion Model
# Get the Google Derm Founadation Model from the ReadMe File URL
cd "../Skin Lesion Diagnostic Model"
python train_skin-diagnosis.py
```

**Note**: Training requires the respective datasets to be present in each model directory. The skin lesion model requires the HAM10000 dataset.

#### Run Backend
```bash
python run.py
# Server will start on http://localhost:5000
```

### 3. Frontend Setup

#### Install Dependencies
```bash
cd ../frontend
npm install
```

#### Configure Environment (if needed)
```bash
# Frontend uses Next.js public env vars
# Configure NEXT_PUBLIC_* variables in .env if required
```

#### Run Frontend
```bash
npm run dev
# Frontend will start on http://localhost:3000
```

### 4. Access the Application
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:5000/api
- **Network Access**: Use the IP shown in backend logs for network access

## Directory Structure

```
hygieia/
├── backend/                          # Python Flask Backend
│   ├── app/
│   │   ├── __init__.py              # Flask app initialization
│   │   ├── models.py                # Database models
│   │   └── routes/                  # API endpoints
│   │       ├── analysis.py          # Diagnostic analysis routes
│   │       ├── auth.py              # Authentication routes
│   │       ├── benchmark.py         # Model benchmarking
│   │       ├── blockchain.py        # Blockchain integration
│   │       ├── chat.py              # AI chat routes
│   │       ├── users.py             # User management
│   │       └── __pycache__/
│   ├── controllers/                 # Business logic controllers
│   │   ├── breast_diagnosis.py      # Breast cancer diagnosis logic
│   │   ├── breast_prediction.py     # Breast cancer prediction logic
│   │   ├── diabetes_prediction.py   # Diabetes prediction logic
│   │   ├── heart_prediction.py      # Heart disease prediction logic
│   │   ├── skin_calibration.py      # Skin model calibration
│   │   └── skin_diagnosis.py        # Skin lesion diagnosis logic
│   ├── models/                      # ML Models and Data
│   │   ├── Model_Metadata.txt       # Model metadata
│   │   ├── model_metadata_manager.py # Metadata management
│   │   ├── BC Diagnostic Model/     # Breast Cancer Diagnosis
│   │   │   ├── breast-diagnosis.joblib          # Trained model
│   │   │   ├── breast_diagnosis_benchmarker.py  # Benchmarking script
│   │   │   └── Wisconsin Diagnosis Dataset - UCI.csv # Training data
│   │   ├── BC Predictive Model/     # Breast Cancer Prediction
│   │   │   ├── breast-prediction.joblib         # Trained model
│   │   │   ├── breast_prediction_benchmarker.py # Benchmarking script
│   │   │   └── BCSC Prediction Factors Dataset - BCSC.csv
│   │   ├── Diabetes Risk Predictive Model/
│   │   │   ├── diabetes-prediction.joblib       # Trained model
│   │   │   ├── diabetes_prediction_benchmarker.py
│   │   │   └── Early Stage Diabetes Risk Prediction - UCI.csv
│   │   ├── Heart Risk Predictive Model/
│   │   │   ├── heart-prediction.joblib          # Trained model
│   │   │   ├── heart_prediction_benchmarker.py
│   │   │   └── Heart Disease Prediction Dataset - Kaggle.csv
│   │   └── Skin Lesion Diagnostic Model/
│   │       ├── fingerprint.pb                   # TensorFlow model
│   │       ├── saved_model.pb                   # TensorFlow model
│   │       ├── scin_dataset_precomputed_embeddings.npz
│   │       ├── skin-diagnosis.joblib            # Scikit-learn wrapper
│   │       ├── skin_diagnosis_benchmarker.py
│   │       ├── HAM10000/                        # Skin lesion dataset
│   │       │   ├── HAM10000_metadata.csv
│   │       │   ├── selected_samples_700.csv
│   │       │   └── images/                      # Skin lesion images
│   │       └── variables/                       # TensorFlow variables
│   ├── config.py                   # Application configuration
│   ├── model_bridge.py             # Model loading utilities
│   ├── requirements.txt            # Python dependencies
│   └── run.py                      # Application entry point
├── frontend/                        # Next.js Frontend
│   ├── public/
│   │   ├── data/
│   │   │   └── team.json           # Team information
│   │   └── llm_integration/        # LLM integration assets
│   ├── src/
│   │   └── app/                    # Next.js app directory
│   ├── package.json                # Node dependencies
│   ├── next.config.js              # Next.js configuration
│   ├── tailwind.config.ts          # Tailwind configuration
│   └── tsconfig.json               # TypeScript configuration
├── database/                        # Database Files
│   └── schema.sql                  # Database schema
├── tools/                          # Utility Tools
│   └── inspect_joblib.py           # Model inspection utility
├── uploads/                        # User Uploads
│   ├── avatars/                    # User profile pictures
│   └── avatars_archive/            # Archived avatars
├── venv-hygieia/                   # Python Virtual Environment
├── .env                            # Environment Variables
├── .env.example                    # Environment Template
├── .gitignore                      # Git Ignore Rules
└── README.md                       # This file
```

## Models Overview

### 1. Breast Cancer Diagnostic Model
**Location**: `backend/models/BC Diagnostic Model/`
**Purpose**: Diagnoses breast cancer from biopsy data
**Files**:
- `breast-diagnosis.joblib` - Trained scikit-learn model
- `breast_diagnosis_benchmarker.py` - Performance evaluation script
- `Wisconsin Diagnosis Dataset - UCI.csv` - Training dataset

### 2. Breast Cancer Predictive Model
**Location**: `backend/models/BC Predictive Model/`
**Purpose**: Predicts breast cancer risk factors
**Files**:
- `breast-prediction.joblib` - Trained predictive model
- `breast_prediction_benchmarker.py` - Benchmarking script
- `BCSC Prediction Factors Dataset - BCSC.csv` - Training data

### 3. Diabetes Risk Predictive Model
**Location**: `backend/models/Diabetes Risk Predictive Model/`
**Purpose**: Assesses diabetes risk from patient data
**Files**:
- `diabetes-prediction.joblib` - Risk prediction model
- `diabetes_prediction_benchmarker.py` - Evaluation script
- `Early Stage Diabetes Risk Prediction - UCI.csv` - Dataset

### 4. Heart Disease Risk Predictive Model
**Location**: `backend/models/Heart Risk Predictive Model/`
**Purpose**: Predicts cardiovascular disease risk
**Files**:
- `heart-prediction.joblib` - Heart disease prediction model
- `heart_prediction_benchmarker.py` - Benchmarking tool
- `Heart Disease Prediction Dataset - Kaggle.csv` - Training data

### 5. Skin Lesion Diagnostic Model
**Location**: `backend/models/Skin Lesion Diagnostic Model/`
**Purpose**: Analyzes dermatological images for skin cancer detection
**Files**:
- `saved_model.pb`, `fingerprint.pb` - TensorFlow model files
- `skin-diagnosis.joblib` - Model wrapper
- `skin_diagnosis_benchmarker.py` - Performance testing
- `HAM10000/` - HAM10000 skin lesion dataset
- `variables/` - TensorFlow model variables

## Usage

### Running the Application
1. Start the backend server: `python backend/run.py`
2. Start the frontend: `cd frontend && npm run dev`
3. Access the application at http://localhost:3000

### API Endpoints
- `POST /api/auth/login` - User authentication
- `POST /api/analysis/breast-diagnosis` - Breast cancer diagnosis
- `POST /api/analysis/diabetes-prediction` - Diabetes risk prediction
- `POST /api/chat/message` - AI chat with Dr. Hygieia

### Model Benchmarking
Run benchmark scripts in each model directory:
```bash
python models/Skin_Disease_Model/benchmark_dermatology_model.py --samples 3 --dataset-dir HAM10000
```

## Development

### Running Tests
```bash
# Backend tests
cd backend
python -m pytest

# Frontend tests
cd frontend
npm test
```

### Code Formatting
```bash
# Python
black backend/
isort backend/

# JavaScript/TypeScript
cd frontend
npm run lint
```

## Deployment

### Local Network Deployment
1. Update `.env` with your network IP
2. Run backend and frontend
3. Access via network IP (e.g., `http://192.168.1.100:3000`)

### Production Deployment
1. Set up production database (Supabase/PostgreSQL)
2. Configure production environment variables
3. Deploy backend and frontend to hosting services
4. Set up domain with DuckDNS or similar

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Security Notice

This application is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions.

## License

[Specify your license here]

## Support

For questions or support, please [contact information or issue tracker].

---

**Disclaimer**: This software is provided "as is" without warranty of any kind. The developers are not liable for any damages or consequences arising from the use of this software.