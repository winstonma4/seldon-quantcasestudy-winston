# Prediction Microservices

This project provides a microservices-based architecture for stock returns prediction using the IntradayCurvePredictor class.

The system consists of three microservices:

1. **Returns Service**: Handles data loading and provides stock returns data
2. **Prediction Service**: Creates prediction models and generates forecasts
3. **Display Service**: Visualizes and displays prediction results

## Directory Structure

```
.
├── data/                          # Shared data directory
│   └── quantum_price_data_winston.xlsx
├── models/                        # Directory for saving trained models
├── returns-service/               # Returns Service
│   ├── app.py
│   ├── Dockerfile
│   ├── intraday_curve_predictor.py
│   └── requirements.txt
├── prediction-service/            # Prediction Service
│   ├── app.py
│   ├── Dockerfile
│   ├── intraday_curve_predictor.py
│   └── requirements.txt
├── display-service/               # Display Service
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
├── docker-compose.yml             # Docker Compose configuration
└── README.md                      # This file
```

## Prerequisites

- Docker
- Docker Compose

## Setup Instructions

1. Clone this repository
2. Place your data file (`quantum_price_data_winston.xlsx`) in the `data` directory
3. Copy the full `intraday_curve_predictor.py` file to both the `returns-service` and `prediction-service` directories

## Running the Microservices

Start all services:

```bash
docker-compose up --build
```

To run in detached mode:

```bash
docker-compose up --build -d
```

## API Endpoints

### Returns Service (port 5001)

- `GET /health`: Health check endpoint
- `GET /returns?tickers=IONQ,QBTS`: Get returns for specified tickers

### Prediction Service (port 5002)

- `GET /health`: Health check endpoint
- `POST /predict`: Create prediction using XGBoost
  ```
{
    'target_ticker': "QSI",
    'model_type': "xgb",      # can be arima, var, or xgb
    'training_days': 30,             
    'forward_periods': 27,           
    'n_features': 10,                
    'n_corr_stocks': 2,              
    'cross_validation': False,       
    'use_pca': False,        # only for VAR
    'pca_variance': 0.95,    # only for VAR
    'df_data': df_to_json(df_recent),
}
  ```
- `POST /save-model`: Save the current model
  ```
  {
    "model_path": "models/xgb_model.pkl"
  }
  ```

### Display Service (port 5003)

- `GET /health`: Health check endpoint
- `GET /`: Home page displaying stored predictions
- `POST /display`: Display prediction results
  ```
  {
    "prediction_results": { ... }
  }
  ```
- `GET /predictions`: Get list of stored predictions
- `GET /predictions/<id>`: Get details of a specific prediction

## Important Notes

1. Make sure to copy the complete `intraday_curve_predictor.py` file to both the returns-service and prediction-service directories.
2. The services communicate over the `stock-prediction-network` Docker network.
3. The data directory is mounted as a volume, so you can add or modify data files without rebuilding containers.
4. Trained models are saved to the `models` directory which persists outside the container.

## Stopping the Services

```bash
docker-compose down
```

To also remove the network:

```bash
docker-compose down --volumes --remove-orphans
```
