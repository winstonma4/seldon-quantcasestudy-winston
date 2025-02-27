#!/bin/bash
# Setup script for Stock Prediction Microservices

# Create project directories
echo "Creating project directory structure..."
mkdir -p data models
mkdir -p returns-service prediction-service display-service

# Copy files to returns-service
echo "Setting up returns-service..."
cp docker-compose.yml .
cp returns-service-dockerfile returns-service/Dockerfile
cp returns-service-requirements returns-service/requirements.txt
cp returns-service-app.py returns-service/app.py
echo "NOTE: You need to place intraday_curve_predictor.py in the returns-service directory"

# Copy files to prediction-service
echo "Setting up prediction-service..."
cp prediction-service-dockerfile prediction-service/Dockerfile
cp prediction-service-requirements prediction-service/requirements.txt
cp prediction-service-app.py prediction-service/app.py
echo "NOTE: You need to place intraday_curve_predictor.py in the prediction-service directory"

# Copy files to display-service
echo "Setting up display-service..."
cp display-service-dockerfile display-service/Dockerfile
cp display-service-requirements display-service/requirements.txt
cp display-service-app.py display-service/app.py

# Reminder for data file
echo "============================================================"
echo "IMPORTANT: Please place your data file in the data directory:"
echo "  data/quantum_price_data_winston.xlsx"
echo ""
echo "You also need to place the full intraday_curve_predictor.py file in both:"
echo "  - returns-service/intraday_curve_predictor.py"
echo "  - prediction-service/intraday_curve_predictor.py"
echo "============================================================"

echo "Setup completed!"
echo "Run 'docker-compose up --build' to start the services"
