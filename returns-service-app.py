import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from intraday_curve_predictor import IntradayCurvePredictor
import logging
import traceback
import json
from datetime import datetime
import os
import threading

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global predictor instance
predictor = None

class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types"""
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                           np.int16, np.int32, np.int64, np.uint8,
                           np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return json.JSONEncoder.default(self, obj)

def initialize_service(file_path='data/quantum_price_data_winston.xlsx'):
    """Initialize the predictor with data file (function that can be called at startup)"""
    global predictor
    
    try:
        logger.info(f"Auto-initializing predictor with file: {file_path}")
        
        # Initialize predictor and parse data
        predictor = IntradayCurvePredictor()
        
        # Check if file exists
        if not os.path.exists(file_path):
            alternative_path = 'quantum_price_data_winston.xlsx'
            logger.warning(f"File {file_path} not found, trying {alternative_path}")
            if os.path.exists(alternative_path):
                file_path = alternative_path
            else:
                logger.error(f"Neither {file_path} nor {alternative_path} exists")
                return False
        
        df = predictor.parse_excel_to_df(file_path)
        
        # Check stationarity to generate returns
        stationarity_results = predictor.check_returns_stationarity(df)
        
        logger.info(f"Predictor initialized successfully with {file_path}")
        logger.info(f"Available tickers: {predictor.tickers}")
        logger.info(f"Data points: {len(df)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error auto-initializing predictor: {str(e)}")
        logger.error(traceback.format_exc())
        return False

# Initialize the service in a background thread to avoid blocking the app startup
def async_initialize():
    threading.Thread(target=initialize_service).start()

# Initialize when app starts
# For production, use the below approach which is more reliable
with app.app_context():
    # Try to determine the right file path
    potential_paths = [
        'data/quantum_price_data_winston.xlsx',  # Docker container path
        'quantum_price_data_winston.xlsx',       # Local development path
        '/app/data/quantum_price_data_winston.xlsx'  # Alternative container path
    ]
    
    initialized = False
    for path in potential_paths:
        if os.path.exists(path):
            logger.info(f"Found data file at {path}")
            initialized = initialize_service(path)
            if initialized:
                break
    
    if not initialized:
        logger.warning("Could not auto-initialize service. Will try async initialization.")
        async_initialize()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    global predictor
    
    is_initialized = predictor is not None
    
    return jsonify({
        "status": "healthy", 
        "service": "returns-service",
        "initialized": is_initialized
    })

@app.route('/initialize', methods=['POST'])
def initialize_endpoint():
    """Initialize the predictor with data file (API endpoint)"""
    global predictor
    
    try:
        data = request.json
        file_path = data.get('file_path', 'quantum_price_data_winston.xlsx')
        
        logger.info(f"Initializing predictor with file: {file_path}")
        
        success = initialize_service(file_path)
        
        if success:
            return jsonify({
                "status": "success",
                "message": f"Predictor initialized with {file_path}",
                "tickers": predictor.tickers,
                "data_points": len(predictor.parse_excel_to_df(file_path))
            })
        else:
            return jsonify({
                "status": "error",
                "message": f"Failed to initialize predictor with {file_path}"
            }), 500
        
    except Exception as e:
        logger.error(f"Error initializing predictor: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error", 
            "message": str(e)
        }), 500

@app.route('/returns', methods=['GET'])
def get_returns():
    """Get returns for requested tickers"""
    global predictor
    
    if predictor is None:
        return jsonify({
            "status": "error",
            "message": "Predictor not initialized. Call /initialize first."
        }), 400
    
    try:
        # Get requested tickers from query parameters
        tickers = request.args.get('tickers')
        if tickers:
            requested_tickers = tickers.split(',')
        else:
            # If no tickers specified, return all
            requested_tickers = predictor.tickers
            
        # Filter to only available tickers
        available_tickers = [t for t in requested_tickers if t in predictor.returns]
        
        # Prepare returns data
        returns_data = {}
        for ticker in available_tickers:
            # Convert the pandas Series to a dictionary with ISO formatted dates
            series = predictor.returns[ticker]
            returns_dict = {index.isoformat(): value for index, value in series.items()}
            returns_data[ticker] = returns_dict
            
        return jsonify({
            "status": "success",
            "requested_tickers": requested_tickers,
            "available_tickers": available_tickers,
            "returns": returns_data
        })
        
    except Exception as e:
        logger.error(f"Error getting returns: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error", 
            "message": str(e)
        }), 500

@app.route('/augmented-features', methods=['POST'])
def get_augmented_features():
    """Generate augmented features for raw data"""
    global predictor
    
    if predictor is None:
        return jsonify({
            "status": "error",
            "message": "Predictor not initialized. Call /initialize first."
        }), 400
    
    try:
        # Get raw data from request
        data = request.json
        
        if 'df' not in data:
            return jsonify({
                "status": "error",
                "message": "No dataframe provided in request"
            }), 400
            
        # Convert JSON to DataFrame
        try:
            # Recreate MultiIndex DataFrame
            df_dict = data['df']
            reconstructed_df = pd.DataFrame()
            
            for ticker in df_dict:
                for field in df_dict[ticker]:
                    # Convert string dates back to datetime
                    dates = [datetime.fromisoformat(d) for d in df_dict[ticker][field]['index']]
                    values = df_dict[ticker][field]['values']
                    
                    # Create series
                    series = pd.Series(values, index=dates)
                    
                    # Add to DataFrame with MultiIndex
                    reconstructed_df[(ticker, field)] = series
                    
            logger.info(f"Successfully reconstructed DataFrame with shape: {reconstructed_df.shape}")
            
            # Augment features
            augmented_df = predictor.augment_features(reconstructed_df)
            
            # Convert augmented DataFrame to dictionary for JSON serialization
            result_dict = {}
            for (ticker, field) in augmented_df.columns:
                if ticker not in result_dict:
                    result_dict[ticker] = {}
                
                series = augmented_df[(ticker, field)]
                result_dict[ticker][field] = {
                    'index': [idx.isoformat() for idx in series.index],
                    'values': series.values.tolist()
                }
                
            return jsonify({
                "status": "success",
                "augmented_features": result_dict
            })
            
        except Exception as e:
            logger.error(f"Error converting JSON to DataFrame: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                "status": "error", 
                "message": f"Error processing dataframe: {str(e)}"
            }), 400
            
    except Exception as e:
        logger.error(f"Error getting augmented features: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error", 
            "message": str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)