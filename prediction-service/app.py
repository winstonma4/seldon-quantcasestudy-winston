import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import logging
import traceback
import json
import pickle
from datetime import datetime
import requests
import os
import threading
from intraday_curve_predictor import IntradayCurvePredictor

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Custom JSON encoder that handles numpy types and MultiIndex
class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types and pandas structures"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, pd.MultiIndex):
            return [list(i) for i in obj.values]
        return super(NumpyEncoder, self).default(obj)

# Initialize Flask app
app = Flask(__name__)
app.json_encoder = NumpyEncoder  # Use our custom JSON encoder

# Global predictor instance
predictor = None

def initialize_service(file_path='data/quantum_price_data_winston.xlsx'):
    """Initialize the predictor with data file (function that can be called at startup)"""
    global predictor
    
    try:
        logger.info(f"Auto-initializing predictor with file: {file_path}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            # Try alternative paths
            potential_paths = [
                'quantum_price_data_winston.xlsx',       # Local development path
                '/app/data/quantum_price_data_winston.xlsx',  # Alternative container path
                '../data/quantum_price_data_winston.xlsx'  # Relative path option
            ]
            
            found_path = None
            for path in potential_paths:
                if os.path.exists(path):
                    logger.info(f"Found data file at {path}")
                    found_path = path
                    break
                    
            if found_path:
                file_path = found_path
            else:
                logger.error(f"Could not find data file in any expected location")
                return False
        
        # Initialize predictor
        predictor = IntradayCurvePredictor()
        
        # Load data
        logger.info("Loading data...")
        df = predictor.parse_excel_to_df(file_path)
        
        # Check if returns were calculated during parsing
        if predictor.returns is None or len(predictor.returns) == 0:
            logger.info("Returns were not calculated during parsing, calculating now...")
            predictor.check_returns_stationarity(df)
        
        # Verify returns were calculated
        if predictor.returns is None or len(predictor.returns) == 0:
            logger.error("Failed to calculate returns locally!")
            return False
        
        logger.info(f"Successfully calculated returns for {len(predictor.returns)} tickers")
        return True
        
    except Exception as e:
        logger.error(f"Error auto-initializing predictor: {str(e)}")
        logger.error(traceback.format_exc())
        return False

# Initialize the service in a background thread to avoid blocking the app startup
def async_initialize():
    def _init_task():
        logger.info("Starting asynchronous initialization...")
        initialize_service()
        
    threading.Thread(target=_init_task).start()

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
                logger.info("Prediction service initialization complete!")
                break
    
    if not initialized:
        logger.warning("Could not auto-initialize service. Will try async initialization.")
        async_initialize()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    global predictor
    
    is_initialized = predictor is not None and hasattr(predictor, 'returns') and predictor.returns is not None
    
    return jsonify({
        "status": "healthy", 
        "service": "prediction-service",
        "initialized": is_initialized
    })

@app.route('/initialize', methods=['POST'])
def initialize_endpoint():
    """Initialize the predictor with data file (API endpoint)"""
    global predictor
    
    try:
        data = request.json
        file_path = data.get('file_path', 'data/quantum_price_data_winston.xlsx')
        returns_service_url = data.get('returns_service_url', 'http://returns-service:5001')
        
        logger.info(f"Initializing predictor with file: {file_path}")
        logger.info(f"Returns service URL: {returns_service_url}")
        
        # Initialize predictor on this thread
        success = initialize_service(file_path)
        
        if not success:
            return jsonify({
                "status": "error",
                "message": "Failed to initialize predictor locally"
            }), 500
        
        # Now try to initialize the returns service as well (but don't fail if it doesn't work)
        try:
            logger.info(f"Initializing returns service at {returns_service_url}...")
            response = requests.post(
                f"{returns_service_url}/initialize",
                json={"file_path": file_path},
                timeout=120
            )
            response.raise_for_status()
            logger.info("Successfully initialized returns service")
        except Exception as e:
            logger.warning(f"Failed to initialize returns service: {str(e)}")
            logger.warning("Continuing with locally calculated returns")
        
        return jsonify({
            "status": "success",
            "message": f"Predictor initialized with {file_path}",
            "tickers": predictor.tickers,
            "data_points": len(predictor.parse_excel_to_df(file_path)),
            "returns_tickers": list(predictor.returns.keys())
        })
        
    except Exception as e:
        logger.error(f"Error initializing predictor: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error", 
            "message": str(e)
        }), 500

# Add new endpoint for handling MultiIndex DataFrames
@app.route('/predict-with-df', methods=['POST'])
def predict_with_df():
    """Create prediction using a MultiIndex DataFrame directly"""
    global predictor
    
    if predictor is None:
        return jsonify({
            "status": "error",
            "message": "Predictor not initialized. Call /initialize first."
        }), 400
    
    try:
        data = request.json
        
        # Required parameters
        target_ticker = data.get('target_ticker')
        if not target_ticker:
            return jsonify({
                "status": "error",
                "message": "target_ticker is required"
            }), 400
            
        # Optional parameters with defaults
        model_type = data.get('model_type', 'xgb')  # New parameter for model selection (xgb, arima, var)
        training_days = data.get('training_days', 30)
        forward_periods = data.get('forward_periods', 27)
        n_features = data.get('n_features', 10)
        n_corr_stocks = data.get('n_corr_stocks', 2)
        cross_validation = data.get('cross_validation', False)
        display_service_url = data.get('display_service_url', 'http://display-service:5003')
        
        # Model-specific parameters
        arima_order = data.get('arima_order', (2, 0, 1))  # (p, d, q) for ARIMA
        var_use_pca = data.get('var_use_pca', False)      # Use PCA for VAR
        xgb_params = data.get('xgb_params', None)         # Custom XGBoost parameters
        
        logger.info(f"Prediction request for {target_ticker} using {model_type.upper()} model")
        logger.info(f"Parameters: training_days={training_days}, forward_periods={forward_periods}")
        logger.info(f"n_features={n_features}, n_corr_stocks={n_corr_stocks}, cross_validation={cross_validation}")
            
        # Check if DataFrame data is provided
        if 'df_data' not in data:
            return jsonify({
                "status": "error",
                "message": "df_data is required"
            }), 400
        
        df_data = data['df_data']
        
        # Check required format
        if 'index' not in df_data or 'columns' not in df_data or 'data' not in df_data:
            return jsonify({
                "status": "error",
                "message": "df_data must contain 'index', 'columns', and 'data' fields"
            }), 400
        
        try:
            # Reconstruct DataFrame
            # 1. Convert index from string to datetime
            index = [datetime.fromisoformat(d) for d in df_data['index']]
            
            # 2. Reconstruct MultiIndex columns
            if isinstance(df_data['columns'][0], list):
                # Already properly formatted as list of lists
                columns = pd.MultiIndex.from_tuples([tuple(c) for c in df_data['columns']])
            else:
                # Try to parse string representation
                try:
                    columns = []
                    for col_str in df_data['columns']:
                        # Expecting format like "('IONQ', 'close')"
                        col_str = col_str.strip("()' ")
                        parts = col_str.split("', '")
                        columns.append(tuple(p.strip("'") for p in parts))
                    columns = pd.MultiIndex.from_tuples(columns)
                except:
                    return jsonify({
                        "status": "error", 
                        "message": "Invalid column format. Expected MultiIndex columns as list of lists."
                    }), 400
            
            # 3. Create DataFrame with MultiIndex
            df = pd.DataFrame(df_data['data'], index=index, columns=columns)
            
            logger.info(f"Successfully reconstructed DataFrame with shape: {df.shape}")
            
            # Ensure returns are calculated if they're not already
            if predictor.returns is None or len(predictor.returns) == 0:
                logger.info("Returns were not calculated, calculating now...")
                predictor.check_returns_stationarity(df)
            
            # Check if target ticker is in returns
            if target_ticker not in predictor.returns:
                # Try to calculate returns from the provided DataFrame
                logger.info(f"Target ticker {target_ticker} not found in returns, trying to calculate from DataFrame...")
                try:
                    price_series = df[target_ticker]['close']
                    predictor.returns[target_ticker] = np.log(price_series / price_series.shift(1)).dropna()
                    logger.info(f"Successfully calculated returns for {target_ticker}")
                except Exception as e:
                    logger.error(f"Failed to calculate returns for {target_ticker}: {str(e)}")
                    return jsonify({
                        "status": "error",
                        "message": f"Target ticker {target_ticker} not found in returns and calculation failed"
                    }), 400
            
            # Run prediction based on model type
            logger.info(f"Running {model_type.upper()} prediction with cross_validation={cross_validation}")
            
            if model_type.lower() == 'arima':
                # Create ARIMA prediction
                prediction_results = predictor.create_arima_predictor(
                    returns=predictor.returns[target_ticker],
                    order=arima_order,
                    training_days=training_days,
                    step_size=forward_periods,
                    cross_validation=cross_validation
                )
                
            elif model_type.lower() == 'var':
                # Create VAR prediction
                prediction_results = predictor.create_var_predictor(
                    raw_df=df,
                    target_ticker=target_ticker,
                    training_days=training_days,
                    forward_periods=forward_periods,
                    n_features=n_features,
                    n_corr_stocks=n_corr_stocks,
                    cross_validation=cross_validation,
                    use_pca=var_use_pca
                )
                
            else:  # Default to XGBoost
                # Create XGBoost prediction
                prediction_results = predictor.create_xgb_predictor(
                    raw_df=df,
                    target_ticker=target_ticker,
                    training_days=training_days,
                    forward_periods=forward_periods,
                    n_features=n_features,
                    n_corr_stocks=n_corr_stocks,
                    cross_validation=cross_validation,
                    custom_xgb_params=xgb_params
                )
            
            # Process prediction results for JSON serialization
            processed_results = process_prediction_results(
                prediction_results,
                target_ticker,
                model_type,
                cross_validation
            )
            
            # Send results to the display service if URL is provided
            if display_service_url:
                try:
                    logger.info(f"Sending results to display service at {display_service_url}")
                    
                    # Convert to JSON string first to ensure serialization works
                    json_data = json.dumps({"prediction_results": processed_results}, cls=NumpyEncoder)
                    
                    # Then parse back to Python object for requests
                    response = requests.post(
                        f"{display_service_url}/display",
                        json=json.loads(json_data),
                        timeout=10
                    )
                    response.raise_for_status()
                    logger.info("Successfully sent results to display service")
                except Exception as e:
                    logger.error(f"Failed to send results to display service: {str(e)}")
                    # Continue anyway to return results directly
            
            return jsonify({
                "status": "success",
                "prediction_results": processed_results
            })
            
        except Exception as e:
            logger.error(f"Error reconstructing DataFrame: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                "status": "error",
                "message": f"Error reconstructing DataFrame: {str(e)}"
            }), 400
            
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error", 
            "message": str(e)
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Create prediction using feature data"""
    global predictor
    
    if predictor is None:
        return jsonify({
            "status": "error",
            "message": "Predictor not initialized. Call /initialize first."
        }), 400
    
    try:
        data = request.json
        
        # Required parameters
        target_ticker = data.get('target_ticker')
        if not target_ticker:
            return jsonify({
                "status": "error",
                "message": "target_ticker is required"
            }), 400
            
        # Optional parameters with defaults
        model_type = data.get('model_type', 'xgb')  # New parameter for model selection (xgb, arima, var)
        training_days = data.get('training_days', 30)
        forward_periods = data.get('forward_periods', 27)
        n_features = data.get('n_features', 10)
        n_corr_stocks = data.get('n_corr_stocks', 2)
        cross_validation = data.get('cross_validation', False)
        display_service_url = data.get('display_service_url', 'http://display-service:5003')
        
        # Model-specific parameters
        arima_order = data.get('arima_order', (2, 0, 1))  # (p, d, q) for ARIMA
        var_use_pca = data.get('var_use_pca', False)      # Use PCA for VAR
        xgb_params = data.get('xgb_params', None)         # Custom XGBoost parameters
        
        logger.info(f"Prediction request for {target_ticker} using {model_type.upper()} model")
        logger.info(f"Parameters: training_days={training_days}, forward_periods={forward_periods}")
        logger.info(f"n_features={n_features}, n_corr_stocks={n_corr_stocks}, cross_validation={cross_validation}")
        
        # DEBUG: Check returns dictionary
        if predictor.returns is None:
            logger.error("predictor.returns is None. Re-initializing returns...")
            predictor.check_returns_stationarity(predictor.parse_excel_to_df('data/quantum_price_data_winston.xlsx'))
        
        # DEBUG: Check if target ticker is in returns
        if target_ticker not in predictor.returns:
            logger.error(f"Target ticker {target_ticker} not found in returns. Available tickers: {list(predictor.returns.keys())}")
            return jsonify({
                "status": "error",
                "message": f"Target ticker {target_ticker} not found in returns"
            }), 400
            
        # Get or reconstruct raw DataFrame
        df = None
        if 'df' in data:
            # Process the provided DataFrame (existing implementation)
            try:
                df_dict = data['df']
                reconstructed_df = pd.DataFrame()
                
                for ticker in df_dict:
                    for field in df_dict[ticker]:
                        # Convert string dates back to datetime
                        dates = [datetime.fromisoformat(d) for d in df_dict[ticker][field]['index']]
                        values = df_dict[ticker][field]['values']
                        
                        # Create series and add to DataFrame
                        series = pd.Series(values, index=dates)
                        reconstructed_df[(ticker, field)] = series
                        
                logger.info(f"Successfully reconstructed DataFrame with shape: {reconstructed_df.shape}")
                df = reconstructed_df
                
            except Exception as e:
                logger.error(f"Error reconstructing DataFrame: {str(e)}")
                return jsonify({
                    "status": "error",
                    "message": f"Error reconstructing DataFrame: {str(e)}"
                }), 400
        else:
            # Use the one loaded during initialization
            logger.info("Using DataFrame from initialization")
            df = predictor.parse_excel_to_df('data/quantum_price_data_winston.xlsx')
        
        # Run prediction based on model type
        logger.info(f"Running {model_type.upper()} prediction with cross_validation={cross_validation}")
        
        if model_type.lower() == 'arima':
            # Create ARIMA prediction
            prediction_results = predictor.create_arima_predictor(
                returns=predictor.returns[target_ticker],
                order=arima_order,
                training_days=training_days,
                step_size=forward_periods,
                cross_validation=cross_validation
            )
            
        elif model_type.lower() == 'var':
            # Create VAR prediction
            prediction_results = predictor.create_var_predictor(
                raw_df=df,
                target_ticker=target_ticker,
                training_days=training_days,
                forward_periods=forward_periods,
                n_features=n_features,
                n_corr_stocks=n_corr_stocks,
                cross_validation=cross_validation,
                use_pca=var_use_pca
            )
            
        else:  # Default to XGBoost
            # Create XGBoost prediction
            prediction_results = predictor.create_xgb_predictor(
                raw_df=df,
                target_ticker=target_ticker,
                training_days=training_days,
                forward_periods=forward_periods,
                n_features=n_features,
                n_corr_stocks=n_corr_stocks,
                cross_validation=cross_validation,
                custom_xgb_params=xgb_params
            )
        
        # Process prediction results for JSON serialization
        processed_results = process_prediction_results(
            prediction_results,
            target_ticker,
            model_type,
            cross_validation
        )
        
        # Send results to the display service if URL is provided
        if display_service_url:
            try:
                logger.info(f"Sending results to display service at {display_service_url}")
                
                # Convert to JSON string first to ensure serialization works
                json_data = json.dumps({"prediction_results": processed_results}, cls=NumpyEncoder)
                
                # Then parse back to Python object for requests
                response = requests.post(
                    f"{display_service_url}/display",
                    json=json.loads(json_data),
                    timeout=10
                )
                response.raise_for_status()
                logger.info("Successfully sent results to display service")
            except Exception as e:
                logger.error(f"Failed to send results to display service: {str(e)}")
                # Continue anyway to return results directly
        
        return jsonify({
            "status": "success",
            "prediction_results": processed_results
        })
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error", 
            "message": str(e)
        }), 500

def convert_tuple_keys(obj):
    """Recursively converts tuple keys to strings in dictionaries"""
    if isinstance(obj, dict):
        return {str(k) if isinstance(k, tuple) else k: convert_tuple_keys(v) 
                for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_tuple_keys(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_tuple_keys(item) for item in obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy arrays to lists
    elif isinstance(obj, np.integer):
        return int(obj)      # Convert numpy integers to Python ints
    elif isinstance(obj, np.floating):
        return float(obj)    # Convert numpy floats to Python floats
    elif isinstance(obj, np.bool_):
        return bool(obj)     # Convert numpy booleans to Python bools
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif hasattr(obj, 'dtype') and pd.api.types.is_numpy_dtype(obj.dtype):
        return obj.item()    # Handle other numpy scalar types
    else:
        return obj

def process_prediction_results(prediction_results, target_ticker, model_type, cross_validation=False):
    """
    Process prediction results for JSON serialization.
    Handles all numpy types and nested structures with tuple keys.
    Updated to support different model types.
    """
    try:
        # Start with a clean dictionary
        processed = {
            "model_type": model_type,
            "target_ticker": target_ticker
        }
        
        # For cross-validation, include metrics and predictions
        if cross_validation:
            if "avg_metrics" in prediction_results:
                processed["metrics"] = {
                    "rmse": float(prediction_results["avg_metrics"]["mean_rmse"]),
                    "mae": float(prediction_results["avg_metrics"]["mean_mae"]),
                    "harmonic_rmse": float(prediction_results["harmonic_metrics"]["harmonic_rmse"]),
                    "harmonic_mae": float(prediction_results["harmonic_metrics"]["harmonic_mae"])
                }
            
            # Process predictions DataFrame
            if "predictions" in prediction_results:
                predictions_df = prediction_results["predictions"]
                processed["predictions"] = {
                    "index": [idx.isoformat() for idx in predictions_df.index],
                    "actual": predictions_df["actual"].tolist(),
                    "predicted": predictions_df["predicted"].tolist()
                }
            
            # Process feature importance for XGBoost
            if model_type.lower() == 'xgb' and "feature_importance" in prediction_results and "average" in prediction_results["feature_importance"]:
                # Convert feature importance dict to simple strings as keys
                importance_dict = {}
                for feat, value in prediction_results["feature_importance"]["average"].items():
                    key = str(feat) if isinstance(feat, tuple) else str(feat)
                    importance_dict[key] = float(value)
                
                processed["feature_importance"] = importance_dict
            
            # Process VAR model diagnostics if available
            if model_type.lower() == 'var' and "var_diagnostics" in prediction_results:
                processed["var_diagnostics"] = {
                    "avg_significant_ratio": float(prediction_results["var_diagnostics"]["avg_significant_ratio"]),
                    "significant_pairs_count": len(prediction_results["var_diagnostics"]["significant_pairs"])
                }
            
            # Process ARIMA model order if available
            if model_type.lower() == 'arima' and "order" in prediction_results:
                processed["arima_order"] = prediction_results["order"]
                
        else:
            # For non-cross-validation, include predictions for next day
            if "predictions" in prediction_results:
                predictions_df = prediction_results["predictions"]
                processed["predictions"] = {
                    "index": [idx.isoformat() for idx in predictions_df.index],
                    "predicted": predictions_df["predicted"].tolist()
                }
            
            # Include prediction date
            if "prediction_date" in prediction_results:
                processed["prediction_date"] = prediction_results["prediction_date"].isoformat()
            
            # Process feature importance for XGBoost
            if model_type.lower() == 'xgb' and "feature_importance" in prediction_results:
                importance_dict = {}
                for feat, value in prediction_results["feature_importance"].items():
                    key = str(feat) if isinstance(feat, tuple) else str(feat)
                    importance_dict[key] = float(value)
                
                processed["feature_importance"] = importance_dict
                
            # Include ARIMA model summary if available
            if model_type.lower() == 'arima' and "model_summary" in prediction_results:
                processed["model_summary"] = {
                    "aic": float(prediction_results["model_summary"]["aic"]),
                    "bic": float(prediction_results["model_summary"]["bic"])
                }
        
        # Ensure all numpy types are converted to Python types
        return convert_tuple_keys(processed)
        
    except Exception as e:
        logger.error(f"Error processing prediction results: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "error": "Failed to process prediction results",
            "model_type": model_type,
            "target_ticker": target_ticker
        }

@app.route('/save-model', methods=['POST'])
def save_model():
    """Save the current model to a file"""
    global predictor
    
    if predictor is None:
        return jsonify({
            "status": "error",
            "message": "No predictor available. Call /initialize first."
        }), 400
    
    try:
        data = request.json
        model_path = data.get('model_path', 'model.pkl')
        model_type = data.get('model_type', 'xgb')  # Which model to save
        
        # Determine which model to save based on model_type
        model_to_save = None
        
        if model_type.lower() == 'xgb' and predictor.xgb_model is not None:
            model_to_save = predictor.xgb_model
        elif model_type.lower() == 'arima' and predictor.arima_model is not None:
            model_to_save = predictor.arima_model
        elif model_type.lower() == 'var' and hasattr(predictor, 'var_model') and predictor.var_model is not None:
            model_to_save = predictor.var_model
        
        if model_to_save is None:
            return jsonify({
                "status": "error",
                "message": f"No trained {model_type.upper()} model available. Run prediction first."
            }), 400
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(model_to_save, f)
            
        return jsonify({
            "status": "success",
            "message": f"{model_type.upper()} model saved to {model_path}"
        })
        
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        return jsonify({
            "status": "error", 
            "message": str(e)
        }), 500

@app.route('/available-models', methods=['GET'])
def available_models():
    """Get information about available model types and their status"""
    global predictor
    
    if predictor is None:
        return jsonify({
            "status": "error",
            "message": "Predictor not initialized. Call /initialize first."
        }), 400
    
    try:
        models = {
            "xgb": {
                "available": True,
                "trained": predictor.xgb_model is not None
            },
            "arima": {
                "available": True,
                "trained": predictor.arima_model is not None
            },
            "var": {
                "available": True,
                "trained": hasattr(predictor, 'var_model') and predictor.var_model is not None
            }
        }
        
        return jsonify({
            "status": "success",
            "models": models
        })
        
    except Exception as e:
        logger.error(f"Error getting model information: {str(e)}")
        return jsonify({
            "status": "error", 
            "message": str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)