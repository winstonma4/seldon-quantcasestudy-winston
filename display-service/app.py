from flask import Flask, request, jsonify, render_template
import logging
import traceback
import json
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import io
import base64

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Store the last few prediction results
stored_predictions = []
MAX_STORED_PREDICTIONS = 10

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "display-service"})

@app.route('/', methods=['GET'])
def home():
    """Home page that displays stored predictions"""
    return render_template('index.html', 
                          predictions=stored_predictions,
                          count=len(stored_predictions))

@app.route('/display', methods=['POST'])
def display_prediction():
    """Display prediction results"""
    global stored_predictions
    
    try:
        data = request.json
        print(f"Received data at /display endpoint: {json.dumps(data, indent=2)[:500]}...")
        
        prediction_results = data.get('prediction_results', {})
        
        if not prediction_results:
            print("ERROR: No prediction_results in received data")
            return jsonify({
                "status": "error",
                "message": "No prediction results provided"
            }), 400
        
        # Extract key information
        model_type = prediction_results.get('model_type', 'unknown')
        target_ticker = prediction_results.get('target_ticker', 'unknown')
        received_time = datetime.now().isoformat()
        
        print(f"Processing prediction for {target_ticker} using {model_type} model")
        
        # Process and format prediction results
        formatted_results = format_prediction_results(prediction_results)
        
        # Generate plots if there's prediction data
        plots = []
        if 'predictions' in prediction_results:
            try:
                plots = generate_plots(prediction_results)
                print(f"Generated {len(plots)} plots")
            except Exception as e:
                print(f"Error generating plots: {str(e)}")
                traceback.print_exc()
        
        # Store this prediction result 
        result_entry = {
            "id": len(stored_predictions) + 1,
            "received_time": received_time,
            "model_type": model_type,
            "target_ticker": target_ticker,
            "formatted_results": formatted_results,
            "raw_results": prediction_results,
            "plots": plots
        }
        
        print(f"Created result entry with ID {result_entry['id']}")
        
        # Add to stored predictions, maintaining max limit
        stored_predictions.append(result_entry)
        print(f"Added to stored_predictions, now contains {len(stored_predictions)} entries")
        
        if len(stored_predictions) > MAX_STORED_PREDICTIONS:
            stored_predictions = stored_predictions[-MAX_STORED_PREDICTIONS:]
        
        # Print to console (the main purpose of this service)
        print_prediction_results(prediction_results, target_ticker, model_type)
        
        return jsonify({
            "status": "success",
            "message": "Prediction results displayed and stored",
            "result_id": result_entry["id"]
        })
        
    except Exception as e:
        print(f"Error in display_prediction: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "status": "error", 
            "message": str(e)
        }), 500

@app.route('/predictions', methods=['GET'])
def get_predictions():
    """Get list of stored predictions"""
    global stored_predictions
    
    try:
        # Return list of predictions with minimal info
        summary = [{
            "id": pred["id"],
            "received_time": pred["received_time"],
            "model_type": pred["model_type"],
            "target_ticker": pred["target_ticker"]
        } for pred in stored_predictions]
        
        return jsonify({
            "status": "success",
            "count": len(summary),
            "predictions": summary
        })
        
    except Exception as e:
        logger.error(f"Error getting predictions: {str(e)}")
        return jsonify({
            "status": "error", 
            "message": str(e)
        }), 500

@app.route('/predictions/<int:prediction_id>', methods=['GET'])
def get_prediction_detail(prediction_id):
    """Get details of a specific prediction"""
    global stored_predictions
    
    try:
        # Find prediction by ID
        for pred in stored_predictions:
            if pred["id"] == prediction_id:
                return jsonify({
                    "status": "success",
                    "prediction": pred
                })
        
        return jsonify({
            "status": "error",
            "message": f"Prediction ID {prediction_id} not found"
        }), 404
        
    except Exception as e:
        logger.error(f"Error getting prediction detail: {str(e)}")
        return jsonify({
            "status": "error", 
            "message": str(e)
        }), 500

def format_prediction_results(prediction_results):
    """Format prediction results for display"""
    formatted = {}
    
    # Add metrics if available
    if 'metrics' in prediction_results:
        metrics = prediction_results['metrics']
        formatted['metrics'] = {
            'RMSE': f"{metrics.get('rmse', 'N/A'):.6f}",
            'MAE': f"{metrics.get('mae', 'N/A'):.6f}",
            'Harmonic RMSE': f"{metrics.get('harmonic_rmse', 'N/A'):.6f}",
            'Harmonic MAE': f"{metrics.get('harmonic_mae', 'N/A'):.6f}"
        }
    
    # Add prediction date if available
    if 'prediction_date' in prediction_results:
        formatted['prediction_date'] = prediction_results['prediction_date']
    
    # Format feature importance
    if 'feature_importance' in prediction_results:
        importance = prediction_results['feature_importance']
        if isinstance(importance, dict):
            # Sort by importance value
            sorted_importance = sorted(
                importance.items(), key=lambda x: x[1], reverse=True
            )
            # Take top 10
            top_features = {k: f"{v:.6f}" for k, v in sorted_importance[:10]}
            formatted['top_features'] = top_features
    
    # Format prediction counts
    if 'predictions' in prediction_results:
        preds = prediction_results['predictions']
        if 'actual' in preds:
            formatted['prediction_count'] = len(preds['actual'])
        else:
            formatted['prediction_count'] = len(preds['predicted'])
    
    return formatted

def generate_plots(prediction_results):
    """Generate plots from prediction data"""
    plots = []
    
    if 'predictions' not in prediction_results:
        return plots
    
    preds = prediction_results['predictions']
    predicted = preds.get('predicted', [])
    
    if not predicted:
        return plots
    
    try:
        # Generate time index
        if 'index' in preds:
            # Convert ISO strings to datetime
            time_index = [datetime.fromisoformat(ts) for ts in preds['index']]
        else:
            # Generate sequential index
            time_index = list(range(len(predicted)))
        
        # Plot 1: Predicted returns
        plt.figure(figsize=(10, 6))
        plt.plot(time_index, predicted, 'b-', label='Predicted Returns')
        
        # Add actual returns if available
        if 'actual' in preds:
            actual = preds['actual']
            plt.plot(time_index, actual, 'r-', label='Actual Returns')
        
        plt.title('Return Predictions')
        plt.xlabel('Time')
        plt.ylabel('Returns')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        plots.append({
            'title': 'Return Predictions',
            'data': plot_data
        })
        
        # Plot 2: Cumulative returns
        plt.figure(figsize=(10, 6))
        
        # Calculate cumulative returns
        cum_predicted = np.exp(np.cumsum(predicted))
        plt.plot(time_index, cum_predicted, 'b-', label='Predicted Cumulative Return')
        
        if 'actual' in preds:
            actual = preds['actual']
            cum_actual = np.exp(np.cumsum(actual))
            plt.plot(time_index, cum_actual, 'r-', label='Actual Cumulative Return')
        
        plt.title('Cumulative Returns')
        plt.xlabel('Time')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        plots.append({
            'title': 'Cumulative Returns',
            'data': plot_data
        })
        
        # Plot 3: Feature importance if available
        if 'feature_importance' in prediction_results:
            importance = prediction_results['feature_importance']
            if isinstance(importance, dict):
                # Sort and get top features
                sorted_importance = sorted(
                    importance.items(), key=lambda x: x[1], reverse=True
                )
                top_10 = sorted_importance[:10]
                
                plt.figure(figsize=(10, 6))
                features = [item[0] for item in top_10]
                values = [item[1] for item in top_10]
                
                # Create horizontal bar chart
                plt.barh(range(len(features)), values, align='center')
                plt.yticks(range(len(features)), features)
                plt.xlabel('Importance')
                plt.title('Feature Importance')
                plt.gca().invert_yaxis()  # Highest values at the top
                
                # Save plot to base64 string
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)
                plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                plt.close()
                
                plots.append({
                    'title': 'Feature Importance',
                    'data': plot_data
                })
        
        return plots
        
    except Exception as e:
        logger.error(f"Error in plot generation: {str(e)}")
        logger.error(traceback.format_exc())
        return []

def print_prediction_results(prediction_results, ticker, model_type):
    """Print prediction results to console in a formatted way"""
    print("\n" + "="*80)
    print(f"PREDICTION RESULTS FOR {ticker} USING {model_type.upper()}")
    print("="*80)
    
    # Print metrics if available
    if 'metrics' in prediction_results:
        print("\nMETRICS:")
        print("-"*40)
        metrics = prediction_results['metrics']
        print(f"RMSE:          {metrics.get('rmse', 'N/A'):.6f}")
        print(f"MAE:           {metrics.get('mae', 'N/A'):.6f}")
        print(f"Harmonic RMSE: {metrics.get('harmonic_rmse', 'N/A'):.6f}")
        print(f"Harmonic MAE:  {metrics.get('harmonic_mae', 'N/A'):.6f}")
    
    # Print prediction date
    if 'prediction_date' in prediction_results:
        print(f"\nPREDICTION DATE: {prediction_results['prediction_date']}")
    
    # Print feature importance
    if 'feature_importance' in prediction_results:
        print("\nTOP FEATURE IMPORTANCE:")
        print("-"*40)
        importance = prediction_results['feature_importance']
        if isinstance(importance, dict):
            # Sort by importance value
            sorted_importance = sorted(
                importance.items(), key=lambda x: x[1], reverse=True
            )
            # Take top 10
            for i, (feature, value) in enumerate(sorted_importance[:10], 1):
                print(f"{i:2}. {feature:30s}: {value:.6f}")
    
    # Print predictions
    if 'predictions' in prediction_results:
        preds = prediction_results['predictions']
        times = [datetime.fromisoformat(ts) for ts in preds['index']]
        predicted = preds['predicted']
        
        print("\nPREDICTIONS:")
        print("-"*60)
        print(f"{'Time':20s} | {'Predicted':10s}", end="")
        
        if 'actual' in preds:
            actual = preds['actual']
            print(f" | {'Actual':10s} | {'Error':10s}")
            
            for i, (t, p, a) in enumerate(zip(times, predicted, actual)):
                error = p - a
                print(f"{t.strftime('%Y-%m-%d %H:%M'):20s} | {p:10.6f} | {a:10.6f} | {error:10.6f}")
        else:
            print("")  # End the header line
            for i, (t, p) in enumerate(zip(times, predicted)):
                print(f"{t.strftime('%Y-%m-%d %H:%M'):20s} | {p:10.6f}")
    
    print("\n" + "="*80 + "\n")

# HTML templates directory
@app.route('/templates/<path:path>')
def serve_template(path):
    return render_template(path)

if __name__ == '__main__':
    # Create templates directory
    import os
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # Create simple index.html template
    with open('templates/index.html', 'w') as f:
        f.write("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Prediction Results</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
                h1 { color: #333; }
                .prediction { border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }
                .metrics { margin: 10px 0; }
                .feature-list { margin: 10px 0; }
                .plot-container { margin: 20px 0; }
                .plot-container img { max-width: 100%; height: auto; }
            </style>
        </head>
        <body>
            <h1>Prediction Results</h1>
            
            {% if count == 0 %}
            <p>No predictions available. Run a prediction first.</p>
            {% else %}
            <p>Showing {{ count }} recent predictions</p>
            
            {% for pred in predictions %}
            <div class="prediction">
                <h2>{{ pred.target_ticker }} ({{ pred.model_type }})</h2>
                <p>Received: {{ pred.received_time }}</p>
                
                {% if pred.formatted_results.metrics %}
                <div class="metrics">
                    <h3>Metrics</h3>
                    <ul>
                    {% for metric, value in pred.formatted_results.metrics.items() %}
                        <li>{{ metric }}: {{ value }}</li>
                    {% endfor %}
                    </ul>
                </div>
                {% endif %}
                
                {% if pred.formatted_results.top_features %}
                <div class="feature-list">
                    <h3>Top Features</h3>
                    <ul>
                    {% for feature, value in pred.formatted_results.top_features.items() %}
                        <li>{{ feature }}: {{ value }}</li>
                    {% endfor %}
                    </ul>
                </div>
                {% endif %}
                
                {% if pred.plots %}
                <h3>Plots</h3>
                {% for plot in pred.plots %}
                <div class="plot-container">
                    <h4>{{ plot.title }}</h4>
                    <img src="data:image/png;base64,{{ plot.data }}" alt="{{ plot.title }}">
                </div>
                {% endfor %}
                {% endif %}
            </div>
            {% endfor %}
            {% endif %}
        </body>
        </html>
        """)
    
    app.run(host='0.0.0.0', port=5003, debug=True)
