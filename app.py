import time
import csv
import os
import signal
import threading
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from PeakPxApi import PeakPx
import uuid
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from cachetools import TTLCache
from collections import Counter
from better_profanity import profanity
import traceback

app = Flask(__name__)
CORS(app)
px = PeakPx()

# Server key
server_key = "wallartify2024new"

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Setup CSV file for logging queries
def setup_csv():
    csv_file = 'ip_query_log.csv'
    if not os.path.exists(csv_file):
        try:
            with open(csv_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["ID", "IP Address", "Query", "Timestamp", "Response Status"])
        except Exception as e:
            logging.error(f"Error setting up CSV file: {e}")

# Dictionary to store client IPs and their corresponding IDs
client_ids = {}

# Log query to CSV
def log_query(ip_address, query, response_success, file):
    try:
        if (client_id := client_ids.get(ip_address)) is None:
            client_id = str(uuid.uuid4())
            client_ids[ip_address] = client_id
        unique_id = client_id
        writer = csv.writer(file)
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        writer.writerow([unique_id, ip_address, query.lower(), timestamp, response_success])
    except Exception as e:
        logging.error(f"Error logging query: {e}")

# Validate client key
def validate_key(request):
    try:
        client_sent_key = request.args.get('key')
        return client_sent_key == server_key
    except Exception as e:
        logging.error(f"Error validating client key: {e}")
        return False

# Caching results of the search for 5 minutes
cache = TTLCache(maxsize=100, ttl=300)

# Search wallpapers using PeakPx API
def search_wallpapers(query):
    if query in cache:
        return cache[query], True

    try:
        wallpapers = px.search_wallpapers(query=query)
        if wallpapers:
            image_urls = [wallpaper['url'] for wallpaper in wallpapers]
            cache[query] = image_urls
            return image_urls, True
        else:
            return [], False
    except Exception as e:
        logging.error(f"Error searching wallpapers: {e}")
        return [], False

# Train model for recommendation system
def train_model(queries):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(queries)
    kmeans = KMeans(n_clusters=5, random_state=0)
    kmeans.fit(X)
    return vectorizer, kmeans

# Partial query recommendation function
def partial_query_recommendation(partial_query, queries, query_counter, vectorizer, model, top_n=5, min_frequency=5):
    try:
        partial_vec = vectorizer.transform([partial_query.lower()])
        similarities = cosine_similarity(partial_vec, vectorizer.transform(queries)).flatten()
        sorted_indices = np.argsort(similarities)[::-1]

        unique_recommendations = []
        seen = set()
        for idx in sorted_indices:
            if queries[idx].startswith(partial_query.lower()) and query_counter[queries[idx]] >= min_frequency and queries[idx] not in seen:
                unique_recommendations.append((queries[idx], query_counter[queries[idx]]))
                seen.add(queries[idx])
            if len(unique_recommendations) == top_n:
                break

        # Sort by frequency
        unique_recommendations.sort(key=lambda x: (-x[1], x[0]))

        return [recommendation for recommendation, _ in unique_recommendations]
    except Exception as e:
        logging.error(f"Error in partial_query_recommendation: {e}")
        return []

# Load data from CSV file for logging queries
def load_data(filename):
    queries = []
    try:
        with open(filename, 'r', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                queries.append(row['Query'].lower())
    except Exception as e:
        logging.error(f"Error loading data: {e}")
    return queries

# Function to check if a query is inappropriate using better_profanity
def is_inappropriate(query):
    return profanity.contains_profanity(query)

# API endpoint to search wallpapers
@app.route('/search_wallpapers', methods=['GET'])
def search_wallpapers_route():
    if not validate_key(request):
        return jsonify({'error': 'Invalid client key'}), 401

    query = request.args.get('query')
    if not query:
        return jsonify({'error': 'Query parameter is required'}), 400

    # Check if the query is inappropriate
    if is_inappropriate(query):
        return jsonify([{'Image': 'https://i.pinimg.com/736x/95/55/07-9555074fb5a23ba2f2513597a95827a1.jpg'}]), 400

    client_ip = request.remote_addr
    image_urls, success = search_wallpapers(query)
    with open('ip_query_log.csv', 'a', newline='') as file:
        log_query(client_ip, query, success, file)

    if success:
        response_data = [{'Image': url} for url in image_urls]
        return jsonify(response_data), 200
    else:
        return jsonify([{'recommend': 'No wallpapers found for the given query'}]), 404

# API endpoint to view logs
@app.route('/view_logs', methods=['GET'])
def view_logs():
    if not validate_key(request):
        return jsonify({'error': 'Invalid client key'}), 401

    logs = []
    try:
        with open('ip_query_log.csv', 'r', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                logs.append({'ID': row['ID'], 'IP Address': row['IP Address'], 'Query': row['Query'], 'Timestamp': row['Timestamp'], 'Response Success': row['Response Status']})
    except Exception as e:
        logging.error(f"An unexpected error occurred while reading logs: {e}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

    return jsonify(logs), 200

# API endpoint to get recommendations for partial queries
@app.route('/recommendations', methods=['GET'])
def get_recommendations():
    if not validate_key(request):
        return jsonify({'error': 'Invalid client key'}), 401

    partial_query = request.args.get('q')
    if not partial_query:
        return jsonify({'error': 'Partial query parameter (q) is required'}), 400

    # Check if the partial query is inappropriate
    if is_inappropriate(partial_query):
        return jsonify([{'recommend': 'Inappropriate query'}]), 400

    queries = load_data('ip_query_log.csv')
    query_counter = Counter(queries)
    vectorizer, model = train_model(queries)
    recommendations = partial_query_recommendation(partial_query, queries, query_counter, vectorizer, model)
    recommendations_list = [{'recommend': recommendation} for recommendation in recommendations if not is_inappropriate(recommendation)]

    return jsonify(recommendations_list), 200

# API endpoint to get trending queries
@app.route('/trending', methods=['GET'])
def get_trending():
    if not validate_key(request):
        return jsonify({'error': 'Invalid client key'}), 401

    queries = load_data('ip_query_log.csv')
    trending_queries = Counter(queries).most_common(10)
    trending_list = [{'query': query, 'count': count} for query, count in trending_queries if not is_inappropriate(query)]

    return jsonify(trending_list), 200

def run_flask_app():
    setup_csv()

    try:
        logging.info("Starting server...")
        app.run(host='0.0.0.0', port=5000)
    except Exception as e:
        logging.error(f"Error running Flask app: {e}")
        traceback.print_exc()  # Print exception traceback

def signal_handler(sig, frame):
    logging.info("Exiting...")
    exit(0)

if __name__ == "__main__":
    profanity.load_censor_words()
    signal.signal(signal.SIGINT, signal_handler)
    run_flask_app()
