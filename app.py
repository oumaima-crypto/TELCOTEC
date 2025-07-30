from flask import Flask, render_template, request, redirect, url_for, session, g, send_file
import tensorflow as tf
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime
import secrets
import logging
import io
import joblib
import trimesh
from pathlib import Path
import os

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = False
app.static_folder = 'static'

DATABASE = 'app.db'
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db():
    with app.app_context():
        db = get_db()
        db.execute('''CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )''')
        db.execute('''CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            building_name TEXT NOT NULL,
            building_address TEXT NOT NULL,
            datetime TEXT NOT NULL,
            x REAL,
            y REAL,
            z REAL,
            x0 REAL,
            y0 REAL,
            z0 REAL,
            obstacles INTEGER,
            clutter_height REAL,
            distance REAL,
            angle REAL,
            path_loss REAL
        )''')
        db.commit()
init_db()

# --- Load TensorFlow DNN Model and Scalers (X and y) ---
try:
    # model_path = "C:/Users/HP PAVILION/telcotec/model_pathloss_dnn.keras"
    model_path = "C:/Users/ghass/Downloads/TELCOTEC-main/TELCOTEC-main/model_pathloss_dnn.keras"
    scaler_x_path = "C:/Users/ghass/Downloads/TELCOTEC-main/TELCOTEC-main/scaler_X.pkl"
    scaler_y_path = "C:/Users/ghass/Downloads/TELCOTEC-main/TELCOTEC-main/scaler_y.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(scaler_x_path):
        raise FileNotFoundError(f"Scaler X file not found: {scaler_x_path}")
    if not os.path.exists(scaler_y_path):
        raise FileNotFoundError(f"Scaler y file not found: {scaler_y_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    scaler = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)
except Exception as e:
    logging.error(f"Model or scaler loading failed: {e}")
    raise RuntimeError(f"Model or scaler loading failed: {e}")

FEATURES = ['x', 'y', 'z', 'x0', 'y0', 'z0', 'obstacles', 'clutter_height', 'distance', 'angle']
# Assurez-vous que l'ordre et le nombre de features correspondent à l'entraînement du modèle et du scaler.

def predict_path_loss(form):
    try:
        # Récupération des features depuis le formulaire
        x = float(form.get('x', 0))
        y = float(form.get('y', 0))
        z = float(form.get('z', 0))
        x0 = float(form.get('x0', 0))
        y0 = float(form.get('y0', 0))
        z0 = float(form.get('z0', 0))
        obstacles = int(form.get('obstacles', 0))
        clutter_height = float(form.get('clutter_height', 0))
        distance = float(form.get('distance', 0))
        angle = float(form.get('angle', 0))
        # Validation des valeurs
        if not (0 <= x <= 20 and 0 <= y <= 20 and 0 <= z <= 4 and
                0 <= x0 <= 20 and 0 <= y0 <= 20 and 2 <= z0 <= 10 and
                0 <= obstacles <= 4 and 0.5 <= clutter_height <= 2.5 and
                1 <= distance <= 30 and 0 <= angle <= 90):
            return None, "Input values out of allowed range."
        features = [[x, y, z, x0, y0, z0, obstacles, clutter_height, distance, angle]]
        # Scale X
        features_scaled = scaler.transform(features)
        # Predict (model expects scaled X)
        y_pred_scaled = model.predict(features_scaled, verbose=0)
        # Inverse transform y
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))[0][0]
        return y_pred, None
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        import traceback; traceback.print_exc()
        return None, f"Prediction error: {e}"

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    try:
        if request.method == 'POST':
            username = request.form['username'].strip()
            password = request.form['password']
            db = get_db()
            user = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
            if user and user['password'] == password:
                session['username'] = username
                return redirect(url_for('home'))
            else:
                error = "Invalid credentials."
    except Exception as e:
        logging.error(f"Login error: {e}")
        error = "Login failed. Please try again."
    return render_template("login.html", error=error)

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        if not username or not password:
            error = "Username and password are required."
        else:
            db = get_db()
            try:
                db.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
                db.commit()
                return redirect(url_for('login'))
            except sqlite3.IntegrityError:
                error = "Username already exists."
    return render_template("register.html", error=error)

@app.route('/', methods=['GET', 'POST'])
def home():
    if 'username' not in session:
        return redirect(url_for('login'))
    error = None
    prediction = None
    show_save = False
    if request.method == 'POST':
        # If the Save modal is triggered
        if request.form.get('show_save') == '1':
            prediction = session.get('last_prediction', {}).get('path_loss')
            show_save = True
        else:
            prediction, error = predict_path_loss(request.form)
            if prediction is not None:
                # Save all input values for later use in the Save modal
                session['last_prediction'] = {
                    'x': float(request.form['x']),
                    'y': float(request.form['y']),
                    'z': float(request.form['z']),
                    'x0': float(request.form['x0']),
                    'y0': float(request.form['y0']),
                    'z0': float(request.form['z0']),
                    'obstacles': int(request.form['obstacles']),
                    'clutter_height': float(request.form['clutter_height']),
                    'distance': float(request.form['distance']),
                    'angle': float(request.form['angle']),
                    'path_loss': float(prediction)
                }
    return render_template("prediction.html", error=error, prediction=prediction, show_save=show_save)

@app.route('/save_prediction', methods=['POST'])
def save_prediction():
    if 'username' not in session or 'last_prediction' not in session:
        return redirect(url_for('login'))
    building_name = request.form['building_name']
    building_address = request.form['building_address']
    pred = session.pop('last_prediction')
    db = get_db()
    db.execute('''INSERT INTO predictions (username, building_name, building_address, datetime, x, y, z, x0, y0, z0, obstacles, clutter_height, distance, angle, path_loss)
                  VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
               (session['username'], building_name, building_address, datetime.now().isoformat(),
                pred['x'], pred['y'], pred['z'], pred['x0'], pred['y0'], pred['z0'], pred['obstacles'], pred['clutter_height'], pred['distance'], pred['angle'], pred['path_loss']))
    db.commit()
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    db = get_db()
    building = request.args.get('building')
    address = request.args.get('address')
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page',6))  # changed from 10 to 3
    query = 'SELECT * FROM predictions WHERE username=?'
    params = [session['username']]
    if building:
        query += ' AND building_name=?'
        params.append(building)
    if address:
        query += ' AND building_address=?'
        params.append(address)
    query += ' ORDER BY datetime DESC'
    # Get total count for pagination
    count_query = 'SELECT COUNT(*) FROM (' + query + ')'
    total = db.execute(count_query, params).fetchone()[0]
    offset = (page - 1) * per_page
    query += ' LIMIT ? OFFSET ?'
    preds = db.execute(query, params + [per_page, offset]).fetchall()
    preds_dicts = [dict(p) for p in preds]
    building_names = sorted(list({p['building_name'] for p in preds_dicts}))
    addresses = sorted(list({p['building_address'] for p in preds_dicts}))
    total_pages = (total + per_page - 1) // per_page
    return render_template(
        "dashboard.html",
        predictions=preds_dicts,
        building_names=building_names,
        addresses=addresses,
        page=page,
        per_page=per_page,
        total=total,
        total_pages=total_pages
    )

# @app.route('/viz')
# def viz():
#     if 'username' not in session:
#         return redirect(url_for('login'))
#     building = request.args.get('building')
#     stl_path = Path(r'C:\Users\HP PAVILION\telcotec\static\floorplan.stl')
#     if not stl_path.exists():
#         return "Fichier .stl non trouvé à l'emplacement spécifié.", 404

#     db = get_db()
#     preds = db.execute('SELECT * FROM predictions WHERE username=? AND building_name=?',
#                       (session['username'], building)).fetchall() if building else []
#     preds_dicts = [dict(p) for p in preds]

#     ap_position = np.array([5.0, 5.0, 2.0])
#     grid_size = 20
#     resolution = 1.0
#     attenuation_per_meter = 2.0
#     wall_attenuation = 5.0

#     mesh = None
#     try:
#         # Try loading as binary STL first
#         mesh = trimesh.load(str(stl_path), file_type='stl', force='mesh', process=False)
#         logging.info("Fichier .stl chargé avec succès via trimesh.")
#     except Exception as e:
#         logging.warning(f"Erreur binaire lors du chargement du fichier .stl: {e}")
#         # Try loading as ASCII STL
#         try:
#             mesh = trimesh.load(str(stl_path), file_type='stl', force='mesh', process=True)
#             logging.info("Fichier .stl ASCII chargé avec succès via trimesh.")
#         except Exception as e2:
#             logging.error(f"Erreur lors du chargement du fichier .stl: {e2}")
#             return f"Erreur de chargement du fichier .stl: {e2}", 500

#     x = np.arange(0, grid_size, resolution)
#     y = np.arange(0, grid_size, resolution)
#     z = np.arange(0, 4, resolution)
#     X, Y, Z = np.meshgrid(x, y, z)
#     positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

#     signal_strength = np.zeros(len(positions))
#     if mesh:
#         for i, pos in enumerate(positions):
#             distance = np.linalg.norm(pos - ap_position)
#             base_loss = attenuation_per_meter * distance
#             obstacle_effect = wall_attenuation if mesh.is_intersecting(pos) else 0.0
#             signal_strength[i] = -base_loss - obstacle_effect
#     else:
#         for i, pos in enumerate(positions):
#             distance = np.linalg.norm(pos - ap_position)
#             signal_strength[i] = -attenuation_per_meter * distance

#     signal_strength = signal_strength.reshape(X.shape)
#     min_loss = np.min(signal_strength)
#     max_loss = np.max(signal_strength)

#     grid = signal_strength[:, :, 0].tolist()
#     return render_template('viz.html', grid=grid, x_range=[0, grid_size], y_range=[0, grid_size], z=0,
#                           min_loss=min_loss, max_loss=max_loss, building=building, predictions=preds_dicts)

@app.route('/viz')
def viz():
    """
    Visualization endpoint that generates a 3D heatmap of signal strength
    """
    # Check if user is logged in (uncomment if authentication is needed)
    # if 'username' not in session:
    #     return redirect(url_for('login'))
    
    building = request.args.get('building', 'Test Building')
    
    # Generate a 3D grid of path-loss values (distance-based model)
    grid_size = 20
    resolution = 1.0
    ap_position = np.array([5.0, 5.0, 2.0])  # Access point position
    attenuation_per_meter = 2.0
    
    x = np.arange(0, grid_size, resolution)
    y = np.arange(0, grid_size, resolution)
    z = 0  # 2D plane at z=0 for simplified visualization
    grid = np.zeros((len(y), len(x)))
    
    # Calculate path loss for each grid point
    for i, yi in enumerate(y):
        for j, xj in enumerate(x):
            distance = np.linalg.norm(np.array([xj, yi, z]) - ap_position)
            grid[i][j] = -attenuation_per_meter * distance  # Negative path-loss
    
    min_loss = np.min(grid)
    max_loss = np.max(grid)
    
    return render_template('viz.html',
                         grid=grid.tolist(),
                         x_range=[0, grid_size],
                         y_range=[0, grid_size],
                         z=z,
                         min_loss=min_loss,
                         max_loss=max_loss,
                         building=building)

@app.route('/export_csv')
def export_csv():
    if 'username' not in session:
        return redirect(url_for('login'))
    building = request.args.get('building')
    address = request.args.get('address')
    db = get_db()
    query = 'SELECT * FROM predictions WHERE username=?'
    params = [session['username']]
    if building:
        query += ' AND building_name=?'
        params.append(building)
    if address:
        query += ' AND building_address=?'
        params.append(address)
    query += ' ORDER BY datetime DESC'
    preds = db.execute(query, params).fetchall()
    df = pd.DataFrame(preds, columns=preds[0].keys() if preds else [])
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return send_file(io.BytesIO(output.getvalue().encode()), mimetype='text/csv', as_attachment=True, download_name='predictions.csv')

@app.route('/prediction/<int:id>')
def view_prediction(id):
    if 'username' not in session:
        return redirect(url_for('login'))
    db = get_db()
    pred = db.execute('SELECT * FROM predictions WHERE id=? AND username=?', (id, session['username'])).fetchone()
    if not pred:
        return "Prediction not found.", 404
    return render_template('view_prediction.html', prediction=dict(pred))

@app.route('/delete_prediction/<int:id>', methods=['POST'])
def delete_prediction(id):
    if 'username' not in session:
        return redirect(url_for('login'))
    db = get_db()
    db.execute('DELETE FROM predictions WHERE id=? AND username=?', (id, session['username']))
    db.commit()
    return redirect(url_for('dashboard'))

if __name__ == '__main__':
    app.run(debug=True)