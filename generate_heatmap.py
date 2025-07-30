import numpy as np
import pyvista as pv
import tensorflow as tf
import pandas as pd

def generate_heatmap(building_name):
    stl_path = f"static/floorplans/{building_name}.stl"
    data = pd.read_csv(f"predictions_{building_name}.csv")  # or query your DB
    # ... generate heatmap as before ...
    # Save as static/signal_heatmap_{building_name}.png

# Usage: generate_heatmap('appartement_ennour')

# --- Load STL Floorplan ---
floor_mesh = pv.read("C:/Users/ghass/Downloads/TELCOTEC-main/TELCOTEC-main/static/floorplan.stl")

# --- Define Grid Parameters ---
x_min, x_max, y_min, y_max = floor_mesh.bounds[0], floor_mesh.bounds[1], floor_mesh.bounds[2], floor_mesh.bounds[3]
z = 1.5  # Height above floor for sensor points

grid_size = 50  # 50x50 grid
xs = np.linspace(x_min, x_max, grid_size)
ys = np.linspace(y_min, y_max, grid_size)
xx, yy = np.meshgrid(xs, ys)
zz = np.full_like(xx, z)

points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

# --- Load Model and Normalization ---
model = tf.keras.models.load_model("path_loss_model.h5", compile=False)
dataset = pd.read_csv("pathloss_dataset.csv")
X = dataset[['distance', 'obstacles', 'hauteur', 'clutter_height', 'angle']].values
y = dataset['path_loss'].values
X_means = X.mean(axis=0)
X_stds = X.std(axis=0)
y_mean = y.mean()
y_std = y.std()

# --- Predict Path Loss at Each Point ---
tx = np.array([x_min, y_min, z])
features = []
for pt in points:
    distance = np.linalg.norm(pt - tx)
    obstacles = 1  # or use a function to estimate obstacles
    hauteur = 3
    clutter_height = 1
    angle = 45
    features.append([distance, obstacles, hauteur, clutter_height, angle])
features = np.array(features)
features_norm = (features - X_means) / X_stds
preds = model.predict(features_norm, verbose=0).flatten()
preds = preds * y_std + y_mean  # De-normalize

# --- Map Predictions to Grid for Visualization ---
point_cloud = pv.PolyData(points)
point_cloud["PathLoss"] = preds

# --- Plotting ---
plotter = pv.Plotter(off_screen=True)
plotter.add_mesh(floor_mesh, color="white", opacity=0.3, show_edges=True)
plotter.add_points(
    point_cloud,
    scalars="PathLoss",
    cmap="coolwarm_r",  # Red (bad) to blue (good)
    point_size=12,
    render_points_as_spheres=True,
    opacity=0.8,
)
plotter.add_scalar_bar(title="Path Loss (dB)", n_labels=5)
plotter.show(screenshot="static/signal_heatmap.png")
print("Visualization complete. PNG saved as static/signal_heatmap.png.")