import trimesh
import io
from stl import mesh

stl_path = r'C:\Users\ghass\Downloads\TELCOTEC-main\TELCOTEC-main\static\floorplan.stl'

try:
    # Test avec trimesh
    with open(stl_path, 'rb') as f:
        binary_data = f.read()
        mesh_trimesh = trimesh.load(file_obj=io.BytesIO(binary_data), file_type='stl', process=False)
    print("Fichier chargé avec trimesh :", mesh_trimesh)
    print("Nombre de faces :", len(mesh_trimesh.faces))
except Exception as e:
    print("Erreur avec trimesh :", e)

try:
    # Test avec numpy-stl (corrigé)
    with open(stl_path, 'rb') as f:
        mesh_numpy = mesh.Mesh.from_file(f.name, fh=f)  # Utiliser le fichier directement
        mesh_trimesh_from_numpy = trimesh.Trimesh(vertices=mesh_numpy.vectors.reshape(-1, 3), faces=mesh_numpy.vectors.shape[:-1])
    print("Fichier chargé avec numpy-stl :", mesh_trimesh_from_numpy)
    print("Nombre de faces :", len(mesh_trimesh_from_numpy.faces))
except Exception as e:
    print("Erreur avec numpy-stl :", e)