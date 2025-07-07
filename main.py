import os
import shutil
import uuid
import pickle
import torch
import clip
import cv2
import numpy as np
import faiss
from PIL import Image
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# ===================== FastAPI App & CORS =====================
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== Model & Data =====================
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

CLUSTER_FILE = "clusters.pkl"
FAISS_FILE = "faiss.index"
clusters = {}          # {game_name: [feature_vectors]}
cluster_labels = []    # list of game names aligned with cluster_features
cluster_features = []  # list of all feature vectors (np.array)
faiss_index = None     # FAISS index for similarity search

CONFIDENCE_THRESHOLD = 0.7  # KNN confidence threshold

# ===================== Utility Functions =====================
def save_clusters():
    with open(CLUSTER_FILE, "wb") as f:
        pickle.dump((clusters, cluster_labels, cluster_features), f)

def load_clusters():
    global clusters, cluster_labels, cluster_features
    if os.path.exists(CLUSTER_FILE):
        with open(CLUSTER_FILE, "rb") as f:
            clusters, cluster_labels, cluster_features = pickle.load(f)

def save_faiss_index():
    if faiss_index is not None:
        faiss.write_index(faiss_index, FAISS_FILE)

def load_faiss_index():
    global faiss_index
    if os.path.exists(FAISS_FILE):
        faiss_index = faiss.read_index(FAISS_FILE)

def build_faiss_index():
    global faiss_index
    if not cluster_features:
        faiss_index = None
        return
    features_np = np.stack(cluster_features).astype(np.float32)
    dim = features_np.shape[1]
    faiss_index = faiss.IndexFlatL2(dim)
    faiss_index.add(features_np)
    save_faiss_index()

def extract_features(image: Image.Image) -> np.ndarray:
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(image_tensor)
    return features.cpu().numpy().astype(np.float32).flatten()

def extract_frames(video_path: str) -> list:
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 1
    frames = []
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % fps == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb_frame))
        frame_idx += 1
    cap.release()
    return frames

def visualize_clusters() -> str | None:
    if not clusters:
        return None
    all_features = []
    all_labels = []
    for game, feats in clusters.items():
        all_features.extend(feats)
        all_labels.extend([game] * len(feats))
    all_features = np.array(all_features)
    if len(all_features) < 2:
        return None
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(all_features)
    unique_labels = list(set(all_labels))
    palette = sns.color_palette("husl", len(unique_labels))
    plt.figure(figsize=(10, 7))
    for i, label in enumerate(unique_labels):
        idxs = [j for j, l in enumerate(all_labels) if l == label]
        plt.scatter(reduced[idxs, 0], reduced[idxs, 1], label=label, color=palette[i], alpha=0.7)
    plt.legend()
    plt.xlabel("PCA Feature 1")
    plt.ylabel("PCA Feature 2")
    plt.title("Gameplay Clustering Visualization")
    os.makedirs("static", exist_ok=True)
    img_path = f"static/cluster_visualization_{uuid.uuid4().hex}.png"
    plt.savefig(img_path)
    plt.close()
    return img_path

def knn_predict(features: list, k=3):
    if faiss_index is None or not cluster_labels:
        return None, 0.0
    features_np = np.stack(features).astype(np.float32)
    avg_feature = np.mean(features_np, axis=0).reshape(1, -1)
    distances, indices = faiss_index.search(avg_feature, k)
    votes = {}
    for idx in indices[0]:
        if idx < len(cluster_labels):
            label = cluster_labels[idx]
            votes[label] = votes.get(label, 0) + 1
    if not votes:
        return None, 0.0
    best_label = max(votes, key=votes.get)
    confidence = votes[best_label] / k
    return best_label, confidence

# ===================== Load Data on Startup =====================
load_clusters()
build_faiss_index()
load_faiss_index()

# ===================== API Endpoints =====================
@app.get("/stats")
async def get_stats():
    stats = {game: len(feats) for game, feats in clusters.items()}
    return JSONResponse(content=stats)

@app.get("/visualize_clusters")
async def get_cluster_visualization():
    img_path = visualize_clusters()
    if img_path:
        return JSONResponse(content={"image_url": "/" + img_path})
    return JSONResponse(content={"error": "No clusters available"}, status_code=404)

@app.post("/upload/")
async def upload_video(
    video: UploadFile = File(...),
    is_labeled: bool = Form(...),
    game_name: str = Form(None)
):
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, video.filename)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    frames = extract_frames(temp_path)
    os.remove(temp_path)

    if not frames:
        return JSONResponse(content={"error": "Could not process video"}, status_code=400)

    all_features = [extract_features(frame) for frame in frames]

    if is_labeled:
        if not game_name or not game_name.strip():
            return JSONResponse(content={"error": "Game name must be provided for labeled videos."}, status_code=400)
        game_name = game_name.strip()
        clusters.setdefault(game_name, []).extend(all_features)
        cluster_labels.extend([game_name] * len(all_features))
        cluster_features.extend(all_features)
        save_clusters()
        build_faiss_index()
        img_url = visualize_clusters()
        return {
            "success": True,
            "game": game_name,
            "duration": len(frames),
            "clusters": [{"label": game_name, "frames": len(frames)}],
            "visualization": f"/{img_url}" if img_url else None
        }
    else:
        if not cluster_features:
            return {"success": False, "message": "No existing clusters to compare with"}
        best_label, confidence = knn_predict(all_features, k=3)
        if confidence < CONFIDENCE_THRESHOLD or not best_label:
            new_game_name = f"Unknown_{uuid.uuid4().hex[:6]}"
            clusters[new_game_name] = all_features
            cluster_labels.extend([new_game_name] * len(all_features))
            cluster_features.extend(all_features)
            save_clusters()
            build_faiss_index()
            img_url = visualize_clusters()
            return {
                "success": True,
                "game": new_game_name,
                "duration": len(frames),
                "clusters": [{"label": new_game_name, "frames": len(frames)}],
                "visualization": f"/{img_url}" if img_url else None
            }
        else:
            clusters[best_label].extend(all_features)
            cluster_labels.extend([best_label] * len(all_features))
            cluster_features.extend(all_features)
            save_clusters()
            build_faiss_index()
            img_url = visualize_clusters()
            return {
                "success": True,
                "game": best_label,
                "duration": len(frames),
                "clusters": [{"label": best_label, "frames": len(frames)}],
                "visualization": f"/{img_url}" if img_url else None
            }

@app.post("/reset_clusters")
async def reset_clusters():
    global clusters, cluster_labels, cluster_features, faiss_index
    clusters = {}
    cluster_labels = []
    cluster_features = []
    faiss_index = None
    save_clusters()
    if os.path.exists(FAISS_FILE):
        os.remove(FAISS_FILE)
    return JSONResponse(content={"success": True, "message": "All clusters have been reset."})