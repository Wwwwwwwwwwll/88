from flask import Flask, render_template, request, jsonify, send_from_directory
import numpy as np
import pandas as pd
import cv2
import os
import base64
import io
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from skimage.feature import graycomatrix, graycoprops
from skimage import img_as_ubyte
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import json
import pickle

warnings.filterwarnings("ignore")

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size

# Create necessary directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("static/plots", exist_ok=True)
os.makedirs("models", exist_ok=True)


class MangoDiseaseDetector:
    def __init__(self):
        self.scaler = StandardScaler()
        self.svm_model = SVC(kernel="rbf", random_state=42, probability=True)
        self.features = []
        self.labels = []
        self.is_trained = False
        self.training_accuracy = 0
        self.label_names = [
            "Alternaria",
            "Anthracnose",
            "Black Mould Rot",
            "Busuk",
            "Sehat",
        ]

    def preprocess_image(self, image_path, target_size=(256, 256)):
        """Preprocessing gambar: resize, convert ke grayscale, dan noise reduction"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None

            # Resize gambar
            img_resized = cv2.resize(img, target_size)
            # Convert ke grayscale
            gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            # Gaussian blur untuk noise reduction
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            # Histogram equalization untuk meningkatkan kontras
            equalized = cv2.equalizeHist(blurred)

            return equalized
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None

    def extract_glcm_features(
        self, image, distances=[1, 2, 3], angles=[0, 45, 90, 135]
    ):
        """Ekstraksi fitur GLCM dari gambar"""
        try:
            # Convert ke format yang sesuai untuk GLCM
            image = img_as_ubyte(image)

            # Hitung GLCM
            glcm = graycomatrix(
                image,
                distances=distances,
                angles=np.radians(angles),
                levels=256,
                symmetric=True,
                normed=True,
            )

            # Ekstraksi properti GLCM
            features = []

            # Contrast - mengukur variasi intensitas lokal
            contrast = graycoprops(glcm, "contrast")
            features.extend(contrast.flatten())

            # Dissimilarity - mengukur ketidaksamaan
            dissimilarity = graycoprops(glcm, "dissimilarity")
            features.extend(dissimilarity.flatten())

            # Homogeneity - mengukur keseragaman tekstur
            homogeneity = graycoprops(glcm, "homogeneity")
            features.extend(homogeneity.flatten())

            # Energy - mengukur keseragaman
            energy = graycoprops(glcm, "energy")
            features.extend(energy.flatten())

            # Correlation - mengukur korelasi linier
            correlation = graycoprops(glcm, "correlation")
            features.extend(correlation.flatten())

            # ASM (Angular Second Moment)
            asm = graycoprops(glcm, "ASM")
            features.extend(asm.flatten())

            return np.array(features)
        except Exception as e:
            print(f"Error extracting GLCM features: {e}")
            return None

    def load_dataset(self, dataset_path):
        """Load dataset dari folder dengan struktur penyakit mangga"""
        dataset_path = Path(dataset_path)

        # Mapping label untuk 5 kriteria penyakit mangga
        label_mapping = {
            "Alternaria": 0,
            "Anthracnose": 1,
            "Black Mould Rot": 2,
            "busuk": 3,
            "sehat": 4,
        }

        features = []
        labels = []
        processed_count = 0

        for category in label_mapping.keys():
            category_path = dataset_path / category
            if not category_path.exists():
                print(f"Warning: Folder {category} tidak ditemukan!")
                continue

            print(f"Memproses kategori: {category}")

            # Proses setiap gambar dalam kategori
            image_files = (
                list(category_path.glob("*.jpg"))
                + list(category_path.glob("*.png"))
                + list(category_path.glob("*.jpeg"))
                + list(category_path.glob("*.JPG"))
                + list(category_path.glob("*.PNG"))
                + list(category_path.glob("*.JPEG"))
            )

            for img_path in image_files:
                # Preprocessing gambar
                processed_img = self.preprocess_image(str(img_path))

                if processed_img is not None:
                    # Ekstraksi fitur GLCM
                    glcm_features = self.extract_glcm_features(processed_img)

                    if glcm_features is not None:
                        features.append(glcm_features)
                        labels.append(label_mapping[category])
                        processed_count += 1

        self.features = np.array(features)
        self.labels = np.array(labels)

        print(f"Total gambar diproses: {processed_count}")
        return len(features) > 0

    def train_model(self, test_size=0.2, random_state=42):
        """Training model SVM"""
        if len(self.features) == 0:
            return False, "Dataset belum dimuat atau kosong"

        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                self.features,
                self.labels,
                test_size=test_size,
                random_state=random_state,
                stratify=self.labels,
            )

            # Normalisasi fitur
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Training SVM
            self.svm_model.fit(X_train_scaled, y_train)

            # Prediksi
            y_pred = self.svm_model.predict(X_test_scaled)

            # Evaluasi
            self.training_accuracy = accuracy_score(y_test, y_pred)
            self.is_trained = True

            # Generate confusion matrix plot
            self.generate_confusion_matrix_plot(y_test, y_pred)

            # Save model
            self.save_model()

            return (
                True,
                f"Model berhasil ditraining dengan akurasi: {self.training_accuracy:.4f}",
            )

        except Exception as e:
            return False, f"Error training model: {str(e)}"

    def generate_confusion_matrix_plot(self, y_test, y_pred):
        """Generate confusion matrix plot"""
        try:
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=self.label_names,
                yticklabels=self.label_names,
            )
            plt.title(
                "Confusion Matrix - Deteksi Penyakit Mangga",
                fontsize=14,
                fontweight="bold",
            )
            plt.ylabel("Actual Label")
            plt.xlabel("Predicted Label")
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(
                "static/plots/confusion_matrix.png", dpi=150, bbox_inches="tight"
            )
            plt.close()
        except Exception as e:
            print(f"Error generating confusion matrix: {e}")

    def predict_single_image(self, image_path):
        """Prediksi untuk satu gambar"""
        if not self.is_trained:
            return None, None, None

        try:
            # Preprocessing
            processed_img = self.preprocess_image(image_path)
            if processed_img is None:
                return None, None, None

            # Ekstraksi fitur
            features = self.extract_glcm_features(processed_img)
            if features is None:
                return None, None, None

            features_scaled = self.scaler.transform([features])

            # Prediksi
            prediction = self.svm_model.predict(features_scaled)[0]
            probability = self.svm_model.predict_proba(features_scaled)[0]

            # Generate visualization
            glcm_plot_path = self.generate_glcm_visualization(
                image_path, processed_img, features
            )

            return self.label_names[prediction], probability, glcm_plot_path

        except Exception as e:
            print(f"Error predicting image: {e}")
            return None, None, None

    def generate_glcm_visualization(self, original_image_path, processed_img, features):
        """Generate GLCM visualization"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            # Gambar asli
            original = cv2.imread(original_image_path)
            axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
            axes[0, 0].set_title("Gambar Asli")
            axes[0, 0].axis("off")

            # Gambar preprocessing
            axes[0, 1].imshow(processed_img, cmap="gray")
            axes[0, 1].set_title("Setelah Preprocessing")
            axes[0, 1].axis("off")

            # GLCM visualization
            glcm = graycomatrix(img_as_ubyte(processed_img), [1], [0], levels=256)
            axes[0, 2].imshow(glcm[:, :, 0, 0], cmap="hot")
            axes[0, 2].set_title("GLCM Matrix")
            axes[0, 2].axis("off")

            # Fitur GLCM
            feature_names = [
                "Contrast",
                "Dissimilarity",
                "Homogeneity",
                "Energy",
                "Correlation",
                "ASM",
            ]
            features_reshaped = features.reshape(6, -1).mean(axis=1)

            axes[1, 0].bar(feature_names, features_reshaped)
            axes[1, 0].set_title("Fitur GLCM")
            axes[1, 0].tick_params(axis="x", rotation=45)

            # Histogram gambar
            axes[1, 1].hist(processed_img.flatten(), bins=50, alpha=0.7)
            axes[1, 1].set_title("Histogram Intensitas")
            axes[1, 1].set_xlabel("Intensitas Pixel")
            axes[1, 1].set_ylabel("Frekuensi")

            # Info fitur
            axes[1, 2].axis("off")
            info_text = f"Total fitur: {len(features)}\n"
            for i, name in enumerate(feature_names):
                info_text += f"{name}: {features_reshaped[i]:.4f}\n"
            axes[1, 2].text(
                0.1, 0.5, info_text, fontsize=10, verticalalignment="center"
            )

            plt.tight_layout()
            plot_path = "static/plots/glcm_analysis.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()

            return plot_path
        except Exception as e:
            print(f"Error generating GLCM visualization: {e}")
            return None

    def save_model(self):
        """Save trained model"""
        try:
            model_data = {
                "svm_model": self.svm_model,
                "scaler": self.scaler,
                "is_trained": self.is_trained,
                "training_accuracy": self.training_accuracy,
                "label_names": self.label_names,
            }
            with open("models/mango_disease_model.pkl", "wb") as f:
                pickle.dump(model_data, f)
        except Exception as e:
            print(f"Error saving model: {e}")

    def load_model(self):
        """Load trained model"""
        try:
            if os.path.exists("models/mango_disease_model.pkl"):
                with open("models/mango_disease_model.pkl", "rb") as f:
                    model_data = pickle.load(f)
                    self.svm_model = model_data["svm_model"]
                    self.scaler = model_data["scaler"]
                    self.is_trained = model_data["is_trained"]
                    self.training_accuracy = model_data["training_accuracy"]
                    self.label_names = model_data["label_names"]
                return True
        except Exception as e:
            print(f"Error loading model: {e}")
        return False


# Initialize detector
detector = MangoDiseaseDetector()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/train", methods=["POST"])
def train_model():
    dataset_path = request.json.get("dataset_path", "")

    if not dataset_path or not os.path.exists(dataset_path):
        return jsonify(
            {
                "success": False,
                "message": "Path dataset tidak valid atau tidak ditemukan",
            }
        )

    # Load dataset
    if not detector.load_dataset(dataset_path):
        return jsonify(
            {"success": False, "message": "Gagal memuat dataset atau dataset kosong"}
        )

    # Train model
    success, message = detector.train_model()

    return jsonify(
        {
            "success": success,
            "message": message,
            "accuracy": detector.training_accuracy if success else 0,
            "confusion_matrix_available": success,
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"success": False, "message": "Tidak ada file yang diupload"})

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "message": "Tidak ada file yang dipilih"})

    # Check if model is trained
    if not detector.is_trained:
        # Try to load existing model
        if not detector.load_model():
            return jsonify(
                {
                    "success": False,
                    "message": "Model belum ditraining. Silakan training model terlebih dahulu.",
                }
            )

    if file:
        filename = file.filename
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # Predict
        prediction, probability, glcm_plot_path = detector.predict_single_image(
            file_path
        )

        if prediction is None:
            return jsonify({"success": False, "message": "Gagal memproses gambar"})

        # Prepare probability data
        prob_data = []
        for i, prob in enumerate(probability):
            prob_data.append(
                {
                    "label": detector.label_names[i],
                    "probability": float(prob),
                    "percentage": float(prob * 100),
                }
            )

        # Determine confidence level
        max_prob = np.max(probability)
        if max_prob > 0.8:
            confidence = "Tinggi"
        elif max_prob > 0.6:
            confidence = "Sedang"
        else:
            confidence = "Rendah"

        return jsonify(
            {
                "success": True,
                "prediction": prediction,
                "probabilities": prob_data,
                "confidence": confidence,
                "confidence_score": float(max_prob),
                "glcm_plot": (
                    glcm_plot_path.replace("static/", "") if glcm_plot_path else None
                ),
                "uploaded_image": f"uploads/{filename}",
            }
        )


@app.route("/model_info")
def model_info():
    # Try to load model if not already loaded
    if not detector.is_trained:
        detector.load_model()

    return jsonify(
        {
            "is_trained": detector.is_trained,
            "training_accuracy": detector.training_accuracy,
            "label_names": detector.label_names,
        }
    )


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)


if __name__ == "__main__":
    # Auto-train if dataset exists
    dataset_path = "dataset"  # Change this to your dataset path
    if os.path.exists(dataset_path):
        print("Loading dataset and training model...")
        if detector.load_dataset(dataset_path):
            success, message = detector.train_model()
            print(f"Training result: {message}")
        else:
            print("Failed to load dataset")
    else:
        print(
            f"Dataset path '{dataset_path}' not found. Model will need to be trained manually."
        )

    app.run(debug=True)
