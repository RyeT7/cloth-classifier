import streamlit as st
import cv2
import numpy as np
import pickle
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ORBDescriptor:
    def __init__(self, n_features=500):
        self.detector = cv2.ORB_create(nfeatures=n_features)
        self.name = "ORB"
    
    def compute(self, image):
        kp, des = self.detector.detectAndCompute(image, None)
        if des is None:
            des = np.zeros((1, 256), dtype=np.uint8)
        return kp, des

class AKAZEDescriptor:
    def __init__(self):
        self.detector = cv2.AKAZE_create()
        self.name = "AKAZE"
    
    def compute(self, image):
        kp, des = self.detector.detectAndCompute(image, None)
        if des is None:
            des = np.zeros((1, 61), dtype=np.uint8)
        return kp, des

class SIFTDescriptor:
    def __init__(self):
        self.detector = cv2.SIFT_create()
        self.name = "SIFT"
    
    def compute(self, image):
        kp, des = self.detector.detectAndCompute(image, None)
        if des is None:
            des = np.zeros((1, 128), dtype=np.float32)
        return kp, des

class MultiDescriptorFeatureExtractor:
    def __init__(self, image_size=(256, 256)):
        self.descriptors = [
            ORBDescriptor(),
            AKAZEDescriptor(),
            SIFTDescriptor()
        ]
        self.image_size = image_size
        self.vocab_sizes = {'ORB': 100, 'AKAZE': 100, 'SIFT': 100}

    def preprocess_image(self, image_path=None, image_array=None):
        if image_path:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        
        if img is None:
            return None
        img = cv2.resize(img, self.image_size)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        return img

    def extract_features(self, image):
        features_dict = {}
        
        for descriptor in self.descriptors:
            kp, des = descriptor.compute(image)
            
            if des is not None and len(des) > 0:
                if descriptor.name == "SIFT":
                    des = des.astype(np.float32)
                    hist = np.histogram(des.flatten(), bins=50, range=(0, 256))[0]
                else:
                    hist = np.histogram(des.flatten(), bins=50, range=(0, 256))[0]
                
                stats = np.array([
                    len(kp),
                    np.mean(des.flatten()) if des.size > 0 else 0,
                    np.std(des.flatten()) if des.size > 0 else 0,
                    np.min(des.flatten()) if des.size > 0 else 0,
                    np.max(des.flatten()) if des.size > 0 else 0,
                ])
                
                features_dict[descriptor.name] = np.concatenate([hist, stats])
            else:
                default_size = 50 + 5
                features_dict[descriptor.name] = np.zeros(default_size)
        
        return features_dict

    def get_combined_features(self, image):
        features = self.extract_features(image)
        combined = np.concatenate([
            features['ORB'],
            features['AKAZE'],
            features['SIFT']
        ])
        return combined

    def get_sift_features(self, image):
        features = self.extract_features(image)
        return features['SIFT']
    
    def get_akaze_features(self, image):
        features = self.extract_features(image)
        return features['AKAZE']
    
    def get_orb_features(self, image):
        features = self.extract_features(image)
        return features['ORB']
    
    def get_all_features_separated(self, image):
        return self.extract_features(image)

class ClothTypeClassifier:
    def __init__(self, classifier_type='rf'):
        self.feature_extractor = MultiDescriptorFeatureExtractor()
        self.classifier_type = classifier_type
        self.label_mapping = {}
        self.reverse_mapping = {}
        self.is_trained = False
        self.metrics = {}

    @staticmethod
    def load_model(load_path):
        import numpy as np
        from sklearn import __version__ as sklearn_version
        
        def aggressive_tree_patch(tree):
            try:
                if hasattr(tree, 'feature'):
                    n_nodes = len(tree.feature)
                    
                    if hasattr(tree, 'missing_go_to_left'):
                        return tree
                    
                    tree.missing_go_to_left = np.zeros(n_nodes, dtype=np.uint8)
                    
                    if not hasattr(tree, '_is_leaf'):
                        tree._is_leaf = np.zeros(n_nodes, dtype=bool)
                        for i in range(n_nodes):
                            if tree.feature[i] == -2:
                                tree._is_leaf[i] = True
            except Exception:
                pass
            return tree
        
        def patch_estimator(est):
            try:
                if hasattr(est, 'tree_'):
                    aggressive_tree_patch(est.tree_)
                if hasattr(est, 'estimators_'):
                    for e in est.estimators_:
                        if hasattr(e, 'tree_'):
                            aggressive_tree_patch(e.tree_)
            except Exception:
                pass
            return est
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            try:
                with open(load_path, 'rb') as f:
                    model_data = pickle.load(f)
            except (ValueError, TypeError) as e:
                if "incompatible dtype" in str(e) or "missing_go_to_left" in str(e):
                    import io
                    import sys
                    
                    with open(load_path, 'rb') as f:
                        try:
                            model_data = pickle.load(f, encoding='bytes')
                        except:
                            f.seek(0)
                            model_data = pickle.load(f, encoding='latin1')
                else:
                    raise
        
        if 'model' in model_data:
            try:
                model_data['model'] = patch_estimator(model_data['model'])
            except Exception:
                pass
        
        classifier = ClothTypeClassifier(classifier_type=model_data['classifier_type'])
        classifier.model = model_data['model']
        classifier.scaler = model_data['scaler']
        classifier.label_mapping = model_data['label_mapping']
        classifier.reverse_mapping = model_data['reverse_mapping']
        classifier.feature_extractor = MultiDescriptorFeatureExtractor(
            image_size=model_data.get('image_size', (256, 256))
        )
        classifier.feature_type = model_data.get('feature_type', 'sift')
        classifier.metrics = model_data.get('metrics', {})
        classifier.is_trained = True
        return classifier

    def predict(self, image_path=None, image_array=None):
        if not self.is_trained:
            raise ValueError("Model must be trained first!")
        
        img = self.feature_extractor.preprocess_image(image_path=image_path, image_array=image_array)
        if img is None:
            return None, None

        if self.feature_type == 'sift':
            features = self.feature_extractor.get_sift_features(img)
        elif self.feature_type == 'akaze':
            features = self.feature_extractor.get_akaze_features(img)
        elif self.feature_type == 'orb':
            features = self.feature_extractor.get_orb_features(img)
        elif self.feature_type == 'combined':
            features = self.feature_extractor.get_combined_features(img)
        else:
            raise ValueError("Unsupported feature type")
        
        features_scaled = self.scaler.transform([features])
        
        prediction = self.model.predict(features_scaled)[0]
        
        try:
            confidence = np.max(self.model.predict_proba(features_scaled))
        except AttributeError:
            confidence = 0.5
        
        label = self.reverse_mapping[prediction]
        return label, float(confidence)


st.set_page_config(page_title="Cloth Classifier", layout="wide")
st.title("👗 Cloth Type Classifier")
st.write("Upload a cloth image and the model will classify it for you!")

@st.cache_resource
def find_available_models():
    models = {}
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    for file in os.listdir(current_dir):
        if file.startswith('cloth_classifier_') and file.endswith('.pkl'):
            file_path = os.path.join(current_dir, file)
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    classifier = ClothTypeClassifier.load_model(file_path)
                    model_type = 'Random Forest' if 'rf' in file else 'SVM'
                    feature_type = classifier.feature_type
                    models[f"{model_type} ({feature_type})"] = file_path
            except Exception as e:
                pass
    
    return models

available_models = find_available_models()

if not available_models:
    st.error("No trained models found! Please train models first using the notebook.")
    st.info("Make sure you have saved models as 'cloth_classifier_rf_*.pkl' or 'cloth_classifier_svm_*.pkl'")
else:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a cloth image",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload an image of a cloth to classify"
        )
    
    with col2:
        selected_model = st.selectbox(
            "Select Model",
            list(available_models.keys()),
            help="Choose which trained model to use for prediction"
        )
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Uploaded Image")
            image = st.image(uploaded_file, use_column_width=True)
        
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        
        with col2:
            st.subheader("Prediction Results")
            
            try:
                model_path = available_models[selected_model]
                classifier = ClothTypeClassifier.load_model(model_path)
                
                label, confidence = classifier.predict(image_array=image_rgb)
                
                if label:
                    st.metric("Predicted Class", label)
                    
                    if confidence > 0.7:
                        confidence_color = "🟢"
                        status = "High"
                    elif confidence > 0.5:
                        confidence_color = "🟡"
                        status = "Medium"
                    else:
                        confidence_color = "🔴"
                        status = "Low"
                    
                    st.metric("Confidence", f"{confidence:.2%}", delta=status)
                    st.metric("Model Used", selected_model)
                    
                    st.progress(confidence)
                else:
                    st.error("Could not process the image")
            
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
        
        st.subheader("Available Cloth Categories")
        if label:
            classifier = ClothTypeClassifier.load_model(available_models[selected_model])
            categories = sorted(classifier.reverse_mapping.values())
            cols = st.columns(len(categories))
            for col, category in zip(cols, categories):
                with col:
                    st.info(category)

st.divider()
st.caption("Built with Streamlit | Computer Vision Cloth Classifier")
