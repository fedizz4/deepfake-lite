import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Deepfake Detector Pro", page_icon="🕵️")


# --- CHARGEMENT DU MODÈLE PERSONNALISÉ ---
@st.cache_resource
def load_my_model():
    # 1. On définit l'architecture (ResNet18)
    model = models.resnet18(weights=None)

    # 2. On adapte la dernière couche (si tu as fait de la classification binaire)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    # 3. On charge tes meilleurs poids (model_best.pth)
    # Le chemin est relatif au dossier de ton projet
    path = "model/model_best.pth"

    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        model.eval()
        return model
    else:
        st.error(f"Fichier {path} introuvable. Vérifiez le dossier 'model'.")
        return None


model = load_my_model()


# --- TRAITEMENT DE L'IMAGE ---
def predict(image, model):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    input_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        # On récupère la probabilité de la classe "Deepfake" (index 1 généralement)
        confidence = probabilities[1].item()
    return confidence


# --- INTERFACE UTILISATEUR ---
st.title("🛡️ Deepfake Detection System")
st.markdown("---")

uploaded_file = st.file_uploader("Importer une image pour analyse...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    img = Image.open(uploaded_file).convert('RGB')

    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Image à analyser")

    with col2:
        if st.button("Lancer le diagnostic"):
            with st.spinner("Analyse des pixels en cours..."):
                score = predict(img, model)

                st.subheader("Résultat de l'IA")
                if score > 0.5:
                    st.error(f"VERDICT : DEEPFAKE ({score * 100:.1f}%)")
                    st.warning("Des artefacts de génération artificielle ont été détectés.")
                else:
                    st.success(f"VERDICT : AUTHENTIQUE ({(1 - score) * 100:.1f}%)")
                    st.info("Aucune trace de manipulation IA détectée.")

st.markdown("---")
st.caption("Projet IA - Déploiement Cloud via Streamlit - Modèle : ResNet18 Custom")