@st.cache_resource
def load_my_model():
    # 1. Architecture
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    # 2. Scanner tous les dossiers pour trouver le fichier
    possible_paths = [
        "model/model_best.pth",
        "model_best.pth",
        "/app/deepfake-lite/model/model_best.pth"  # Chemin absolu probable sur Streamlit
    ]

    found_path = None
    for p in possible_paths:
        if os.path.exists(p):
            found_path = p
            break

    # 3. Si toujours rien, on cherche récursivement
    if not found_path:
        import glob
        matches = glob.glob("**/model_best.pth", recursive=True)
        if matches:
            found_path = matches[0]

    # 4. Chargement ou Diagnostic
    if found_path:
        st.sidebar.success(f"✅ Modèle chargé depuis : {found_path}")
        model.load_state_dict(torch.load(found_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    else:
        # Affichage du contenu du dossier pour comprendre l'erreur
        st.error("❌ Fichier introuvable après scan complet.")
        st.write("Dossiers vus par l'app :", os.listdir("."))
        if os.path.exists("model"):
            st.write("Contenu du dossier 'model' :", os.listdir("model"))
        return None