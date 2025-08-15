Voici une présentation complète du projet et comment l'exécuter :

### 🚀 **Projet d'Analyse Faciale Intelligente**

Un système complet qui combine :
- **Reconnaissance faciale** en temps réel
- **Analyse des émotions** (joie, colère, tristesse, surprise)
- **Détection d'objets** (téléphone, sac, etc.)
- **Interface professionnelle** avec animations

---

### 📋 **Prérequis**
1. **Matériel** :
   - Webcam fonctionnelle
   - PC/Mac avec GPU recommandé (pour de meilleures performances)

2. **Système** :
   - Python 3.8+
   - Pip installé

---

### 🔧 **Installation**

#### **1. Cloner le dépôt**
```bash
git clone https://github.com/votre-utilisateur/Projet-Analyse-Faciale.git
cd Projet-Analyse-Faciale
```

#### **2. Installer les dépendances**
```bash
pip install -r requirements.txt
```

#### **3. Télécharger les modèles**
- **MediaPipe** : Installé automatiquement via pip
- **YOLOv8** : Téléchargé automatiquement au premier lancement
- **Dlib** (optionnel) :
  ```bash
  wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
  bunzip2 shape_predictor_68_face_landmarks.dat.bz2
  ```

---

### 🖥️ **Exécution**

#### **Lancement simple**
```bash
python main.py
```

#### **Options avancées**
```bash
python main.py \
  --resolution 1280x720 \  # Résolution caméra
  --show-fps \            # Afficher les FPS
  --debug                 # Mode debug
```

---

### 🎮 **Fonctionnement**

1. **Écran de connexion** :
   - Identifiant : `admin`
   - Mot de passe : `admin`

2. **Animation de chargement** :
   - Initialisation des modèles IA
   - Barre de progression animée

3. **Interface principale** :
   - 🔍 **Zone de détection** : Votre visage encadré
   - 😊 **Émotions** : Affichage en temps réel
   - 📊 **Concentration** : Jauge dynamique
   - 🎒 **Objets détectés** : Liste mise à jour

---

### ⚙️ **Structure du code**

```python
# Architecture principale
main.py
├── LoginSystem()          # Gestion connexion
├── AdvancedFaceAnalyzer() # Analyse faciale
├── ObjectDetector()       # Détection YOLO
└── UIRenderer()           # Interface graphique
```

---

### 🛠 **Personnalisation**

1. **Modifier les identifiants** :
   Éditez `config.ini` :
   ```ini
   [LOGIN]
   username = mon_identifiant
   password = mon_mot_de_passe
   ```

2. **Ajouter des objets** :
   Modifiez `FRENCH_TRANSLATIONS` dans le code :
   ```python
   FRENCH_TRANSLATIONS = {
       "new_object": "Nouvel objet en français",
       ...
   }
   ```

---

### 🐛 **Dépannage**

**Problème** : Webcam non détectée  
**Solution** :
```bash
# Vérifier les périphériques
ls /dev/video*
# Forcer l'index caméra
python main.py --camera-index 1
```

**Problème** : Erreur CUDA  
**Solution** :
```bash
pip uninstall torch torchvision
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu117
```

---

### 📊 **Performances**

| Composant | FPS (CPU) | FPS (GPU) |
|-----------|----------|----------|
| Détection faciale | 15-20 | 30-45 |
| Détection objets | 8-12 | 25-35 |
| Interface | 30+ | 60+ |

---

### 📌 **Bonnes pratiques**

1. **Environnement virtuel** :
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

2. **Mise à jour** :
   ```bash
   git pull origin main
   pip install --upgrade -r requirements.txt
   ```

---

### 🎯 **Cas d'usage**

1. **Analyse d'attention** : Mesure de concentration
2. **Sécurité** : Détection d'intrus
3. **Retail** : Analyse des clients
4. **Accessibilité** : Assistance aux malentendants
