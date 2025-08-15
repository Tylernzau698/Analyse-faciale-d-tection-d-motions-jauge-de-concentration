import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import time
from collections import deque
import random

# Initialisation Mediapipe avec configuration avancée
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialisation YOLOv8 avec modèle custom
model = YOLO('yolov8n-seg.pt')  # Version avec segmentation

# Configuration avancée des couleurs
COLOR_PALETTE = {
    'dark_bg': (18, 18, 18),
    'light_bg': (30, 30, 30),
    'primary': (10, 220, 255),
    'secondary': (255, 150, 50),
    'success': (100, 255, 100),
    'warning': (255, 200, 50),
    'danger': (255, 50, 50),
    'text': (240, 240, 240),
    'text_secondary': (180, 180, 180)
}

# Traduction française enrichie
FRENCH_TRANSLATIONS = {
    "person": "une autre personne",
    "cell phone": "un smartphone",
    "handbag": "un sac à main",
    "backpack": "un sac à dos",
    "book": "un livre",
    "laptop": "un ordinateur portable",
    "bottle": "une bouteille",
    "cup": "une tasse",
    "remote": "une télécommande",
    "mouse": "une souris d'ordinateur",
    "keyboard": "un clavier",
    "chair": "une chaise",
    "dining table": "une table",
    "tv": "une télévision",
    "clock": "une horloge",
    "vase": "un vase",
    "scissors": "des ciseaux",
    "teddy bear": "un ours en peluche",
    "hair drier": "un sèche-cheveux",
    "toothbrush": "une brosse à dents"
}

# Configuration des émotions
EMOTION_CONFIG = {
    'happy': {
        'color': (50, 255, 50),
        'icon': '😊',
        'conditions': [
            lambda data: data['mouth_aspect_ratio'] < 0.25,
            lambda data: data['mouth_width'] / data['mouth_height'] > 2.5
        ]
    },
    'angry': {
        'color': (50, 50, 255),
        'icon': '😠',
        'conditions': [
            lambda data: data['brow_eye_distance'] < 15,
            lambda data: data['mouth_aspect_ratio'] > 0.3,
            lambda data: data['mouth_width'] / data['mouth_height'] < 1.8
        ]
    },
    'surprised': {
        'color': (255, 150, 50),
        'icon': '😲',
        'conditions': [
            lambda data: data['brow_eye_distance'] > 35,
            lambda data: data['mouth_aspect_ratio'] > 0.4
        ]
    },
    'sad': {
        'color': (150, 150, 255),
        'icon': '😢',
        'conditions': [
            lambda data: data['brow_eye_distance'] < 18,
            lambda data: data['mouth_corners_angle'] < -10
        ]
    },
    'neutral': {
        'color': (200, 200, 200),
        'icon': '😐',
        'conditions': [
            lambda data: True  # Default
        ]
    }
}

class AdvancedFaceAnalyzer:
    def __init__(self):
        self.emotion_history = deque(maxlen=30)
        self.object_history = deque(maxlen=10)
        self.concentration_history = deque(maxlen=15)
        self.last_emotion_change = time.time()
        self.current_emotion = 'neutral'
        self.emotion_intensity = 0
        self.animation_frame = 0
        self.performance_stats = {
            'fps': 0,
            'detection_time': 0,
            'frame_count': 0,
            'start_time': time.time()
        }

    def update_performance_stats(self, processing_time):
        self.performance_stats['frame_count'] += 1
        elapsed = time.time() - self.performance_stats['start_time']
        self.performance_stats['fps'] = self.performance_stats['frame_count'] / elapsed
        self.performance_stats['detection_time'] = processing_time * 1000  # ms

    def analyze_face(self, landmarks, frame_shape):
        h, w = frame_shape
        analysis_data = {}

        # Extraction des points clés
        left_mouth = self._get_landmark_coords(landmarks, 61, (h, w))
        right_mouth = self._get_landmark_coords(landmarks, 291, (h, w))
        mouth_top = self._get_landmark_coords(landmarks, 13, (h, w))
        mouth_bottom = self._get_landmark_coords(landmarks, 14, (h, w))
        left_brow = self._get_landmark_coords(landmarks, 70, (h, w))
        left_eye = self._get_landmark_coords(landmarks, 33, (h, w))
        right_brow = self._get_landmark_coords(landmarks, 300, (h, w))
        right_eye = self._get_landmark_coords(landmarks, 263, (h, w))
        mouth_left_corner = self._get_landmark_coords(landmarks, 61, (h, w))
        mouth_right_corner = self._get_landmark_coords(landmarks, 291, (h, w))

        # Calcul des métriques
        analysis_data['mouth_width'] = self._euclidean_dist(left_mouth, right_mouth)
        analysis_data['mouth_height'] = self._euclidean_dist(mouth_top, mouth_bottom)
        analysis_data['mouth_aspect_ratio'] = analysis_data['mouth_height'] / analysis_data['mouth_width']
        analysis_data['brow_eye_distance'] = (self._euclidean_dist(left_brow, left_eye) + 
                                            self._euclidean_dist(right_brow, right_eye)) / 2
        analysis_data['mouth_corners_angle'] = self._calculate_mouth_angle(mouth_left_corner, mouth_right_corner, mouth_bottom)
        analysis_data['face_center'] = self._get_face_center(landmarks, (h, w))

        # Détection d'émotion
        detected_emotion, intensity = self._detect_emotion(analysis_data)
        
        # Calcul de la concentration
        frame_center = (w // 2, h // 2)
        dist_to_center = self._euclidean_dist(analysis_data['face_center'], frame_center)
        max_dist = np.sqrt((w//2)**2 + (h//2)**2)
        concentration = int(100 - (dist_to_center / max_dist) * 100)
        
        return {
            'emotion': detected_emotion,
            'intensity': intensity,
            'concentration': concentration,
            'analysis_data': analysis_data
        }

    def _detect_emotion(self, data):
        max_score = 0
        detected_emotion = 'neutral'
        
        for emotion, config in EMOTION_CONFIG.items():
            score = sum(1 for condition in config['conditions'] if condition(data))
            if score > max_score:
                max_score = score
                detected_emotion = emotion
        
        intensity = min(100, max_score * 25)  # Convert score to percentage
        
        # Smooth emotion transitions
        if detected_emotion != self.current_emotion:
            if time.time() - self.last_emotion_change > 1.5:  # Debounce
                self.current_emotion = detected_emotion
                self.last_emotion_change = time.time()
        
        return self.current_emotion, intensity

    def _get_landmark_coords(self, landmarks, idx, shape):
        h, w = shape
        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)
        return (x, y)

    def _euclidean_dist(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def _calculate_mouth_angle(self, left, right, bottom):
        vec_left = np.array(left) - np.array(bottom)
        vec_right = np.array(right) - np.array(bottom)
        cos_angle = np.dot(vec_left, vec_right) / (np.linalg.norm(vec_left) * np.linalg.norm(vec_right))
        return np.degrees(np.arccos(cos_angle)) - 90  # Normalized

    def _get_face_center(self, landmarks, shape):
        h, w = shape
        xs = [l.x * w for l in landmarks]
        ys = [l.y * h for l in landmarks]
        return (int(np.mean(xs)), int(np.mean(ys)))

class UIRenderer:
    def __init__(self):
        self.animation_state = 0
        self.animation_direction = 1
        self.fonts = {
            'title': cv2.FONT_HERSHEY_COMPLEX,
            'header': cv2.FONT_HERSHEY_DUPLEX,
            'body': cv2.FONT_HERSHEY_SIMPLEX,
            'small': cv2.FONT_HERSHEY_PLAIN
        }
        self.ui_elements = []
        self.animation_start_time = time.time()

    def draw_main_ui(self, frame, analyzer, detected_objects, performance):
        h, w = frame.shape[:2]
        
        # Draw dark translucent overlay for UI
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 220), COLOR_PALETTE['dark_bg'], -1)
        cv2.rectangle(overlay, (0, h-150), (w, h), COLOR_PALETTE['dark_bg'], -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw animated header
        self._draw_animated_header(frame, w)
        
        # Draw emotion panel
        self._draw_emotion_panel(frame, analyzer)
        
        # Draw object detection panel
        self._draw_object_panel(frame, w, h, detected_objects)
        
        # Draw performance stats
        self._draw_performance_stats(frame, w, h, performance)
        
        # Draw concentration meter with animation
        self._draw_concentration_meter(frame, w, h, analyzer.concentration_history)
        
        # Draw subtle grid overlay
        self._draw_grid_overlay(frame, w, h)

    def _draw_animated_header(self, frame, w):
        self.animation_state += 0.05 * self.animation_direction
        if self.animation_state > 1 or self.animation_state < 0:
            self.animation_direction *= -1
        
        # Animated gradient
        for i in range(0, w, 2):
            ratio = i / w
            r = int(10 + ratio * 245 * (1 - 0.5 * self.animation_state))
            g = int(10 + (1 - ratio) * 245 * (0.5 + 0.5 * self.animation_state))
            b = 255
            cv2.line(frame, (i, 0), (i, 60), (b, g, r), 2)
        
        # Title text with shadow
        text = "SYSTÈME INTELLIGENT D'ANALYSE FACIALE"
        text_size = cv2.getTextSize(text, self.fonts['title'], 0.8, 2)[0]
        text_x = (w - text_size[0]) // 2
        
        cv2.putText(frame, text, (text_x+2, 42), 
                   self.fonts['title'], 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, text, (text_x, 40), 
                   self.fonts['title'], 0.8, COLOR_PALETTE['primary'], 2, cv2.LINE_AA)

    def _draw_emotion_panel(self, frame, analyzer):
        emotion_config = EMOTION_CONFIG.get(analyzer.current_emotion, EMOTION_CONFIG['neutral'])
        emoji = emotion_config['icon']
        color = emotion_config['color']
        
        # Emotion main card
        cv2.rectangle(frame, (20, 80), (300, 200), COLOR_PALETTE['light_bg'], -1)
        cv2.rectangle(frame, (20, 80), (300, 200), color, 2)
        
        # Emotion text
        emotion_text = f"ÉMOTION: {analyzer.current_emotion.upper()}"
        cv2.putText(frame, emotion_text, (40, 120), 
                   self.fonts['header'], 0.7, COLOR_PALETTE['text'], 1, cv2.LINE_AA)
        
        # Emoji with animation
        emoji_size = 60 + int(10 * np.sin(time.time() * 3))
        cv2.putText(frame, emoji, (220, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3, cv2.LINE_AA)
        
        # Intensity meter
        cv2.putText(frame, f"Intensité: {analyzer.emotion_intensity}%", (40, 160), 
                   self.fonts['body'], 0.6, COLOR_PALETTE['text_secondary'], 1, cv2.LINE_AA)
        self._draw_meter(frame, (40, 170), 200, 8, analyzer.emotion_intensity, color)

    def _draw_object_panel(self, frame, w, h, objects):
        # Object detection panel
        cv2.rectangle(frame, (320, 80), (w-20, 200), COLOR_PALETTE['light_bg'], -1)
        cv2.rectangle(frame, (320, 80), (w-20, 200), COLOR_PALETTE['secondary'], 2)
        
        cv2.putText(frame, "OBJETS DÉTECTÉS:", (340, 110), 
                   self.fonts['header'], 0.6, COLOR_PALETTE['text'], 1, cv2.LINE_AA)
        
        if objects:
            y_offset = 140
            for i, obj in enumerate(objects[:3]):  # Show max 3 objects
                cv2.putText(frame, f"• {obj.capitalize()}", (340, y_offset + i*30), 
                           self.fonts['body'], 0.6, COLOR_PALETTE['text'], 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Aucun objet détecté", (340, 140), 
                       self.fonts['body'], 0.6, COLOR_PALETTE['text_secondary'], 1, cv2.LINE_AA)

    def _draw_performance_stats(self, frame, w, h, stats):
        stats_text = [
            f"FPS: {stats['fps']:.1f}",
            f"Temps détection: {stats['detection_time']:.1f}ms",
            f"Résolution: {w}x{h}"
        ]
        
        for i, text in enumerate(stats_text):
            cv2.putText(frame, text, (w - 250, h - 120 + i*25), 
                       self.fonts['small'], 0.7, COLOR_PALETTE['text_secondary'], 1, cv2.LINE_AA)

    def _draw_concentration_meter(self, frame, w, h, concentration_history):
        if not concentration_history:
            return
            
        avg_concentration = np.mean(concentration_history)
        
        # Main meter
        meter_x, meter_y = 20, h - 120
        meter_w, meter_h = w - 40, 30
        self._draw_meter(frame, (meter_x, meter_y), meter_w, meter_h, avg_concentration, 
                        COLOR_PALETTE['primary'], True)
        
        # Animated wave effect
        wave_height = 5
        points = []
        for x in range(meter_x, meter_x + meter_w + 1, 5):
            y_offset = wave_height * np.sin(x/50 + time.time()*3)
            y = meter_y + meter_h - (meter_h * avg_concentration/100) + y_offset
            points.append((x, int(y)))
        
        if len(points) > 1:
            cv2.polylines(frame, [np.array(points)], False, (255, 255, 255, 100), 1)
        
        # Concentration text
        status = "CONCENTRATION ÉLEVÉE" if avg_concentration > 70 else \
                "CONCENTRATION MOYENNE" if avg_concentration > 40 else \
                "CONCENTRATION FAIBLE"
        
        status_color = COLOR_PALETTE['success'] if avg_concentration > 70 else \
                      COLOR_PALETTE['warning'] if avg_concentration > 40 else \
                      COLOR_PALETTE['danger']
        
        cv2.putText(frame, f"{status} ({avg_concentration:.0f}%)", 
                   (meter_x, meter_y - 10), self.fonts['header'], 0.6, status_color, 1, cv2.LINE_AA)

    def _draw_meter(self, frame, pos, width, height, value, color, rounded=False):
        x, y = pos
        value = max(0, min(100, value))
        fill_width = int(width * value / 100)
        
        if rounded:
            # Draw rounded rectangle background
            radius = height // 2
            cv2.rectangle(frame, (x + radius, y), (x + width - radius, y + height), 
                         COLOR_PALETTE['light_bg'], -1)
            cv2.circle(frame, (x + radius, y + radius), radius, COLOR_PALETTE['light_bg'], -1)
            cv2.circle(frame, (x + width - radius, y + radius), radius, COLOR_PALETTE['light_bg'], -1)
            
            # Draw rounded fill
            if fill_width > 0:
                fill_end = x + fill_width
                if fill_end > x + radius:
                    cv2.rectangle(frame, (x + radius, y), (min(x + width, fill_end), y + height), 
                                 color, -1)
                    if fill_end > x + width - radius:
                        cv2.circle(frame, (x + width - radius, y + radius), radius, color, -1)
                    else:
                        cv2.circle(frame, (fill_end, y + radius), radius, color, -1)
                cv2.circle(frame, (x + radius, y + radius), radius, color, -1)
        else:
            cv2.rectangle(frame, (x, y), (x + width, y + height), COLOR_PALETTE['light_bg'], -1)
            cv2.rectangle(frame, (x, y), (x + fill_width, y + height), color, -1)
        
        cv2.rectangle(frame, (x, y), (x + width, y + height), COLOR_PALETTE['text'], 1)

    def _draw_grid_overlay(self, frame, w, h):
        # Draw subtle grid pattern
        for x in range(0, w, 40):
            alpha = 0.1 + 0.1 * np.sin(x/100 + time.time())
            overlay = frame.copy()
            cv2.line(overlay, (x, 0), (x, h), (255, 255, 255, 50), 1)
            frame[:] = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        for y in range(0, h, 40):
            alpha = 0.1 + 0.1 * np.cos(y/100 + time.time())
            overlay = frame.copy()
            cv2.line(overlay, (0, y), (w, y), (255, 255, 255, 50), 1)
            frame[:] = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

def main():
    analyzer = AdvancedFaceAnalyzer()
    ui_renderer = UIRenderer()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Erreur: Impossible d'accéder à la caméra.")
        return
    
    # Set camera resolution if possible
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Erreur de lecture du flux vidéo.")
            break
        
        # Mirror the frame
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        # Convert to RGB for face mesh
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process face mesh
        face_results = face_mesh.process(rgb_frame)
        
        # Process objects with YOLO
        object_results = model(frame, verbose=False)
        detected_objects = []
        
        for r in object_results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if conf > 0.5:
                    label = r.names[cls]
                    if label in FRENCH_TRANSLATIONS:
                        txt = FRENCH_TRANSLATIONS[label]
                        if txt not in detected_objects:
                            detected_objects.append(txt)
        
        # Analyze face if detected
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                # Draw face mesh (custom style)
                mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style()
                )
                
                # Analyze facial features
                analysis_result = analyzer.analyze_face(face_landmarks.landmark, (h, w))
                analyzer.emotion_history.append(analysis_result['emotion'])
                analyzer.concentration_history.append(analysis_result['concentration'])
                
                # Draw emotion-specific features
                if analysis_result['emotion'] == 'happy':
                    # Highlight smile
                    mouth_points = [
                        analyzer._get_landmark_coords(face_landmarks.landmark, i, (h, w)) 
                        for i in [61, 291, 0, 17]
                    ]
                    cv2.fillPoly(frame, [np.array(mouth_points)], (0, 255, 0, 50))
                
                elif analysis_result['emotion'] == 'angry':
                    # Highlight brows
                    left_brow = analyzer._get_landmark_coords(face_landmarks.landmark, 70, (h, w))
                    right_brow = analyzer._get_landmark_coords(face_landmarks.landmark, 300, (h, w))
                    cv2.line(frame, left_brow, right_brow, (0, 0, 255, 150), 2)
        
        # Update performance stats
        processing_time = time.time() - start_time
        analyzer.update_performance_stats(processing_time)
        
        # Render advanced UI
        ui_renderer.draw_main_ui(frame, analyzer, detected_objects, analyzer.performance_stats)
        
        # Display frame
        cv2.imshow('Analyse Faciale Avancée', frame)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()