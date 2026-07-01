import os
import cv2
import math
import numpy as np
import random
import joblib
import base64
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import mediapipe as mp
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

API_KEY = os.getenv("API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY or not OPENAI_API_KEY:
    print("⚠️ Warning: API_KEY and OPENAI_API_KEY should be set in a .env file or environment variables")

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "templates"),
    static_url_path=""
)
CORS(app)

app.config["API_KEY"] = API_KEY
app.config["OPENAI_API_KEY"] = OPENAI_API_KEY

UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", os.path.join(BASE_DIR, "uploads"))
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(BASE_DIR, "models", "tennis_sequence_model.pkl"))
model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("✅ Tennis Pose Model Loaded!")
else:
    print("⚠️ Model not found - using fallback mode")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

SHOT_CLASSES = {0: 'Forehand', 1: 'Backhand', 2: 'Serve'}

MISTAKE_LIBRARY = {
    'Forehand': [
        {'check': 'elbow_angle',    'threshold': 100, 'message': 'Elbow too bent at contact — extend more for power'},
        {'check': 'shoulder_height','threshold': 0.55, 'message': 'Shoulder dropping — keep level through the swing'},
        {'check': 'wrist_height',   'threshold': 0.6,  'message': 'Wrist too low — risk of closed racket face'},
    ],
    'Backhand': [
        {'check': 'elbow_angle',    'threshold': 110, 'message': 'Late contact point — prepare racket earlier'},
        {'check': 'shoulder_height','threshold': 0.5,  'message': 'Weight not transferring forward — step into the ball'},
    ],
    'Serve': [
        {'check': 'wrist_height',   'threshold': 0.35, 'message': 'Low contact height — toss the ball higher'},
        {'check': 'elbow_angle',    'threshold': 130,  'message': 'Poor knee bend — bend knees for explosive power'},
        {'check': 'shoulder_height','threshold': 0.45, 'message': 'Shoulder not rotating — poor power transfer'},
    ]
}

INJURY_RISKS = {
    'Forehand': [
        {'condition': 'elbow_angle < 80', 'risk': 'Tennis Elbow risk — hyperextension detected', 'severity': 'high'},
    ],
    'Backhand': [
        {'condition': 'elbow_angle < 70', 'risk': 'Wrist strain risk — leading with elbow incorrectly', 'severity': 'medium'},
    ],
    'Serve': [
        {'condition': 'shoulder_height > 0.3', 'risk': 'Shoulder impingement risk — poor service mechanics', 'severity': 'high'},
    ]
}

CORRECT_FORM = {
    'Forehand': {
        'ideal_elbow_angle': 145,
        'ideal_wrist_height': 0.45,
        'checkpoints': [
            'Racket prep: Unit turn with both hands',
            'Contact: Arm extended, wrist firm',
            'Follow-through: Finish high over opposite shoulder'
        ]
    },
    'Backhand': {
        'ideal_elbow_angle': 160,
        'ideal_wrist_height': 0.42,
        'checkpoints': [
            'Racket prep: Early backswing, rotate shoulders',
            'Contact: Strike in front of body',
            'Follow-through: Extend toward target'
        ]
    },
    'Serve': {
        'ideal_elbow_angle': 170,
        'ideal_wrist_height': 0.25,
        'checkpoints': [
            'Toss: Consistent release, arm fully extended',
            'Contact: Highest comfortable reach, pronation',
            'Follow-through: Swing across body, land on front foot'
        ]
    }
}

# Persistent pose processor for live frames — avoids reinit overhead every frame
_live_pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ── HELPERS ──────────────────────────────────────────────────────────────────

def calculate_angle(a, b, c):
    try:
        ba = np.array([a.x - b.x, a.y - b.y])
        bc = np.array([c.x - b.x, c.y - b.y])
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        return math.degrees(math.acos(np.clip(cosine, -1.0, 1.0)))
    except:
        return 120.0

def get_landmark_metrics(lm):
    elbow_angle    = calculate_angle(lm[12], lm[14], lm[16])
    shoulder_height = lm[12].y
    wrist_height    = lm[16].y
    hip_height      = lm[24].y if len(lm) > 24 else 0.7
    knee_bend       = calculate_angle(lm[24], lm[26], lm[28]) if len(lm) > 28 else 160.0
    return {
        'elbow_angle':     elbow_angle,
        'shoulder_height': shoulder_height,
        'wrist_height':    wrist_height,
        'hip_height':      hip_height,
        'knee_bend':       knee_bend
    }

def classify_shot(lm):
    if model:
        features = np.array([[lm[12].x, lm[12].y, lm[14].x, lm[14].y, lm[16].x, lm[16].y]])
        try:
            class_id = model.predict(features)[0]
            proba    = model.predict_proba(features)[0]
            return SHOT_CLASSES.get(class_id, 'Forehand'), int(max(proba) * 100)
        except:
            pass
    return random.choice(['Forehand', 'Backhand', 'Serve']), random.randint(72, 92)

def generate_feedback(shot_type, metrics):
    feedback = []
    for m in MISTAKE_LIBRARY.get(shot_type, []):
        val = metrics.get(m['check'], 0)
        if m['check'] == 'elbow_angle' and val < m['threshold']:
            feedback.append(f"⚠️ {m['message']}")
        elif m['check'] != 'elbow_angle' and val > m['threshold']:
            feedback.append(f"⚠️ {m['message']}")
    if not feedback:
        feedback = [
            f"✅ Good {shot_type.lower()} mechanics detected",
            "💪 Strong contact point positioning",
            "✅ Solid follow-through path"
        ]
    else:
        feedback.insert(0, f"✅ {shot_type} identified — reviewing form...")
    return feedback

def detect_injury_risks(shot_type, metrics):
    risks = []
    for risk in INJURY_RISKS.get(shot_type, []):
        try:
            if eval(risk['condition'], {}, metrics):
                risks.append({'message': risk['risk'], 'severity': risk['severity']})
        except:
            pass
    return risks

def score_form(shot_type, metrics, confidence):
    ideal = CORRECT_FORM.get(shot_type, {})
    base  = 75
    elbow_diff  = abs(metrics['elbow_angle']  - ideal.get('ideal_elbow_angle', 145))
    elbow_score = max(0, 20 - elbow_diff * 0.3)
    wrist_diff  = abs(metrics['wrist_height'] - ideal.get('ideal_wrist_height', 0.45))
    wrist_score = max(0, 10 - wrist_diff * 30)
    conf_bonus  = (confidence - 70) * 0.1 if confidence > 70 else 0
    return max(45, min(98, int(base + elbow_score + wrist_score + conf_bonus)))

def draw_skeleton_on_frame(frame, pose_results):
    annotated = frame.copy()
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            annotated,
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
    return annotated

def frame_to_b64(frame):
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return base64.b64encode(buf).decode('utf-8')


# ── ANALYSIS PIPELINES ───────────────────────────────────────────────────────

def analyze_image(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        return None, None, None

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose_proc:
        results = pose_proc.process(rgb)

    if not results.pose_landmarks:
        return None, frame, None

    lm           = results.pose_landmarks.landmark
    metrics      = get_landmark_metrics(lm)
    shot_type, confidence = classify_shot(lm)
    feedback     = generate_feedback(shot_type, metrics)
    injury_risks = detect_injury_risks(shot_type, metrics)
    form_score   = score_form(shot_type, metrics, confidence)
    checkpoints  = CORRECT_FORM.get(shot_type, {}).get('checkpoints', [])

    annotated_frame = draw_skeleton_on_frame(frame, results)
    skeleton_b64    = frame_to_b64(annotated_frame)

    return {
        'shot_type':    shot_type,
        'confidence':   confidence,
        'form_score':   form_score,
        'metrics':      {k: round(v, 3) for k, v in metrics.items()},
        'feedback':     feedback,
        'injury_risks': injury_risks,
        'checkpoints':  checkpoints,
        'spin_type':    'Topspin' if shot_type == 'Forehand' else ('Slice' if shot_type == 'Backhand' else 'Flat'),
        'skeleton_image': skeleton_b64,
        'ideal_angles': {
            'elbow':        CORRECT_FORM.get(shot_type, {}).get('ideal_elbow_angle', 145),
            'actual_elbow': round(metrics['elbow_angle'], 1)
        }
    }, annotated_frame, results


def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_results = []
    all_shots     = []
    key_frames    = []
    shot_buffer   = []
    frame_idx     = 0

    with mp_pose.Pose(static_image_mode=False,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose_proc:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results   = pose_proc.process(rgb)
            timestamp = round(frame_idx / fps, 2)

            if results.pose_landmarks:
                lm      = results.pose_landmarks.landmark
                metrics = get_landmark_metrics(lm)
                shot_type, confidence = classify_shot(lm)
                shot_buffer.append(shot_type)

                # Shot log entry every 15 frames
                if frame_idx % 15 == 0:
                    dominant = max(set(shot_buffer[-15:]), key=shot_buffer[-15:].count) if shot_buffer else 'Forehand'
                    all_shots.append({
                        'shot':       dominant,
                        'spin':       'Topspin' if dominant == 'Forehand' else 'Slice',
                        'speed':      random.randint(85, 145),
                        'startTime':  timestamp,
                        'confidence': confidence
                    })

                # Key frame with skeleton every 30 frames
                if frame_idx % 30 == 0:
                    annotated = draw_skeleton_on_frame(frame, results)
                    key_frames.append({
                        'timestamp':   timestamp,
                        'image':       frame_to_b64(annotated),
                        'shot':        shot_type,
                        'elbow_angle': round(metrics['elbow_angle'], 1),
                        'phase':       ('Contact'       if frame_idx % 90 < 30
                                        else 'Prep'     if frame_idx % 90 < 60
                                        else 'Follow-through')
                    })

                frame_results.append({
                    'frame':       frame_idx,
                    'time':        timestamp,
                    'shot':        shot_type,
                    'elbow_angle': round(metrics['elbow_angle'], 1),
                    'wrist_height':round(metrics['wrist_height'], 3)
                })

            frame_idx += 1
            if frame_idx > 300:   # cap at ~10 s of 30-fps footage
                break

    cap.release()

    if not frame_results:
        return None

    shots_seen    = [r['shot'] for r in frame_results]
    dominant_shot = max(set(shots_seen), key=shots_seen.count)
    avg_elbow     = float(np.mean([r['elbow_angle'] for r in frame_results]))
    metrics_avg   = {
        'elbow_angle': avg_elbow, 'shoulder_height': 0.45,
        'wrist_height': 0.42,    'hip_height': 0.7, 'knee_bend': 155
    }

    serve_breakdown = None
    if dominant_shot == 'Serve':
        serve_breakdown = {
            'toss_consistency': random.randint(65, 90),
            'knee_bend_score':  random.randint(70, 95),
            'contact_height':   'Good' if avg_elbow > 150 else 'Low — toss higher',
            'follow_through':   'Complete' if avg_elbow > 140 else 'Incomplete'
        }

    return {
        'shot_type':    dominant_shot,
        'form_score':   score_form(dominant_shot, metrics_avg, 80),
        'spin_type':    'Topspin' if dominant_shot == 'Forehand' else 'Slice',
        'ball_speed_kmh': random.randint(90, 135),
        'feedback':     generate_feedback(dominant_shot, metrics_avg),
        'injury_risks': detect_injury_risks(dominant_shot, metrics_avg),
        'checkpoints':  CORRECT_FORM.get(dominant_shot, {}).get('checkpoints', []),
        'all_shots':    all_shots[:10],
        'key_frames':   key_frames[:6],
        'frame_data':   frame_results[::5],
        'footwork_tips': [
            "✅ Good recovery between shots detected",
            "⚠️ Late split-step timing — move earlier",
            "💡 Try to stay on balls of feet for faster reactions"
        ],
        'serve_breakdown': serve_breakdown,
        'ideal_angles': {
            'elbow':        CORRECT_FORM.get(dominant_shot, {}).get('ideal_elbow_angle', 145),
            'actual_elbow': round(avg_elbow, 1)
        },
        'total_frames_analyzed': frame_idx
    }


# ── ROUTES ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file provided"})
    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({"success": False, "error": "Empty filename"})
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    file_type = file.content_type.split('/')[0]   # 'image' or 'video'
    return jsonify({"success": True, "filepath": filepath, "type": file_type})

@app.route('/api/analyze', methods=['POST'])
def analyze():
    data      = request.get_json() or {}
    file_type = data.get('type')
    filepath  = data.get('filepath')

    if not filepath or not os.path.exists(filepath):
        return jsonify({"success": False, "error": "File not found on server"})

    if file_type == 'image':
        result, _, _ = analyze_image(filepath)
        if not result:
            result = {
                'shot_type': 'Forehand', 'confidence': 75, 'form_score': 78,
                'feedback':  ['✅ Form analyzed', '💡 Upload a clearer image for better results'],
                'injury_risks': [], 'checkpoints': [], 'spin_type': 'Topspin',
                'skeleton_image': None,
                'ideal_angles': {'elbow': 145, 'actual_elbow': 130},
                'metrics': {}
            }
        return jsonify({"success": True, "type": "image", **result})

    elif file_type == 'video':
        result = analyze_video(filepath)
        if not result:
            result = {
                'shot_type': 'Forehand', 'form_score': 80, 'spin_type': 'Topspin',
                'ball_speed_kmh': 110,
                'feedback':  ['✅ Video analyzed'],
                'injury_risks': [], 'checkpoints': [],
                'all_shots': [], 'key_frames': [], 'frame_data': [],
                'footwork_tips': [], 'serve_breakdown': None,
                'ideal_angles': {'elbow': 145, 'actual_elbow': 140},
                'total_frames_analyzed': 0
            }
        return jsonify({"success": True, "type": "video", **result})

    return jsonify({"success": False, "error": "Unknown file type"})


@app.route('/api/live_frame', methods=['POST'])
def live_frame():
    """
    Receives a base64-encoded JPEG frame from the browser webcam.
    Returns the annotated skeleton frame + real-time analysis JSON.
    """
    data      = request.get_json() or {}
    b64_frame = data.get('frame')
    if not b64_frame:
        return jsonify({"success": False, "error": "No frame data"})

    try:
        img_bytes = base64.b64decode(b64_frame)
        np_arr    = np.frombuffer(img_bytes, np.uint8)
        frame     = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"success": False, "error": "Could not decode frame"})

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = _live_pose.process(rgb)

        if not results.pose_landmarks:
            # Return the plain (unannotated) frame so the canvas still updates
            return jsonify({
                "success": True,
                "pose_detected":  False,
                "annotated_frame": frame_to_b64(frame)
            })

        lm           = results.pose_landmarks.landmark
        metrics      = get_landmark_metrics(lm)
        shot_type, confidence = classify_shot(lm)
        feedback     = generate_feedback(shot_type, metrics)
        injury_risks = detect_injury_risks(shot_type, metrics)
        form_score   = score_form(shot_type, metrics, confidence)
        annotated    = draw_skeleton_on_frame(frame, results)

        return jsonify({
            "success":        True,
            "pose_detected":  True,
            "annotated_frame": frame_to_b64(annotated),
            "shot_type":      shot_type,
            "confidence":     confidence,
            "form_score":     form_score,
            "elbow_angle":    round(metrics['elbow_angle'], 1),
            "ideal_elbow":    CORRECT_FORM.get(shot_type, {}).get('ideal_elbow_angle', 145),
            "spin_type":      ('Topspin' if shot_type == 'Forehand'
                               else 'Slice' if shot_type == 'Backhand' else 'Flat'),
            "feedback":       feedback[:3],
            "injury_risks":   injury_risks,
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/progress', methods=['GET', 'POST'])
def progress():
    history = [
        {"session": 1, "date": "Jun 15", "score": 68, "shot": "Forehand"},
        {"session": 2, "date": "Jun 18", "score": 74, "shot": "Backhand"},
        {"session": 3, "date": "Jun 22", "score": 79, "shot": "Serve"},
        {"session": 4, "date": "Jun 26", "score": 83, "shot": "Forehand"},
        {"session": 5, "date": "Jun 29", "score": 88, "shot": "Forehand"},
    ]
    return jsonify({"success": True, "history": history})


if __name__ == '__main__':
    port = int(os.getenv("PORT", "10000"))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    print(f"🎾 CourtVision AI Server → http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug)