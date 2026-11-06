import os
import re
import base64
import hashlib
import shutil
import uuid
import zipfile
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, Response, flash, redirect, render_template, request, send_file, send_from_directory, session, url_for, jsonify
from flask_bcrypt import Bcrypt
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func
from ultralytics import YOLO
from werkzeug.utils import secure_filename

try:
    import pytesseract
except ImportError:
    pytesseract = None

try:
    import razorpay
except ImportError:
    razorpay = None

try:
    from twilio.rest import Client as TwilioClient
    from twilio.base.exceptions import TwilioRestException
except ImportError:
    TwilioClient = None
    TwilioRestException = Exception

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'best.pt')
PLATE_MODEL_PATH = os.getenv('PLATE_MODEL_PATH', os.path.join(BASE_DIR, 'plate_best.pt'))


def load_dotenv_file(path):
    if not os.path.exists(path):
        return
    with open(path, 'r', encoding='utf-8') as env_file:
        for line in env_file:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


load_dotenv_file(os.path.join(BASE_DIR, '.env'))
load_dotenv_file(os.path.join(BASE_DIR, '.env.example'))

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ['FLASK_SECRET_KEY']
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True

UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

VIDEO_UPLOAD_FOLDER = os.path.join(UPLOAD_FOLDER, 'videos')
VIDEO_RESULTS_FOLDER = os.path.join(VIDEO_UPLOAD_FOLDER, 'results')
for folder in (VIDEO_UPLOAD_FOLDER, VIDEO_RESULTS_FOLDER):
    os.makedirs(folder, exist_ok=True)

EXTRACTED_RESULTS_FOLDER = os.path.join(BASE_DIR, 'extracted_results')
VIOLATION_RESULTS_FOLDER = os.path.join(EXTRACTED_RESULTS_FOLDER, 'violation_images')
PLATE_RESULTS_FOLDER = os.path.join(EXTRACTED_RESULTS_FOLDER, 'number_plates')
for folder in (EXTRACTED_RESULTS_FOLDER, VIOLATION_RESULTS_FOLDER, PLATE_RESULTS_FOLDER):
    os.makedirs(folder, exist_ok=True)

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
ALLOWED_PROGRAMS = {'BTECH', 'MTECH'}
ALLOWED_YEARS = {'1st Year', '2nd Year', '3rd Year', '4th Year'}
ADMIN_USERNAME = os.getenv('ADMIN_USERNAME', 'admin')
ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD', 'admin123')
RAZORPAY_KEY_ID = os.getenv('RAZORPAY_KEY_ID', '').strip()
RAZORPAY_KEY_SECRET = os.getenv('RAZORPAY_KEY_SECRET', '').strip()
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID', '').strip()
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN', '').strip()
TWILIO_FROM_NUMBER = os.getenv('TWILIO_FROM_NUMBER', '').strip()
TWILIO_SMS_FROM_NUMBER = os.getenv('TWILIO_SMS_FROM_NUMBER', '').strip()
TWILIO_WHATSAPP_FROM_NUMBER = os.getenv('TWILIO_WHATSAPP_FROM_NUMBER', '').strip()
TWILIO_CHANNEL = os.getenv('TWILIO_CHANNEL', 'sms').strip().lower()
FINE_MAP = {
    'No Parking': 500,
    'Not Wearing Helmet': 500,
    'Triple Riding': 1000,
    'Usage Of Phone While Riding': 1500,
    'Wheeling': 2000,
}
DETECTION_THRESHOLDS = {
    'No Parking': 0.35,
    'Not Wearing Helmet': 0.25,
    'Triple Riding': 0.45,
    'Usage Of Phone While Riding': 0.30,
    'Wheeling': 0.45,
}
TARGET_CLASSES = tuple(FINE_MAP.keys())
MIN_MODEL_CONFIDENCE = min(DETECTION_THRESHOLDS.values())
PLATE_DETECTION_CONFIDENCE = float(os.getenv('PLATE_DETECTION_CONFIDENCE', '0.25'))
PLATE_LABEL_KEYWORDS = ('plate', 'license', 'licence', 'number')
ENABLE_LEGACY_PLATE_OCR = os.getenv('ENABLE_LEGACY_PLATE_OCR', '').strip().lower() in {'1', 'true', 'yes'}
ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov'}


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(1), nullable=False)
    mobile = db.Column(db.String(15), nullable=False)
    vehicle_number = db.Column(db.String(20), unique=True, nullable=False)
    first_name = db.Column(db.String(100), nullable=True)
    last_name = db.Column(db.String(100), nullable=True)
    college_reg_no = db.Column(db.String(50), nullable=True)
    program = db.Column(db.String(20), nullable=True)
    study_year = db.Column(db.String(20), nullable=True)


class PredictionResults(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_filename = db.Column(db.String(150), nullable=False)
    detected_class = db.Column(db.String(150), nullable=False)
    evidence_hash = db.Column(db.String(64), nullable=True)
    plate_number = db.Column(db.String(20), nullable=False, default='UNKNOWN')
    user_id = db.Column(db.Integer, nullable=True)
    fine_amount = db.Column(db.Integer, nullable=False, default=0)
    status = db.Column(db.String(20), nullable=False, default='PENDING')
    is_seen_by_user = db.Column(db.Boolean, nullable=False, default=False)
    detection_time = db.Column(db.DateTime, default=datetime.utcnow)
    paid_at = db.Column(db.DateTime, nullable=True)
    payment_reference = db.Column(db.String(50), nullable=True)


def normalize_plate_number(value):
    if not value:
        return ''
    return ''.join(ch for ch in value.upper() if ch.isalnum())


def is_valid_plate_number(value):
    return bool(
        re.fullmatch(r'[A-Z0-9]{6,12}', value)
        and any(ch.isalpha() for ch in value)
        and any(ch.isdigit() for ch in value)
    )


def normalize_phone_number(value):
    raw = (value or '').strip()
    digits = ''.join(ch for ch in raw if ch.isdigit())
    if not digits:
        return ''
    if len(digits) == 10:
        return f'+91{digits}'
    if len(digits) == 12 and digits.startswith('91'):
        return f'+{digits}'
    if raw.startswith('+') and 10 <= len(digits) <= 15:
        return f'+{digits}'
    return ''


def is_valid_phone_number(value):
    return bool(re.fullmatch(r'\+[1-9][0-9]{9,14}', value or ''))


def score_plate_candidate(candidate):
    if not candidate:
        return 0
    score = 0
    if any(ch.isalpha() for ch in candidate):
        score += 2
    if any(ch.isdigit() for ch in candidate):
        score += 2
    if re.fullmatch(r'[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}', candidate):
        score += 4
    if 8 <= len(candidate) <= 11:
        score += 1
    return score


def read_plate_text_from_roi(roi):
    if pytesseract is None:
        return ''
    if roi is None or roi.size == 0:
        return ''

    try:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)

        variants = [
            gray,
            cv2.equalizeHist(gray),
            cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                5,
            ),
        ]
        configs = [
            '--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            '--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            '--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        ]

        best_candidate = ''
        best_score = -1

        for variant in variants:
            for cfg in configs:
                text = pytesseract.image_to_string(variant, config=cfg)
                for token in re.findall(r'[A-Z0-9]{6,12}', text.upper()):
                    normalized = normalize_plate_number(token)
                    score = score_plate_candidate(normalized)
                    if score > best_score:
                        best_score = score
                        best_candidate = normalized

        return best_candidate
    except Exception:
        return ''


_plate_model = None
_plate_model_checked = False


def get_plate_model():
    global _plate_model, _plate_model_checked
    if _plate_model_checked:
        return _plate_model

    _plate_model_checked = True
    if not os.path.exists(PLATE_MODEL_PATH):
        return None

    try:
        _plate_model = YOLO(PLATE_MODEL_PATH)
    except Exception:
        _plate_model = None
    return _plate_model


def preprocess_plate_roi(roi):
    if roi is None or roi.size == 0:
        return roi

    target_width = 420
    h, w = roi.shape[:2]
    if w > 0 and w < target_width:
        scale = target_width / float(w)
        roi = cv2.resize(roi, (target_width, int(h * scale)), interpolation=cv2.INTER_CUBIC)

    return roi


def detect_plate_rois_with_model(image):
    plate_model = get_plate_model()
    if plate_model is None or image is None or image.size == 0:
        return []

    try:
        result = plate_model.predict(image, conf=PLATE_DETECTION_CONFIDENCE, verbose=False)[0]
        boxes_obj = result.boxes
        if boxes_obj is None or len(boxes_obj) == 0:
            return []

        h, w = image.shape[:2]
        boxes = boxes_obj.xywh.cpu().numpy()
        class_ids = boxes_obj.cls.cpu().numpy() if boxes_obj.cls is not None else np.array([])
        confidences = boxes_obj.conf.cpu().numpy() if boxes_obj.conf is not None else np.array([])
        names = result.names or {}
        single_class_model = len(names) <= 1
        candidates = []

        for i, box in enumerate(boxes):
            class_id = int(class_ids[i]) if i < len(class_ids) else 0
            label = str(names.get(class_id, class_id)).lower()
            if not single_class_model and not any(keyword in label for keyword in PLATE_LABEL_KEYWORDS):
                continue

            conf = float(confidences[i]) if i < len(confidences) else 0.0
            x1, y1, x2, y2 = clamp_detection_box(box, w, h)
            box_w = max(0, x2 - x1)
            box_h = max(0, y2 - y1)
            if box_w < 20 or box_h < 8:
                continue

            pad_x = int(box_w * 0.16)
            pad_y = int(box_h * 0.35)
            crop_x1 = max(0, x1 - pad_x)
            crop_y1 = max(0, y1 - pad_y)
            crop_x2 = min(w, x2 + pad_x)
            crop_y2 = min(h, y2 + pad_y)
            roi = image[crop_y1:crop_y2, crop_x1:crop_x2]
            if roi.size == 0:
                continue
            candidates.append((conf, preprocess_plate_roi(roi)))

        candidates.sort(key=lambda item: item[0], reverse=True)
        return [roi for _, roi in candidates[:5]]
    except Exception:
        return []


def extract_plate_number(image):
    try:
        h, w, _ = image.shape
        model_rois = detect_plate_rois_with_model(image)
        best_candidate = ''
        best_score = -1
        for roi in model_rois:
            candidate = normalize_plate_number(read_plate_text_from_roi(roi))
            score = score_plate_candidate(candidate)
            if score > best_score:
                best_score = score
                best_candidate = candidate

        if best_candidate and is_valid_plate_number(best_candidate):
            return best_candidate

        if not ENABLE_LEGACY_PLATE_OCR:
            return ''

        lower_half = image[int(h * 0.35):h, 0:w]
        gray = cv2.cvtColor(lower_half, cv2.COLOR_BGR2GRAY)
        blur = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(blur, 70, 200)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

        rois = []
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bh == 0:
                continue
            aspect_ratio = bw / float(bh)
            area = bw * bh
            if 2.0 <= aspect_ratio <= 6.5 and 1200 <= area <= (w * h * 0.2):
                pad_x = int(bw * 0.08)
                pad_y = int(bh * 0.2)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(lower_half.shape[1], x + bw + pad_x)
                y2 = min(lower_half.shape[0], y + bh + pad_y)
                rois.append(lower_half[y1:y2, x1:x2])

        rois.append(lower_half)
        rois.append(image)

        for roi in rois:
            candidate = read_plate_text_from_roi(roi)
            candidate = normalize_plate_number(candidate)
            score = score_plate_candidate(candidate)
            if score > best_score:
                best_score = score
                best_candidate = candidate

        if best_candidate and is_valid_plate_number(best_candidate):
            return best_candidate
    except Exception:
        return ''
    return ''


def get_logged_in_user():
    user_id = session.get('user_id')
    if not user_id:
        return None
    return db.session.get(User, user_id)


def is_admin_logged_in():
    return bool(session.get('is_admin'))


def get_razorpay_client():
    if not razorpay:
        return None
    if not RAZORPAY_KEY_ID or not RAZORPAY_KEY_SECRET:
        return None
    return razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))


def get_twilio_client():
    if not TwilioClient:
        return None
    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
        return None
    return TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)


def get_sms_config_status():
    missing = []
    channel = TWILIO_CHANNEL if TWILIO_CHANNEL in {'sms', 'whatsapp', 'both'} else 'sms'
    if not TwilioClient:
        missing.append('twilio package')
    if not TWILIO_ACCOUNT_SID or not TWILIO_ACCOUNT_SID.startswith('AC') or 'x' in TWILIO_ACCOUNT_SID.lower():
        missing.append('TWILIO_ACCOUNT_SID')
    if not TWILIO_AUTH_TOKEN or TWILIO_AUTH_TOKEN in {'your_twilio_auth_token', 'your_real_auth_token'}:
        missing.append('TWILIO_AUTH_TOKEN')
    sms_from = TWILIO_SMS_FROM_NUMBER or (
        TWILIO_FROM_NUMBER
        if TWILIO_FROM_NUMBER and not TWILIO_FROM_NUMBER.startswith('whatsapp:')
        else ''
    )
    whatsapp_from = TWILIO_WHATSAPP_FROM_NUMBER or (
        TWILIO_FROM_NUMBER
        if TWILIO_FROM_NUMBER.startswith('whatsapp:')
        else ''
    )

    if channel in {'sms', 'both'}:
        if not sms_from or sms_from == '+1234567890' or sms_from.startswith('whatsapp:'):
            missing.append('TWILIO_SMS_FROM_NUMBER')
    if channel in {'whatsapp', 'both'}:
        if not whatsapp_from or not whatsapp_from.startswith('whatsapp:'):
            if not sms_from:
                missing.append('TWILIO_WHATSAPP_FROM_NUMBER')
    return missing


def send_twilio_message(client, channel, to_number, message_body):
    if channel == 'whatsapp':
        from_number = TWILIO_WHATSAPP_FROM_NUMBER or TWILIO_FROM_NUMBER
        if not to_number.startswith('whatsapp:'):
            to_number = f'whatsapp:{to_number}'
    else:
        from_number = TWILIO_SMS_FROM_NUMBER or TWILIO_FROM_NUMBER

    if not from_number:
        return False, f'{channel.title()} failed: sender number is not configured'

    try:
        message = client.messages.create(
            body=message_body,
            from_=from_number,
            to=to_number,
        )
        return True, f'{channel.title()} alert sent ({message.sid})'
    except TwilioRestException as exc:
        detail = exc.msg or str(exc)
        return False, f'{channel.title()} failed: {detail}'
    except Exception as exc:
        return False, f'{channel.title()} failed: {str(exc)}'


def send_violation_sms(user, plate_number, detected_classes):
    if not user:
        return False, ''
    recipient_number = normalize_phone_number(user.mobile)
    if not is_valid_phone_number(recipient_number):
        return False, 'SMS skipped: user phone number is missing/invalid.'
    missing_config = get_sms_config_status()
    if missing_config:
        return False, f'SMS not configured. Missing: {", ".join(missing_config)}.'
    client = get_twilio_client()
    if not client:
        return False, 'SMS not configured on server.'

    total_fine = sum(FINE_MAP.get(c, 0) for c in detected_classes)
    violation_text = ', '.join(detected_classes[:3])
    if len(detected_classes) > 3:
        violation_text += f' +{len(detected_classes) - 3} more'

    message_body = (
        f'ViolaTracker: Violation detected for plate {plate_number}. '
        f'Type: {violation_text}. Pending fine: Rs {total_fine}. '
        'Login to check My Challans.'
    )
    channel = TWILIO_CHANNEL if TWILIO_CHANNEL in {'sms', 'whatsapp', 'both'} else 'sms'
    sms_available = bool(
        TWILIO_SMS_FROM_NUMBER or (TWILIO_FROM_NUMBER and not TWILIO_FROM_NUMBER.startswith('whatsapp:'))
    )
    whatsapp_available = bool(
        TWILIO_WHATSAPP_FROM_NUMBER or (TWILIO_FROM_NUMBER and TWILIO_FROM_NUMBER.startswith('whatsapp:'))
    )

    if channel == 'both':
        channels = []
        if sms_available:
            channels.append('sms')
        if whatsapp_available:
            channels.append('whatsapp')
    elif channel == 'whatsapp':
        channels = ['whatsapp'] if whatsapp_available else []
        if sms_available:
            channels.append('sms')
    else:
        channels = ['sms'] if sms_available else []

    if not channels:
        return False, 'SMS not configured. No usable Twilio sender number is available.'

    send_notes = []
    sent_any = False

    for channel in channels:
        ok, note = send_twilio_message(client, channel, recipient_number, message_body)
        send_notes.append(note)
        sent_any = sent_any or ok
        if ok:
            break

    return sent_any, '; '.join(send_notes) + '.'


def mark_challan_paid(challan, reference_prefix='VTX'):
    challan.status = 'PAID'
    challan.paid_at = datetime.utcnow()
    challan.payment_reference = f'{reference_prefix}-{challan.id}-{uuid.uuid4().hex[:8].upper()}'
    db.session.commit()


def validate_student_profile_fields(first_name, last_name, college_reg_no, program, study_year, vehicle_number, mobile):
    if not first_name or not last_name:
        return 'First name and last name are required.'
    if not college_reg_no:
        return 'College registration number is required.'
    if program not in ALLOWED_PROGRAMS:
        return 'Program must be BTech or MTech.'
    if study_year not in ALLOWED_YEARS:
        return 'Study year must be 1st, 2nd, 3rd, or 4th Year.'
    if not is_valid_plate_number(vehicle_number):
        return 'Vehicle number must contain 6-12 letters/numbers.'
    if not is_valid_phone_number(mobile):
        return 'Phone number must be a valid mobile (for example: +919876543210).'
    return None


def ensure_schema_updates():
    user_columns = {row[1] for row in db.session.execute(db.text('PRAGMA table_info("user")'))}
    if 'vehicle_number' not in user_columns:
        db.session.execute(db.text('ALTER TABLE "user" ADD COLUMN vehicle_number VARCHAR(20)'))
        db.session.execute(db.text('UPDATE "user" SET vehicle_number = "UNASSIGNED" || id WHERE vehicle_number IS NULL'))
    if 'first_name' not in user_columns:
        db.session.execute(db.text('ALTER TABLE "user" ADD COLUMN first_name VARCHAR(100)'))
    if 'last_name' not in user_columns:
        db.session.execute(db.text('ALTER TABLE "user" ADD COLUMN last_name VARCHAR(100)'))
    if 'college_reg_no' not in user_columns:
        db.session.execute(db.text('ALTER TABLE "user" ADD COLUMN college_reg_no VARCHAR(50)'))
    if 'program' not in user_columns:
        db.session.execute(db.text('ALTER TABLE "user" ADD COLUMN program VARCHAR(20)'))
    if 'study_year' not in user_columns:
        db.session.execute(db.text('ALTER TABLE "user" ADD COLUMN study_year VARCHAR(20)'))
    if 'mobile' not in user_columns:
        db.session.execute(db.text('ALTER TABLE "user" ADD COLUMN mobile VARCHAR(15)'))
        db.session.execute(db.text('UPDATE "user" SET mobile = "+910000000000" WHERE mobile IS NULL OR mobile = ""'))

    prediction_columns = {row[1] for row in db.session.execute(db.text('PRAGMA table_info("prediction_results")'))}
    if 'plate_number' not in prediction_columns:
        db.session.execute(db.text("ALTER TABLE \"prediction_results\" ADD COLUMN plate_number VARCHAR(20) DEFAULT 'UNKNOWN'"))
    if 'evidence_hash' not in prediction_columns:
        db.session.execute(db.text('ALTER TABLE "prediction_results" ADD COLUMN evidence_hash VARCHAR(64)'))
    if 'user_id' not in prediction_columns:
        db.session.execute(db.text('ALTER TABLE "prediction_results" ADD COLUMN user_id INTEGER'))
    if 'fine_amount' not in prediction_columns:
        db.session.execute(db.text('ALTER TABLE "prediction_results" ADD COLUMN fine_amount INTEGER DEFAULT 0'))
    if 'status' not in prediction_columns:
        db.session.execute(db.text('ALTER TABLE "prediction_results" ADD COLUMN status VARCHAR(20) DEFAULT "PENDING"'))
    if 'is_seen_by_user' not in prediction_columns:
        db.session.execute(db.text('ALTER TABLE "prediction_results" ADD COLUMN is_seen_by_user BOOLEAN DEFAULT 0'))
    if 'paid_at' not in prediction_columns:
        db.session.execute(db.text('ALTER TABLE "prediction_results" ADD COLUMN paid_at DATETIME'))
    if 'payment_reference' not in prediction_columns:
        db.session.execute(db.text('ALTER TABLE "prediction_results" ADD COLUMN payment_reference VARCHAR(50)'))

    db.session.commit()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password, password):
            session['user_id'] = user.id
            unseen = PredictionResults.query.filter_by(user_id=user.id, is_seen_by_user=False, status='PENDING').count()
            if unseen > 0:
                flash(f'You have {unseen} new pending challan(s).', 'info')
            return redirect(url_for('home'))

        flash('Invalid email or password.', 'danger')

    return render_template('auth.html', show_form='login')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        first_name = request.form.get('first_name', '').strip()
        last_name = request.form.get('last_name', '').strip()
        college_reg_no = request.form.get('college_reg_no', '').strip().upper()
        program = request.form.get('program', '').strip().upper()
        study_year = request.form.get('study_year', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        vehicle_number = normalize_plate_number(request.form.get('vehicle_number', ''))
        mobile = normalize_phone_number(request.form.get('mobile', ''))

        validation_error = validate_student_profile_fields(
            first_name, last_name, college_reg_no, program, study_year, vehicle_number, mobile
        )
        if validation_error:
            flash(validation_error, 'danger')
            return render_template('auth.html', show_form='register')

        if User.query.filter_by(email=email).first():
            flash('Email address already in use. Please choose a different one.', 'danger')
            return render_template('auth.html', show_form='register')

        if User.query.filter_by(college_reg_no=college_reg_no).first():
            flash('College registration number already exists.', 'danger')
            return render_template('auth.html', show_form='register')

        if User.query.filter_by(vehicle_number=vehicle_number).first():
            flash('Vehicle number already registered.', 'danger')
            return render_template('auth.html', show_form='register')

        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return render_template('auth.html', show_form='register')

        if len(password) < 8:
            flash('Password must be at least 8 characters long.', 'danger')
            return render_template('auth.html', show_form='register')

        username = college_reg_no
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(
            username=username,
            email=email,
            password=hashed_password,
            age=18,
            gender='O',
            mobile=mobile,
            vehicle_number=vehicle_number,
            first_name=first_name,
            last_name=last_name,
            college_reg_no=college_reg_no,
            program=program,
            study_year=study_year,
        )
        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful! You can now log in.', 'success')
        return redirect(url_for('login'))

    return render_template('auth.html', show_form='register')


@app.route('/home')
def home():
    logged_user = get_logged_in_user()
    if not logged_user:
        flash('Please log in to continue.', 'info')
        return redirect(url_for('login'))
    pending_count = PredictionResults.query.filter_by(user_id=logged_user.id, status='PENDING').count()
    return render_template('home.html', pending_count=pending_count)


@app.route('/profile', methods=['GET', 'POST'])
def profile():
    logged_user = get_logged_in_user()
    if not logged_user:
        flash('Please log in to continue.', 'info')
        return redirect(url_for('login'))

    if request.method == 'POST':
        first_name = request.form.get('first_name', '').strip()
        last_name = request.form.get('last_name', '').strip()
        college_reg_no = request.form.get('college_reg_no', '').strip().upper()
        program = request.form.get('program', '').strip().upper()
        study_year = request.form.get('study_year', '').strip()
        email = request.form.get('email', '').strip().lower()
        vehicle_number = normalize_plate_number(request.form.get('vehicle_number', ''))
        mobile = normalize_phone_number(request.form.get('mobile', ''))

        validation_error = validate_student_profile_fields(
            first_name, last_name, college_reg_no, program, study_year, vehicle_number, mobile
        )
        if validation_error:
            flash(validation_error, 'danger')
            return render_template('profile.html', user=logged_user)

        existing_email_user = User.query.filter_by(email=email).first()
        if existing_email_user and existing_email_user.id != logged_user.id:
            flash('Email address already in use. Please choose a different one.', 'danger')
            return render_template('profile.html', user=logged_user)

        existing_reg_user = User.query.filter_by(college_reg_no=college_reg_no).first()
        if existing_reg_user and existing_reg_user.id != logged_user.id:
            flash('College registration number already exists.', 'danger')
            return render_template('profile.html', user=logged_user)

        existing_vehicle_user = User.query.filter_by(vehicle_number=vehicle_number).first()
        if existing_vehicle_user and existing_vehicle_user.id != logged_user.id:
            flash('Vehicle number already registered.', 'danger')
            return render_template('profile.html', user=logged_user)

        logged_user.first_name = first_name
        logged_user.last_name = last_name
        logged_user.college_reg_no = college_reg_no
        logged_user.username = college_reg_no
        logged_user.program = program
        logged_user.study_year = study_year
        logged_user.email = email
        logged_user.vehicle_number = vehicle_number
        logged_user.mobile = mobile
        db.session.commit()

        flash('Profile updated successfully.', 'success')
        return redirect(url_for('profile'))

    return render_template('profile.html', user=logged_user)


_model = None


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f'Model file not found: {MODEL_PATH}')
        _model = YOLO(MODEL_PATH)
    return _model


def decode_data_url_image(data_url):
    try:
        if not data_url or ',' not in data_url:
            return None, 'Invalid captured image payload.'
        encoded = data_url.split(',', 1)[1]
        image_bytes = base64.b64decode(encoded, validate=True)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            return None, 'Unable to read captured frame.'
        return image, ''
    except Exception:
        return None, 'Failed to process captured frame.'


def encode_image_data_url(image, quality=78):
    ok, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return ''
    encoded = base64.b64encode(buffer).decode('ascii')
    return f'data:image/jpeg;base64,{encoded}'


def get_image_hash(image):
    ok, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        return ''
    return hashlib.sha256(buffer.tobytes()).hexdigest()


def copy_result_to_exports(result_filename):
    source_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
    export_path = os.path.join(VIOLATION_RESULTS_FOLDER, result_filename)
    if os.path.exists(source_path):
        shutil.copy2(source_path, export_path)
    return export_path


def save_plate_crops_to_exports(image, safe_filename, plate_number='UNKNOWN'):
    rois = detect_plate_rois_with_model(image)
    if not rois:
        return []

    root_name, _ = os.path.splitext(safe_filename)
    plate_label = normalize_plate_number(plate_number) or 'UNKNOWN'
    saved_files = []

    for index, roi in enumerate(rois[:3], start=1):
        if roi is None or roi.size == 0:
            continue
        crop_filename = f'plate_{plate_label}_{root_name}_{index}.jpg'
        crop_path = os.path.join(PLATE_RESULTS_FOLDER, crop_filename)
        if cv2.imwrite(crop_path, roi):
            saved_files.append(crop_filename)

    return saved_files


def clamp_detection_box(box, frame_w, frame_h):
    x_center, y_center, width, height = box
    x1 = max(0, int(x_center - width / 2))
    y1 = max(0, int(y_center - height / 2))
    x2 = min(frame_w - 1, int(x_center + width / 2))
    y2 = min(frame_h - 1, int(y_center + height / 2))
    return x1, y1, x2, y2


def run_violation_detection(image):
    frame_h, frame_w = image.shape[:2]
    frame_area = float(max(1, frame_h * frame_w))
    result = get_model().predict(image, conf=MIN_MODEL_CONFIDENCE, verbose=False)[0]
    boxes_obj = result.boxes
    if boxes_obj is None or len(boxes_obj) == 0:
        return []

    boxes = boxes_obj.xywh.cpu().numpy()
    class_ids = boxes_obj.cls.cpu().numpy() if boxes_obj.cls is not None else np.array([])
    confidences = boxes_obj.conf.cpu().numpy() if boxes_obj.conf is not None else np.array([])
    detections = []

    for i, box in enumerate(boxes):
        if i >= len(class_ids):
            continue
        class_id = int(class_ids[i])
        label_name = result.names.get(class_id, str(class_id))
        conf = float(confidences[i]) if i < len(confidences) else 0.0
        threshold = DETECTION_THRESHOLDS.get(label_name)
        if threshold is None or conf < threshold:
            continue

        x1, y1, x2, y2 = clamp_detection_box(box, frame_w, frame_h)
        box_area_ratio = (max(0, x2 - x1) * max(0, y2 - y1)) / frame_area
        if label_name == 'Usage Of Phone While Riding' and box_area_ratio > 0.72:
            continue

        detections.append({
            'label': label_name,
            'confidence': conf,
            'box': (x1, y1, x2, y2),
        })

    return detections


def unique_detection_classes(detections):
    detected_classes = []
    for detection in detections:
        label_name = detection['label']
        if label_name not in detected_classes:
            detected_classes.append(label_name)
    return detected_classes


def annotate_detections(image, detections):
    for detection in detections:
        x1, y1, x2, y2 = detection['box']
        label_name = detection['label']
        conf = detection['confidence']
        display_label = f'{label_name} {conf:.2f}'

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 80), 2)
        (text_w, text_h), baseline = cv2.getTextSize(display_label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        label_y = max(text_h + baseline + 8, y1)
        bg_x2 = min(image.shape[1] - 1, x1 + text_w + 10)
        bg_y1 = max(0, label_y - text_h - baseline - 8)
        bg_y2 = min(image.shape[0] - 1, label_y + baseline)
        cv2.rectangle(image, (x1, bg_y1), (bg_x2, bg_y2), (0, 220, 80), -1)
        cv2.putText(
            image,
            display_label,
            (x1 + 5, label_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def process_detection_image(image, safe_filename, manual_plate_number=''):
    original_image = image.copy()
    evidence_hash = get_image_hash(original_image)
    detections = run_violation_detection(image)
    annotate_detections(image, detections)
    detected_classes = unique_detection_classes(detections)
    class_detected = bool(detected_classes)

    manual_plate_number = normalize_plate_number(manual_plate_number)
    if manual_plate_number and not is_valid_plate_number(manual_plate_number):
        return None, None, None, None, 'Entered plate number is invalid. Use 6-12 letters/numbers.'

    plate_number = manual_plate_number or extract_plate_number(original_image)
    if not is_valid_plate_number(plate_number):
        plate_number = 'UNKNOWN'

    matched_user = (
        User.query.filter(func.upper(User.vehicle_number) == plate_number).first()
        if plate_number != 'UNKNOWN'
        else None
    )

    result_filename = f'result_{safe_filename}'
    result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
    if not cv2.imwrite(result_image_path, image):
        return None, None, None, None, 'Unable to save detection result image.'
    copy_result_to_exports(result_filename)
    plate_crop_files = save_plate_crops_to_exports(original_image, safe_filename, plate_number)

    if class_detected:
        saved_classes = []
        duplicate_classes = []
        for detected_class in detected_classes:
            duplicate = (
                PredictionResults.query.filter_by(
                    plate_number=plate_number,
                    detected_class=detected_class,
                    evidence_hash=evidence_hash,
                ).first()
                if evidence_hash
                else None
            )
            if duplicate:
                duplicate_classes.append(detected_class)
                continue

            new_prediction = PredictionResults(
                image_filename=result_filename,
                detected_class=detected_class,
                evidence_hash=evidence_hash,
                plate_number=plate_number,
                user_id=matched_user.id if matched_user else None,
                fine_amount=FINE_MAP.get(detected_class, 0),
                status='PENDING',
                is_seen_by_user=False,
            )
            db.session.add(new_prediction)
            saved_classes.append(detected_class)
        db.session.commit()

        if not saved_classes and duplicate_classes:
            msg = 'Duplicate evidence already assigned for this plate and violation.'
        elif matched_user:
            msg = f'Violation assigned to {matched_user.first_name or matched_user.username} ({plate_number}).'
            _, sms_note = send_violation_sms(matched_user, plate_number, saved_classes)
            if sms_note:
                msg = f'{msg} {sms_note}'
        elif plate_number == 'UNKNOWN':
            msg = 'Violation detected but number plate could not be read.'
        else:
            msg = f'Violation detected for plate {plate_number}, but no registered user found.'
        if saved_classes and duplicate_classes:
            msg = f'{msg} Duplicate skipped: {", ".join(duplicate_classes)}.'
    else:
        msg = 'No target class detected.'

    if plate_crop_files:
        msg = f'{msg} Plate crop saved in extracted_results/number_plates.'

    return result_filename, plate_number, matched_user, detected_classes, msg


def process_detection(file_storage, manual_plate_number=''):
    original_filename = secure_filename(file_storage.filename)
    _, ext = os.path.splitext(original_filename)
    if ext.lower() not in {'.jpg', '.jpeg', '.png', '.bmp', '.jfif', '.webp', '.avif'}:
        ext = '.jpg'
    safe_filename = f'upload_{uuid.uuid4().hex}{ext.lower()}'

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
    file_storage.save(file_path)
    image = cv2.imread(file_path)
    if image is None:
        return None, None, None, None, 'Invalid image format.'

    return process_detection_image(image, safe_filename, manual_plate_number)


def process_detection_from_data_url(data_url, manual_plate_number=''):
    image, error = decode_data_url_image(data_url)
    if error:
        return None, None, None, None, error

    filename = f'capture_{uuid.uuid4().hex}.jpg'
    return process_detection_image(image, filename, manual_plate_number)


def is_allowed_video_file(filename):
    _, ext = os.path.splitext(filename or '')
    return ext.lower() in ALLOWED_VIDEO_EXTENSIONS


def process_video_file(input_path, output_path):
    video_capture = cv2.VideoCapture(input_path)
    if not video_capture.isOpened():
        return False, 'Video cannot be opened.'

    writer = None
    try:
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 24.0

        success, frame = video_capture.read()
        if not success or frame is None:
            return False, 'Video has no readable frames.'

        frame_h, frame_w = frame.shape[:2]
        if frame_w <= 0 or frame_h <= 0:
            return False, 'Video frame size is invalid.'

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))
        if not writer.isOpened():
            return False, 'Processed video writer could not be created.'

        while success:
            detections = run_violation_detection(frame)
            annotate_detections(frame, detections)
            writer.write(frame)
            success, frame = video_capture.read()
    except Exception as exc:
        return False, f'Model prediction failed: {str(exc)}'
    finally:
        video_capture.release()
        if writer is not None:
            writer.release()

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        return False, 'Output video was not generated.'

    return True, ''


@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['is_admin'] = True
            flash('Admin login successful.', 'success')
            return redirect(url_for('admin_dashboard'))
        flash('Invalid admin credentials.', 'danger')
    return render_template('admin_login.html')


@app.route('/admin')
@app.route('/admin/dashboard')
def admin_dashboard():
    if not is_admin_logged_in():
        flash('Admin login required.', 'danger')
        return redirect(url_for('admin_login'))

    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    total_detections = PredictionResults.query.count()
    today_detections = PredictionResults.query.filter(PredictionResults.detection_time >= today_start).count()
    pending_count = PredictionResults.query.filter_by(status='PENDING').count()
    paid_count = PredictionResults.query.filter_by(status='PAID').count()
    assigned_count = PredictionResults.query.filter(PredictionResults.user_id.isnot(None)).count()
    unassigned_count = PredictionResults.query.filter(PredictionResults.user_id.is_(None)).count()
    unique_plates = (
        db.session.query(func.count(func.distinct(PredictionResults.plate_number)))
        .filter(PredictionResults.plate_number != 'UNKNOWN')
        .scalar()
        or 0
    )
    total_pending_amount = (
        db.session.query(func.coalesce(func.sum(PredictionResults.fine_amount), 0))
        .filter_by(status='PENDING')
        .scalar()
        or 0
    )

    violation_rows = (
        db.session.query(
            PredictionResults.detected_class,
            func.count(PredictionResults.id),
            func.coalesce(func.sum(PredictionResults.fine_amount), 0),
        )
        .group_by(PredictionResults.detected_class)
        .order_by(func.count(PredictionResults.id).desc())
        .all()
    )
    max_violation_count = max((row[1] for row in violation_rows), default=1)

    recent_detections = (
        db.session.query(PredictionResults, User)
        .outerjoin(User, PredictionResults.user_id == User.id)
        .order_by(PredictionResults.detection_time.desc())
        .limit(8)
        .all()
    )

    return render_template(
        'admin_dashboard.html',
        total_detections=total_detections,
        today_detections=today_detections,
        pending_count=pending_count,
        paid_count=paid_count,
        assigned_count=assigned_count,
        unassigned_count=unassigned_count,
        unique_plates=unique_plates,
        total_pending_amount=total_pending_amount,
        violation_rows=violation_rows,
        max_violation_count=max_violation_count,
        recent_detections=recent_detections,
        twilio_channel=TWILIO_CHANNEL,
    )


@app.route('/admin/export-results')
def admin_export_results():
    if not is_admin_logged_in():
        flash('Admin login required.', 'danger')
        return redirect(url_for('admin_login'))

    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    zip_filename = f'extracted_results_{timestamp}.zip'
    zip_path = os.path.join(EXTRACTED_RESULTS_FOLDER, zip_filename)

    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        if filename.startswith('result_'):
            source_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            export_path = os.path.join(VIOLATION_RESULTS_FOLDER, filename)
            if os.path.isfile(source_path) and not os.path.exists(export_path):
                shutil.copy2(source_path, export_path)

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for folder_name in ('violation_images', 'number_plates'):
            folder_path = os.path.join(EXTRACTED_RESULTS_FOLDER, folder_name)
            for root, _, files in os.walk(folder_path):
                for filename in files:
                    file_path = os.path.join(root, filename)
                    archive.write(file_path, os.path.relpath(file_path, EXTRACTED_RESULTS_FOLDER))

    return send_file(zip_path, as_attachment=True, download_name=zip_filename)


@app.route('/admin/detect', methods=['GET', 'POST'])
def admin_detect():
    if not is_admin_logged_in():
        flash('Admin login required.', 'danger')
        return redirect(url_for('admin_login'))

    if request.method == 'POST':
        uploaded_file = request.files.get('image_file')
        captured_image = request.form.get('captured_image', '')
        manual_plate_number = request.form.get('manual_plate_number', '')
        has_upload = uploaded_file and uploaded_file.filename != ''
        has_capture = bool(captured_image.strip())
        if not has_upload and not has_capture:
            flash('Upload an image or capture from webcam.', 'danger')
            return render_template('admin_detect.html')
        if has_capture:
            result_filename, plate_number, matched_user, detected_classes, detection_message = process_detection_from_data_url(
                captured_image,
                manual_plate_number=manual_plate_number,
            )
        else:
            result_filename, plate_number, matched_user, detected_classes, detection_message = process_detection(
                uploaded_file,
                manual_plate_number=manual_plate_number,
            )
        if not result_filename:
            flash(detection_message, 'danger')
            return render_template('admin_detect.html')

        return render_template(
            'admin_detect.html',
            result_image=url_for('uploaded_file', filename=result_filename),
            detection_message=detection_message,
            detected_plate=plate_number,
            matched_user=matched_user,
            detected_classes=detected_classes,
        )

    return render_template('admin_detect.html')


@app.route('/admin/prediction')
def admin_prediction():
    if not is_admin_logged_in():
        flash('Admin login required.', 'danger')
        return redirect(url_for('admin_login'))
    return render_template('admin_prediction.html')


@app.route('/admin/video_feed')
def admin_video_feed():
    if not is_admin_logged_in():
        flash('Admin login required.', 'danger')
        return redirect(url_for('admin_login'))
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/detect-frame', methods=['POST'])
def detect_frame_api():
    if not is_admin_logged_in():
        return jsonify({'ok': False, 'message': 'Admin login required.'}), 401

    payload = request.get_json(silent=True) or {}
    image, error = decode_data_url_image(payload.get('image', ''))
    if error:
        return jsonify({'ok': False, 'message': error}), 400

    try:
        detections = run_violation_detection(image)
        annotate_detections(image, detections)
        detected_classes = unique_detection_classes(detections)
        encoded_image = encode_image_data_url(image)
        if not encoded_image:
            return jsonify({'ok': False, 'message': 'Unable to encode detection frame.'}), 500

        return jsonify({
            'ok': True,
            'image': encoded_image,
            'detected_classes': detected_classes,
            'detections': [
                {
                    'label': detection['label'],
                    'confidence': round(detection['confidence'], 3),
                    'box': detection['box'],
                }
                for detection in detections
            ],
            'message': ', '.join(detected_classes) if detected_classes else 'No target class detected.',
        })
    except Exception as exc:
        return jsonify({'ok': False, 'message': f'Detection failed: {str(exc)}'}), 500


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    return redirect(url_for('admin_detect'))


@app.route('/upload_video', methods=['GET', 'POST'])
def upload_video():
    if not is_admin_logged_in():
        flash('Admin login required.', 'danger')
        return redirect(url_for('admin_login'))

    result_video = None
    result_video_filename = None
    detection_message = None

    if request.method == 'POST':
        uploaded_file = request.files.get('video_file')
        if not uploaded_file or uploaded_file.filename == '':
            flash('Upload a video file.', 'danger')
            return render_template('upload_video.html')

        original_filename = secure_filename(uploaded_file.filename)
        if not is_allowed_video_file(original_filename):
            flash('Invalid file type. Upload .mp4, .avi, or .mov only.', 'danger')
            return render_template('upload_video.html')

        _, ext = os.path.splitext(original_filename)
        safe_video_filename = f'video_{uuid.uuid4().hex}{ext.lower()}'
        input_path = os.path.join(VIDEO_UPLOAD_FOLDER, safe_video_filename)
        uploaded_file.save(input_path)

        output_filename = f'result_{os.path.splitext(safe_video_filename)[0]}.mp4'
        output_path = os.path.join(VIDEO_RESULTS_FOLDER, output_filename)
        ok, message = process_video_file(input_path, output_path)
        if not ok:
            flash(message, 'danger')
            return render_template('upload_video.html')

        result_video = url_for('uploaded_video_result', filename=output_filename)
        result_video_filename = output_filename
        detection_message = 'Video processed successfully.'

    return render_template(
        'upload_video.html',
        result_video=result_video,
        result_video_filename=result_video_filename,
        detection_message=detection_message,
    )


@app.route('/uploads/videos/results/<filename>')
def uploaded_video_result(filename):
    if not is_admin_logged_in():
        flash('Admin login required.', 'danger')
        return redirect(url_for('admin_login'))
    return send_from_directory(VIDEO_RESULTS_FOLDER, filename)


def generate_frames():
    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 270)

    target_classes = list(TARGET_CLASSES)
    frame_index = 0
    infer_every_n_frames = 3
    jpeg_quality = 60
    plate_extract_interval = 18
    stable_required = 2
    last_detections = []
    stable_violations = set()
    stable_count = {c: 0 for c in target_classes}
    last_live_plate = 'UNKNOWN'

    while True:
        success, frame = video_capture.read()
        if not success:
            break

        if frame_index % infer_every_n_frames == 0:
            last_detections = run_violation_detection(frame)
            current_violations = {detection['label'] for detection in last_detections}

            for violation_name in target_classes:
                if violation_name in current_violations:
                    stable_count[violation_name] = min(stable_count[violation_name] + 1, stable_required + 1)
                else:
                    stable_count[violation_name] = 0

            stable_violations = {v for v, count in stable_count.items() if count >= stable_required}

            # OCR is expensive; run less frequently to improve stream smoothness.
            if frame_index % plate_extract_interval == 0:
                last_live_plate = extract_plate_number(frame) or 'UNKNOWN'

        annotate_detections(frame, last_detections)

        if stable_violations:
            cv2.putText(frame, 'ALERT: VIOLATION DETECTED!', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            stable_text = ', '.join(sorted(stable_violations))
            cv2.putText(frame, stable_text[:65], (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (40, 180, 255), 2)

        cv2.putText(frame, f'Plate: {last_live_plate}', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        frame_index += 1

    video_capture.release()


@app.route('/video_feed')
def video_feed():
    if not get_logged_in_user():
        flash('Please log in to continue.', 'info')
        return redirect(url_for('login'))
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/predictions')
def predictions():
    return redirect(url_for('my_challans'))


@app.route('/my-challans')
def my_challans():
    logged_user = get_logged_in_user()
    if not logged_user:
        flash('Please log in to continue.', 'info')
        return redirect(url_for('login'))

    challans = (
        PredictionResults.query.filter_by(user_id=logged_user.id)
        .order_by(PredictionResults.detection_time.desc())
        .all()
    )
    pending_count = PredictionResults.query.filter_by(user_id=logged_user.id, status='PENDING').count()
    paid_count = PredictionResults.query.filter_by(user_id=logged_user.id, status='PAID').count()
    total_pending_amount = (
        db.session.query(func.coalesce(func.sum(PredictionResults.fine_amount), 0))
        .filter_by(user_id=logged_user.id, status='PENDING')
        .scalar()
    )
    new_count = PredictionResults.query.filter_by(user_id=logged_user.id, is_seen_by_user=False).count()

    if new_count > 0:
        PredictionResults.query.filter_by(user_id=logged_user.id, is_seen_by_user=False).update({'is_seen_by_user': True})
        db.session.commit()

    return render_template(
        'challans.html',
        challans=challans,
        registered_plate=logged_user.vehicle_number,
        pending_count=pending_count,
        paid_count=paid_count,
        total_pending_amount=total_pending_amount,
        new_count=new_count,
    )


@app.route('/challans/<int:challan_id>/pay', methods=['POST'])
def pay_challan(challan_id):
    logged_user = get_logged_in_user()
    if not logged_user:
        flash('Please log in to continue.', 'info')
        return redirect(url_for('login'))

    challan = PredictionResults.query.filter_by(id=challan_id, user_id=logged_user.id).first()
    if not challan:
        flash('Challan not found for this user.', 'danger')
        return redirect(url_for('my_challans'))

    if challan.status == 'PAID':
        flash('This challan is already paid.', 'info')
        return redirect(url_for('my_challans'))

    mark_challan_paid(challan, reference_prefix='LOCAL')

    flash(f'Payment successful for challan #{challan.id}. Ref: {challan.payment_reference}', 'success')
    return redirect(url_for('my_challans'))


@app.route('/challans/<int:challan_id>/create-order', methods=['POST'])
def create_challan_order(challan_id):
    try:
        logged_user = get_logged_in_user()
        if not logged_user:
            return jsonify({'ok': False, 'message': 'Please log in first.'}), 401

        challan = PredictionResults.query.filter_by(id=challan_id, user_id=logged_user.id).first()
        if not challan:
            return jsonify({'ok': False, 'message': 'Challan not found.'}), 404
        if challan.status == 'PAID':
            return jsonify({'ok': False, 'message': 'This challan is already paid.'}), 400

        client = get_razorpay_client()
        if not client:
            return jsonify({
                'ok': True,
                'fallback': True,
                'message': 'Razorpay is not configured. Using local payment fallback.',
                'challan_id': challan.id,
            })

        order_amount = int(challan.fine_amount) * 100
        if order_amount <= 0:
            return jsonify({'ok': False, 'message': 'Invalid challan amount.'}), 400

        order_data = {
            'amount': order_amount,
            'currency': 'INR',
            'receipt': f'challan-{challan.id}-{uuid.uuid4().hex[:8]}',
            'notes': {
                'challan_id': str(challan.id),
                'user_id': str(logged_user.id),
            },
        }
        try:
            order = client.order.create(data=order_data)
        except Exception as exc:
            return jsonify({'ok': False, 'message': f'Razorpay order creation failed: {str(exc)}'}), 400

        return jsonify({
            'ok': True,
            'key': RAZORPAY_KEY_ID,
            'order_id': order.get('id'),
            'amount': order_amount,
            'challan_id': challan.id,
            'name': f'{logged_user.first_name or ""} {logged_user.last_name or ""}'.strip() or logged_user.username,
            'email': logged_user.email,
            'description': f'Challan Payment #{challan.id}',
        })
    except Exception as exc:
        return jsonify({'ok': False, 'message': f'Internal payment error: {str(exc)}'}), 500


@app.route('/challans/<int:challan_id>/pay-local', methods=['POST'])
def pay_challan_local(challan_id):
    logged_user = get_logged_in_user()
    if not logged_user:
        return jsonify({'ok': False, 'message': 'Please log in first.'}), 401

    challan = PredictionResults.query.filter_by(id=challan_id, user_id=logged_user.id).first()
    if not challan:
        return jsonify({'ok': False, 'message': 'Challan not found.'}), 404
    if challan.status == 'PAID':
        return jsonify({'ok': True, 'message': 'Already paid.'})

    mark_challan_paid(challan, reference_prefix='LOCAL')
    return jsonify({'ok': True, 'message': f'Local payment success for challan #{challan.id}.'})


@app.route('/payments/verify', methods=['POST'])
def verify_payment():
    logged_user = get_logged_in_user()
    if not logged_user:
        return jsonify({'ok': False, 'message': 'Please log in first.'}), 401

    client = get_razorpay_client()
    if not client:
        return jsonify({'ok': False, 'message': 'Razorpay is not configured on server.'}), 400

    payload = request.get_json(silent=True) or {}
    challan_id = payload.get('challan_id')
    payment_id = payload.get('razorpay_payment_id')
    order_id = payload.get('razorpay_order_id')
    signature = payload.get('razorpay_signature')

    if not challan_id or not payment_id or not order_id or not signature:
        return jsonify({'ok': False, 'message': 'Missing payment fields.'}), 400

    challan = PredictionResults.query.filter_by(id=challan_id, user_id=logged_user.id).first()
    if not challan:
        return jsonify({'ok': False, 'message': 'Challan not found.'}), 404
    if challan.status == 'PAID':
        return jsonify({'ok': True, 'message': 'Already paid.'})

    try:
        client.utility.verify_payment_signature({
            'razorpay_order_id': order_id,
            'razorpay_payment_id': payment_id,
            'razorpay_signature': signature,
        })
    except Exception:
        return jsonify({'ok': False, 'message': 'Payment signature verification failed.'}), 400

    challan.status = 'PAID'
    challan.paid_at = datetime.utcnow()
    challan.payment_reference = payment_id
    db.session.commit()

    return jsonify({'ok': True, 'message': f'Payment successful for challan #{challan.id}.'})


@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('is_admin', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


@app.route('/admin/logout')
def admin_logout():
    session.pop('is_admin', None)
    flash('Admin logged out.', 'info')
    return redirect(url_for('admin_login'))


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        ensure_schema_updates()
    app.run(debug=False, port=5001)
