from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth
from ultralytics import YOLO
from passlib.context import CryptContext
import cv2
import numpy as np
import base64
import os
import json
import re
import random
import smtplib
import uuid
import time
import tempfile
from email.mime.text import MIMEText
from datetime import datetime
from jinja2 import Template

os.environ['YOLO_CONFIG_DIR'] = tempfile.gettempdir()

app = FastAPI()

# Needed so we can remember the logged-in user between requests (via a signed cookie)
app.add_middleware(SessionMiddleware, secret_key=os.environ.get("SESSION_SECRET", "change-this-secret"))

# ---- Google OAuth setup ----
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")

oauth = OAuth()
oauth.register(
    name='google',
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'},
)

# ---- Email (Gmail SMTP) setup ----
GMAIL_ADDRESS = os.environ.get("GMAIL_ADDRESS", "")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "")


def send_verification_email(to_email, code):
    msg = MIMEText(f"Your SudanScan verification code is: {code}\n\nEnter this code on the verification page to activate your account.")
    msg['Subject'] = 'SudanScan - Verify your email'
    msg['From'] = GMAIL_ADDRESS
    msg['To'] = to_email

    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
        server.send_message(msg)

# دالة إرسال رابط استعادة كلمة المرور الجديدة
def send_reset_email(to_email, reset_link):
    msg = MIMEText(f"You requested to reset your password.\n\nClick the link below to set a new password:\n{reset_link}\n\nIf you did not request this, please ignore this email.")
    msg['Subject'] = 'SudanScan - Reset Password'
    msg['From'] = GMAIL_ADDRESS
    msg['To'] = to_email

    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
        server.send_message(msg)


food_model = None
cloth_model = None


def get_food_model():
    global food_model
    if food_model is None:
        food_model = YOLO('Sudanese-food-detection.pt')
    return food_model


def get_cloth_model():
    global cloth_model
    if cloth_model is None:
        cloth_model = YOLO('best.pt')
    return cloth_model


# التعديل الأساسي هنا: توجيه كل البيانات لمجلد data
DATA_DIR = 'data'
os.makedirs(DATA_DIR, exist_ok=True)

UPLOAD_FOLDER = os.path.join(DATA_DIR, 'uploaded_images')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

RESULTS_FILE = os.path.join(DATA_DIR, 'detection_results.json')
USERS_FILE = os.path.join(DATA_DIR, 'users.json')
TOKENS_FILE = os.path.join(DATA_DIR, 'tokens.json') # ملف حفظ رموز استعادة الباسوورد

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def is_valid_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

# دالة التحقق من قوة كلمة المرور
def is_strong_password(password):
    # الشروط: 8 خانات على الأقل، حرف إنجليزي كبير، حرف صغير، ورقم
    pattern = r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)[a-zA-Z\d@$!%*?&#]{8,}$'
    return re.match(pattern, password) is not None


def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def save_users(users):
    with open(USERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(users, f, indent=2, ensure_ascii=False)


def find_user(email):
    for u in load_users():
        if u['email'] == email:
            return u
    return None

# دوال التعامل مع الرموز السرية للاستعادة
def load_tokens():
    if os.path.exists(TOKENS_FILE):
        with open(TOKENS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_tokens(tokens):
    with open(TOKENS_FILE, 'w', encoding='utf-8') as f:
        json.dump(tokens, f, indent=2, ensure_ascii=False)


def save_result_to_json(image_name, model_type, detections):
    record = {
        'image': image_name,
        'model': model_type,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'detections': detections
    }
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
    else:
        all_results = []
    all_results.append(record)
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)


def resize_if_large(img, max_dimension=640):
    h, w = img.shape[:2]
    if max(h, w) > max_dimension:
        scale = max_dimension / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h))
    return img


def run_detection(model, img):
    img = resize_if_large(img)
    results = model.predict(img, conf=0.5, verbose=False, device='cpu', imgsz=640)
    annotated_img = results[0].plot()

    detections = []
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0]) * 100
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        if confidence >= 80:
            conf_class = 'high-conf'
        elif confidence >= 50:
            conf_class = 'mid-conf'
        else:
            conf_class = 'low-conf'

        detections.append({
            'name': class_name,
            'confidence': round(confidence, 1),
            'conf_class': conf_class,
            'box': {
                'x1': round(x1, 1), 'y1': round(y1, 1),
                'x2': round(x2, 1), 'y2': round(y2, 1)
            }
        })

    return annotated_img, detections


BASE_STYLE = """
:root {
  --sand: #f2e6d3;
  --sand-dark: #e6d3b3;
  --clay: #b5652d;
  --clay-dark: #8a4a1f;
  --coffee: #4a3222;
  --coffee-light: #6b4a35;
  --gold: #c9932f;
}
body {
  font-family: 'Georgia', 'Amiri', serif;
  background: var(--sand);
  color: var(--coffee);
  margin: 0;
}
"""

PATTERN_BG = """
background-image:
  radial-gradient(circle at 20% 20%, rgba(181,101,45,0.08) 0, transparent 40%),
  radial-gradient(circle at 80% 80%, rgba(201,147,47,0.10) 0, transparent 40%),
  repeating-linear-gradient(45deg, rgba(74,50,34,0.04) 0 2px, transparent 2px 26px),
  repeating-linear-gradient(-45deg, rgba(74,50,34,0.04) 0 2px, transparent 2px 26px);
"""


LOGIN_PAGE = """
<!DOCTYPE html>
<html>
<head>
<title>SudanScan - Login</title>
<style>
""" + BASE_STYLE + """
.hero {
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  """ + PATTERN_BG + """
}
.card {
  background: #fffaf2;
  border: 1px solid var(--sand-dark);
  border-top: 4px solid var(--clay);
  border-radius: 14px;
  padding: 36px 32px;
  width: 300px;
  box-sizing: border-box;
}
.brand { text-align: center; margin-bottom: 6px; }
.brand-icon { font-size: 30px; color: var(--clay); }
h2 { text-align: center; margin: 6px 0 2px; font-weight: normal; color: var(--coffee); }
.subtitle { text-align: center; color: var(--coffee-light); font-size: 13px; margin: 0 0 24px; }
input {
  width: 100%; padding: 11px 12px; margin: 8px 0;
  border-radius: 8px; border: 1px solid var(--sand-dark);
  background: #fff; color: var(--coffee); box-sizing: border-box;
  font-family: inherit; font-size: 14px;
}
input:focus { outline: none; border-color: var(--clay); }
button {
  width: 100%; padding: 11px; margin-top: 12px;
  background: var(--clay); color: #fff; border: none;
  border-radius: 8px; font-size: 14px; cursor: pointer;
  font-family: inherit;
}
button:hover { background: var(--clay-dark); }
.error { color: #a3372d; text-align: center; font-size: 13px; margin-top: 10px; }
.link { text-align: center; margin-top: 16px; font-size: 13px; color: var(--coffee-light); }
.link a { color: var(--clay); text-decoration: none; }
</style>
</head>
<body>
<div class="hero">
  <div class="card">
    <div class="brand"><i class="ti ti-viewfinder brand-icon"></i></div>
    <h2>SudanScan</h2>
    <p class="subtitle">Sudanese Food & Cloth Detection System</p>
    <form method="post" action="/login">
      <input type="email" name="email" placeholder="Email" required>
      <input type="password" name="password" placeholder="Password" required>
      <button type="submit">Login</button>
    </form>
    <div style="text-align: right; margin-top: 5px;">
        <a href="/forgot" style="font-size: 12px; color: var(--clay); text-decoration: none;">Forgot password?</a>
    </div>
    <div style="text-align:center; margin: 14px 0; color: var(--coffee-light); font-size: 12px;">or</div>
    <a href="/auth/google" style="display:block; text-align:center; padding:11px; border-radius:8px; border:1px solid var(--sand-dark); color: var(--coffee); text-decoration:none; font-size:14px; background:#fff;">Continue with Google</a>
    {% if error %}<p class="error">{{ error }}</p>{% endif %}
    <p class="link">Don't have an account? <a href="/signup">Create one</a></p>
  </div>
</div>
</body>
</html>
"""

SIGNUP_PAGE = """
<!DOCTYPE html>
<html>
<head>
<title>SudanScan - Sign up</title>
<style>
""" + BASE_STYLE + """
.hero { min-height: 100vh; display: flex; align-items: center; justify-content: center; """ + PATTERN_BG + """ }
.card {
  background: #fffaf2; border: 1px solid var(--sand-dark);
  border-top: 4px solid var(--clay); border-radius: 14px;
  padding: 36px 32px; width: 300px; box-sizing: border-box;
}
h2 { text-align: center; margin: 0 0 20px; font-weight: normal; color: var(--coffee); }
input {
  width: 100%; padding: 11px 12px; margin: 8px 0; border-radius: 8px;
  border: 1px solid var(--sand-dark); background: #fff; color: var(--coffee);
  box-sizing: border-box; font-family: inherit; font-size: 14px;
}
input:focus { outline: none; border-color: var(--clay); }
button {
  width: 100%; padding: 11px; margin-top: 12px; background: var(--clay);
  color: #fff; border: none; border-radius: 8px; font-size: 14px;
  cursor: pointer; font-family: inherit;
}
button:hover { background: var(--clay-dark); }
.error { color: #a3372d; text-align: center; font-size: 13px; margin-top: 10px; }
.link { text-align: center; margin-top: 16px; font-size: 13px; color: var(--coffee-light); }
.link a { color: var(--clay); text-decoration: none; }
</style>
</head>
<body>
<div class="hero">
  <div class="card">
    <h2>Create account</h2>
    <form method="post" action="/signup">
      <input type="email" name="email" placeholder="Email" required>
      <input type="password" name="password" placeholder="Password" required>
      <button type="submit">Sign up</button>
    </form>
    {% if error %}<p class="error">{{ error }}</p>{% endif %}
    <p class="link">Already have an account? <a href="/">Login</a></p>
  </div>
</div>
</body>
</html>
"""

VERIFY_PAGE = """
<!DOCTYPE html>
<html>
<head>
<title>SudanScan - Verify email</title>
<style>
""" + BASE_STYLE + """
.hero { min-height: 100vh; display: flex; align-items: center; justify-content: center; """ + PATTERN_BG + """ }
.card {
  background: #fffaf2; border: 1px solid var(--sand-dark);
  border-top: 4px solid var(--clay); border-radius: 14px;
  padding: 36px 32px; width: 300px; box-sizing: border-box;
}
h2 { text-align: center; margin: 0 0 6px; font-weight: normal; color: var(--coffee); }
.subtitle { text-align: center; color: var(--coffee-light); font-size: 13px; margin: 0 0 20px; }
input {
  width: 100%; padding: 11px 12px; margin: 8px 0; border-radius: 8px;
  border: 1px solid var(--sand-dark); background: #fff; color: var(--coffee);
  box-sizing: border-box; font-family: inherit; font-size: 20px; text-align: center;
  letter-spacing: 6px;
}
input:focus { outline: none; border-color: var(--clay); }
button {
  width: 100%; padding: 11px; margin-top: 12px; background: var(--clay);
  color: #fff; border: none; border-radius: 8px; font-size: 14px;
  cursor: pointer; font-family: inherit;
}
button:hover { background: var(--clay-dark); }
.error { color: #a3372d; text-align: center; font-size: 13px; margin-top: 10px; }
.link { text-align: center; margin-top: 16px; font-size: 13px; color: var(--coffee-light); }
.link a { color: var(--clay); text-decoration: none; }
</style>
</head>
<body>
<div class="hero">
  <div class="card">
    <h2>Verify your email</h2>
    <p class="subtitle">We sent a 6-digit code to {{ email }}</p>
    <form method="post" action="/verify">
      <input type="hidden" name="email" value="{{ email }}">
      <input type="text" name="code" placeholder="000000" maxlength="6" required>
      <button type="submit">Verify</button>
    </form>
    {% if error %}<p class="error">{{ error }}</p>{% endif %}
    <p class="link">Didn't get a code? <a href="/resend?email={{ email }}">Resend</a></p>
  </div>
</div>
</body>
</html>
"""

FORGOT_PAGE = """
<!DOCTYPE html>
<html>
<head>
<title>SudanScan - Forgot Password</title>
<style>
""" + BASE_STYLE + """
.hero { min-height: 100vh; display: flex; align-items: center; justify-content: center; """ + PATTERN_BG + """ }
.card { background: #fffaf2; border: 1px solid var(--sand-dark); border-top: 4px solid var(--clay); border-radius: 14px; padding: 36px 32px; width: 300px; box-sizing: border-box; }
h2 { text-align: center; margin: 0 0 10px; font-weight: normal; color: var(--coffee); }
p { text-align: center; font-size: 13px; color: var(--coffee-light); }
input { width: 100%; padding: 11px 12px; margin: 8px 0; border-radius: 8px; border: 1px solid var(--sand-dark); background: #fff; color: var(--coffee); box-sizing: border-box; font-family: inherit; font-size: 14px; }
input:focus { outline: none; border-color: var(--clay); }
button { width: 100%; padding: 11px; margin-top: 12px; background: var(--clay); color: #fff; border: none; border-radius: 8px; font-size: 14px; cursor: pointer; font-family: inherit; }
button:hover { background: var(--clay-dark); }
.message { color: #3b6d11; text-align: center; font-size: 13px; margin-top: 10px; }
.link { text-align: center; margin-top: 16px; font-size: 13px; color: var(--coffee-light); }
.link a { color: var(--clay); text-decoration: none; }
</style>
</head>
<body>
<div class="hero">
  <div class="card">
    <h2>Reset Password</h2>
    <p>Enter your email to receive a reset link.</p>
    <form method="post" action="/forgot">
      <input type="email" name="email" placeholder="Email" required>
      <button type="submit">Send Link</button>
    </form>
    {% if message %}<p class="message">{{ message }}</p>{% endif %}
    <p class="link"><a href="/">&larr; Back to login</a></p>
  </div>
</div>
</body>
</html>
"""

RESET_PAGE = """
<!DOCTYPE html>
<html>
<head>
<title>SudanScan - New Password</title>
<style>
""" + BASE_STYLE + """
.hero { min-height: 100vh; display: flex; align-items: center; justify-content: center; """ + PATTERN_BG + """ }
.card { background: #fffaf2; border: 1px solid var(--sand-dark); border-top: 4px solid var(--clay); border-radius: 14px; padding: 36px 32px; width: 300px; box-sizing: border-box; }
h2 { text-align: center; margin: 0 0 20px; font-weight: normal; color: var(--coffee); }
input { width: 100%; padding: 11px 12px; margin: 8px 0; border-radius: 8px; border: 1px solid var(--sand-dark); background: #fff; color: var(--coffee); box-sizing: border-box; font-family: inherit; font-size: 14px; }
input:focus { outline: none; border-color: var(--clay); }
button { width: 100%; padding: 11px; margin-top: 12px; background: var(--clay); color: #fff; border: none; border-radius: 8px; font-size: 14px; cursor: pointer; font-family: inherit; }
button:hover { background: var(--clay-dark); }
.error { color: #a3372d; text-align: center; font-size: 13px; margin-top: 10px; }
</style>
</head>
<body>
<div class="hero">
  <div class="card">
    <h2>Enter New Password</h2>
    {% if error %}
      <p class="error">{{ error }}</p>
      <p style="text-align:center;"><a href="/forgot" style="color:var(--clay);">Request a new link</a></p>
    {% else %}
      <form method="post" action="/reset">
        <input type="hidden" name="token" value="{{ token }}">
        <input type="password" name="new_password" placeholder="New Password" required>
        <button type="submit">Update Password</button>
      </form>
    {% endif %}
  </div>
</div>
</body>
</html>
"""

MENU_PAGE = """
<!DOCTYPE html>
<html>
<head>
<title>SudanScan - Choose detector</title>
<style>
""" + BASE_STYLE + """
.hero { min-height: 100vh; display: flex; align-items: center; justify-content: center; }
.wrap { text-align: center; }
h2 { font-weight: normal; margin-bottom: 6px; color: var(--coffee); }
.subtitle { color: var(--coffee-light); font-size: 13px; margin: 0 0 28px; }
.cards { display: flex; gap: 16px; justify-content: center; }
a.card {
  display: block; background: #fffaf2; border: 1px solid var(--sand-dark);
  border-radius: 14px; padding: 32px 40px; text-decoration: none;
  color: var(--coffee); width: 150px; transition: border-color 0.15s;
}
a.card:hover { border-color: var(--clay); }
a.card i { font-size: 26px; color: var(--clay); }
a.card p { margin: 12px 0 0; font-size: 14px; }
</style>
</head>
<body>
<div class="hero">
  <div class="wrap">
    <h2>Choose a detector</h2>
    <p class="subtitle">SudanScan</p>
    <div class="cards">
      <a class="card" href="/food"><i class="ti ti-soup"></i><p>Food detector</p></a>
      <a class="card" href="/cloth"><i class="ti ti-shirt"></i><p>Cloth detector</p></a>
    </div>
  </div>
</div>
</body>
</html>
"""

DETECTOR_PAGE = """
<!DOCTYPE html>
<html>
<head>
<title>{{ title }}</title>
<style>
""" + BASE_STYLE + """
.container { max-width: 520px; margin: 0 auto; padding: 40px 20px; }
.back { display: block; text-align: center; color: var(--clay); margin-bottom: 20px; text-decoration: none; font-size: 13px; }
h1 { text-align: center; margin-bottom: 4px; font-weight: normal; color: var(--coffee); }
.subtitle { text-align: center; color: var(--coffee-light); font-size: 14px; margin-bottom: 24px; }
.upload-box { background: #fffaf2; border: 1.5px dashed var(--sand-dark); border-radius: 12px; padding: 30px 20px; text-align: center; }
.upload-box i { font-size: 24px; color: var(--clay); }
.btn { width: 100%; margin-top: 14px; background: var(--clay); color: #fff; border: none; border-radius: 8px; height: 44px; font-size: 15px; cursor: pointer; font-family: inherit; }
.btn:hover { background: var(--clay-dark); }
.result-label { font-size: 13px; color: var(--coffee-light); margin: 24px 0 8px; }
.frame { position: relative; border-radius: 12px; overflow: hidden; border: 1px solid var(--sand-dark); }
.result-img { width: 100%; display: block; }
.detections { margin-top: 16px; background: #fffaf2; border: 1px solid var(--sand-dark); border-radius: 12px; padding: 12px 16px; }
.det-row { display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid var(--sand); font-size: 14px; }
.det-row:last-child { border-bottom: none; }
.det-coords { font-size: 11px; color: var(--coffee-light); margin-top: 2px; }
.badge { font-size: 12px; font-weight: bold; padding: 2px 10px; border-radius: 999px; }
.high-conf { background: #e4ecd8; color: #3b6d11; }
.mid-conf { background: #faeeda; color: #854f0b; }
.low-conf { background: #fcebeb; color: #a32d2d; }
</style>
</head>
<body>
<div class="container">
  <a class="back" href="/menu">&larr; Back to menu</a>
  <h1>{{ title }}</h1>
  <p class="subtitle">{{ subtitle }}</p>

  <form method="POST" enctype="multipart/form-data">
    <div class="upload-box">
      <i class="ti ti-viewfinder"></i>
      <div><input type="file" name="image" accept="image/*" required></div>
    </div>
    <button type="submit" class="btn">Detect</button>
  </form>

  {% if result_image %}
    <p class="result-label">Result</p>
    <div class="frame"><img class="result-img" src="data:image/jpeg;base64,{{ result_image }}"></div>
    <div class="detections">
      <p class="result-label" style="margin:0 0 10px;">Detections</p>
      {% if detections %}
        {% for d in detections %}
          <div class="det-row">
            <div>
              <div>{{ d.name }}</div>
              <div class="det-coords">x1:{{ d.box.x1 }} y1:{{ d.box.y1 }} x2:{{ d.box.x2 }} y2:{{ d.box.y2 }}</div>
            </div>
            <span class="badge {{ d.conf_class }}">{{ d.confidence }}%</span>
          </div>
        {% endfor %}
      {% else %}
        <div class="det-row"><span>No items detected</span></div>
      {% endif %}
    </div>
  {% endif %}
</div>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def login_page():
    return Template(LOGIN_PAGE).render(error=None)


@app.post("/login", response_class=HTMLResponse)
def login(request: Request, email: str = Form(...), password: str = Form(...)):
    user = find_user(email)
    if user and user.get('password_hash') and pwd_context.verify(password, user['password_hash']):
        if not user.get('verified', False):
            return RedirectResponse(url=f"/verify?email={email}", status_code=303)
        request.session['user_email'] = email
        return RedirectResponse(url="/menu", status_code=303)
    return Template(LOGIN_PAGE).render(error="Invalid email or password")


@app.get("/auth/google")
async def auth_google(request: Request):
    redirect_uri = request.url_for('auth_google_callback')
    return await oauth.google.authorize_redirect(request, redirect_uri)


@app.get("/auth/callback")
async def auth_google_callback(request: Request):
    token = await oauth.google.authorize_access_token(request)
    user_info = token.get('userinfo')
    email = user_info['email']

    # Create the account automatically the first time this Google email logs in
    # Google already verified the email, so mark it as verified
    if not find_user(email):
        users = load_users()
        users.append({'email': email, 'password_hash': None, 'provider': 'google', 'verified': True})
        save_users(users)

    request.session['user_email'] = email
    return RedirectResponse(url="/menu", status_code=303)


@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/", status_code=303)


@app.get("/signup", response_class=HTMLResponse)
def signup_page():
    return Template(SIGNUP_PAGE).render(error=None)


@app.post("/signup", response_class=HTMLResponse)
def signup(email: str = Form(...), password: str = Form(...)):
    if not is_valid_email(email):
        return Template(SIGNUP_PAGE).render(error="Please enter a valid email address")

    if find_user(email):
        return Template(SIGNUP_PAGE).render(error="Email already registered")
        
    # التحقق من قوة الباسوورد بدل شرط الـ 4 حروف القديم
    if not is_strong_password(password):
        return Template(SIGNUP_PAGE).render(error="Password must be at least 8 characters, include an uppercase letter, a lowercase letter, and a number")

    code = str(random.randint(100000, 999999))
    users = load_users()
    users.append({
        'email': email,
        'password_hash': pwd_context.hash(password),
        'verified': False,
        'verification_code': code
    })
    save_users(users)

    try:
        send_verification_email(email, code)
    except Exception as e:
        print(f"Failed to send verification email: {e}")

    return RedirectResponse(url=f"/verify?email={email}", status_code=303)


@app.get("/verify", response_class=HTMLResponse)
def verify_page(email: str):
    return Template(VERIFY_PAGE).render(email=email, error=None)


@app.post("/verify", response_class=HTMLResponse)
def verify(request: Request, email: str = Form(...), code: str = Form(...)):
    users = load_users()
    for u in users:
        if u['email'] == email:
            if u.get('verification_code') == code:
                u['verified'] = True
                u.pop('verification_code', None)
                save_users(users)
                request.session['user_email'] = email
                return RedirectResponse(url="/menu", status_code=303)
            else:
                return Template(VERIFY_PAGE).render(email=email, error="Incorrect code, please try again")
    return Template(VERIFY_PAGE).render(email=email, error="Account not found")


@app.get("/resend")
def resend(email: str):
    users = load_users()
    for u in users:
        if u['email'] == email:
            code = str(random.randint(100000, 999999))
            u['verification_code'] = code
            save_users(users)
            try:
                send_verification_email(email, code)
            except Exception as e:
                print(f"Failed to resend verification email: {e}")
            break
    return RedirectResponse(url=f"/verify?email={email}", status_code=303)

# ---- دوال استعادة كلمة المرور ----

@app.get("/forgot", response_class=HTMLResponse)
def forgot_page():
    return Template(FORGOT_PAGE).render(message=None)

@app.post("/forgot", response_class=HTMLResponse)
def forgot_password_request(request: Request, email: str = Form(...)):
    user = find_user(email)
    
    # رسالة أمنية عامة
    success_msg = "If your email is registered, you will receive a reset link."
    
    if user:
        token = str(uuid.uuid4())
        tokens = load_tokens()
        tokens[token] = {
            'email': email,
            'expires': time.time() + 3600 # الرمز صالح لمدة ساعة
        }
        save_tokens(tokens)
        
        # إنشاء الرابط باستخدام رابط موقعكم
        reset_link = f"https://sudan-culture.duckdns.org/reset?token={token}"
        
        try:
            send_reset_email(email, reset_link)
        except Exception as e:
            print(f"Failed to send reset email: {e}")
        
    return Template(FORGOT_PAGE).render(message=success_msg)

@app.get("/reset", response_class=HTMLResponse)
def reset_page(token: str):
    tokens = load_tokens()
    if token not in tokens or time.time() > tokens[token]['expires']:
        return Template(RESET_PAGE).render(error="Invalid or expired reset link.", token=None)
    
    return Template(RESET_PAGE).render(error=None, token=token)

@app.post("/reset", response_class=HTMLResponse)
def reset_password_action(token: str = Form(...), new_password: str = Form(...)):
    tokens = load_tokens()
    
    if token not in tokens or time.time() > tokens[token]['expires']:
        return Template(RESET_PAGE).render(error="Invalid or expired reset link.", token=None)
    
    if not is_strong_password(new_password):
        return Template(RESET_PAGE).render(
            error="Password must be at least 8 characters, with 1 uppercase, 1 lowercase, and 1 number.", 
            token=token
        )
    
    email = tokens[token]['email']
    users = load_users()
    
    for u in users:
        if u['email'] == email:
            u['password_hash'] = pwd_context.hash(new_password)
            break
            
    save_users(users)
    
    del tokens[token]
    save_tokens(tokens)
    
    return RedirectResponse(url="/", status_code=303)

# -----------------------------------

@app.get("/menu", response_class=HTMLResponse)
def menu_page():
    return MENU_PAGE


@app.get("/food", response_class=HTMLResponse)
def food_page():
    return Template(DETECTOR_PAGE).render(
        title="Food detector", subtitle="Upload a photo to detect zalabia, cay, or mol5iya",
        result_image=None, detections=None
    )


@app.post("/food", response_class=HTMLResponse)
async def food_detect(image: UploadFile = File(...)):
    contents = await image.read()
    file_bytes = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    image_name = f'{timestamp}_{image.filename}'
    cv2.imwrite(os.path.join(UPLOAD_FOLDER, image_name), img)

    annotated_img, detections = run_detection(get_food_model(), img)
    cv2.imwrite(os.path.join(UPLOAD_FOLDER, f'{timestamp}_detected_{image.filename}'), annotated_img)
    save_result_to_json(image_name, 'food', detections)

    _, buffer = cv2.imencode('.jpg', annotated_img)
    result_image = base64.b64encode(buffer).decode('utf-8')

    return Template(DETECTOR_PAGE).render(
        title="Food detector", subtitle="Upload a photo to detect zalabia, cay, or mol5iya",
        result_image=result_image, detections=detections
    )


@app.get("/cloth", response_class=HTMLResponse)
def cloth_page():
    return Template(DETECTOR_PAGE).render(
        title="Cloth detector", subtitle="Upload a photo to detect clothing items",
        result_image=None, detections=None
    )


@app.post("/cloth", response_class=HTMLResponse)
async def cloth_detect(image: UploadFile = File(...)):
    contents = await image.read()
    file_bytes = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    image_name = f'{timestamp}_{image.filename}'
    cv2.imwrite(os.path.join(UPLOAD_FOLDER, image_name), img)

    annotated_img, detections = run_detection(get_cloth_model(), img)
    cv2.imwrite(os.path.join(UPLOAD_FOLDER, f'{timestamp}_detected_{image.filename}'), annotated_img)
    save_result_to_json(image_name, 'cloth', detections)

    _, buffer = cv2.imencode('.jpg', annotated_img)
    result_image = base64.b64encode(buffer).decode('utf-8')

    return Template(DETECTOR_PAGE).render(
        title="Cloth detector", subtitle="Upload a photo to detect clothing items",
        result_image=result_image, detections=detections
    )