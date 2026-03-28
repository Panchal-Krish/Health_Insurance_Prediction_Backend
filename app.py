from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta, timezone
from bson import ObjectId
import uuid
import jwt
import os
import joblib
import numpy as np
import pandas as pd
from functools import wraps
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# ==============================
# Configuration
# ==============================
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config['JWT_EXPIRATION_HOURS'] = 24

if not app.config['SECRET_KEY']:
    raise RuntimeError("SECRET_KEY is not set in .env")

CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# ==============================
# MongoDB
# ==============================
MONGO_URI = os.getenv('MONGO_URI')
if not MONGO_URI:
    raise RuntimeError("MONGO_URI is not set in .env")

try:
    client              = MongoClient(MONGO_URI)
    db                  = client["insurance_data"]
    users_collection    = db["customers"]
    premium_collection  = db["premium_history"]
    prediction_logs     = db["prediction_logs"]
    tickets_collection  = db["support_tickets"]
    contacts_collection = db["contacts"]
    qa_collection       = db["Chatbot"]           # ← Chatbot Q&A

    users_collection.create_index("email", unique=True)
    premium_collection.create_index("email", unique=True)
    prediction_logs.create_index("email")
    contacts_collection.create_index("created_at")

    print("✅ Database connected successfully")
except Exception as e:
    print(f"❌ Database connection failed: {e}")
    exit(1)

# ==============================
# ML Model — load once at startup
# ==============================
MODEL_PATH = os.getenv('MODEL_PATH', 'insurance_extra_trees_model.pkl')

try:
    ml_model = joblib.load(MODEL_PATH)
    print(f"✅ ML model loaded from {MODEL_PATH}")
except Exception as e:
    ml_model = None
    print(f"⚠️  ML model not loaded ({e}) — fallback formula will be used")

# ==============================
# ML Prediction
# ==============================
def predict_premium_ml(age, sex, bmi, children, smoker, region):
    if ml_model is not None:
        region_northwest = (region == 'northwest')
        region_southeast = (region == 'southeast')
        region_southwest = (region == 'southwest')

        features = pd.DataFrame([{
            'age':              age,
            'sex':              sex,
            'bmi':              bmi,
            'children':         children,
            'smoker':           smoker,
            'region_northwest': region_northwest,
            'region_southeast': region_southeast,
            'region_southwest': region_southwest,
            'age_bmi':          float(age * bmi),
            'age_smoker':       int(age * smoker),
            'bmi_smoker':       float(bmi * smoker),
            'children_smoker':  int(children * smoker),
            'bmi_sq':           float(bmi ** 2),
            'age_sq':           int(age ** 2)
        }])

        if hasattr(ml_model, "feature_names_in_"):
            features = features[ml_model.feature_names_in_]

        log_pred = ml_model.predict(features)[0]
        return round(float(np.expm1(log_pred)), 2)

    print("⚠️  Using fallback formula — model not loaded")
    if bmi < 18.5:    bmi_p = 1000
    elif bmi < 25:    bmi_p = 0
    elif bmi < 30:    bmi_p = 1500
    else:             bmi_p = 3000

    region_p = {'northeast': 0, 'northwest': 500, 'southeast': 1000, 'southwest': 1500}.get(region, 0)
    return round(3000 + age*250 + bmi_p + children*500 + (20000 if smoker else 0) + (500 if sex else 0) + region_p, 2)


# ==============================
# JWT Helpers
# ==============================
def generate_token(email, role):
    try:
        payload = {
            'email': email,
            'role':  role,
            'exp':   datetime.now(timezone.utc) + timedelta(hours=app.config['JWT_EXPIRATION_HOURS']),
            'iat':   datetime.now(timezone.utc)
        }
        return jwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')
    except Exception:
        return None

def decode_token(token):
    try:
        return jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None

# ==============================
# Decorators
# ==============================
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            try:
                token = request.headers['Authorization'].split(" ")[1]
            except IndexError:
                return jsonify({'message': 'Invalid token format'}), 401

        if not token:
            return jsonify({'message': 'Token is missing'}), 401

        payload = decode_token(token)
        if not payload:
            return jsonify({'message': 'Token is invalid or expired'}), 401

        request.current_user = payload
        return f(*args, **kwargs)
    return decorated

def role_required(allowed_roles):
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if not hasattr(request, 'current_user'):
                return jsonify({'message': 'Unauthorized'}), 401
            if request.current_user.get('role') not in allowed_roles:
                return jsonify({'message': f'Access denied. Required role: {", ".join(allowed_roles)}'}), 403
            return f(*args, **kwargs)
        return decorated
    return decorator


# ======================================================
#                   CHATBOT — Q&A PRELOAD
# ======================================================

DEFAULT_QA = [
    {
        "keywords": ["premium", "cost", "price", "how much", "amount"],
        "answer": "Your **insurance premium** is the amount you pay regularly for your health insurance plan.\n\nFactors that affect your premium:\n• Age — older age = higher premium\n• BMI — higher BMI = higher premium\n• Smoking — smokers pay significantly more\n• Pre-existing diseases — increases premium\n• Number of children/dependents\n• Annual income\n• Region"
    },
    {
        "keywords": ["bmi", "body mass index", "weight", "obesity"],
        "answer": "**BMI (Body Mass Index)** is a measure of body fat based on height and weight.\n\nBMI Categories:\n• Below 18.5 — Underweight\n• 18.5 to 24.9 — Normal weight\n• 25 to 29.9 — Overweight\n• 30 and above — Obese\n\nHigher BMI means higher health risks, so insurers charge a higher premium. Maintaining a healthy BMI can reduce your insurance costs."
    },
    {
        "keywords": ["smoke", "smoking", "smoker", "cigarette", "tobacco"],
        "answer": "**Smoking** is one of the biggest factors that increases your insurance premium.\n\n• Smokers can pay up to **₹10,000 more** in premium\n• Smoking increases risk of heart disease, cancer, and lung disease\n• Quitting smoking can significantly reduce your premium over time\n• Always be honest about smoking status — false information can void your policy"
    },
    {
        "keywords": ["age", "old", "young", "older", "younger"],
        "answer": "**Age** directly affects your health insurance premium.\n\n• The older you are, the higher your premium\n• Each year of age adds approximately ₹120 to your annual premium\n• Younger people are considered lower risk\n• It's advisable to get health insurance early when premiums are lower"
    },
    {
        "keywords": ["children", "dependents", "kids", "family", "child"],
        "answer": "**Dependents/Children** affect your premium as follows:\n\n• Each child/dependent adds approximately ₹800 to your annual premium\n• Family floater plans cover all family members under one policy\n• Adding more dependents increases the overall premium\n• Children are generally low-risk but still add to the cost"
    },
    {
        "keywords": ["pre-existing", "disease", "condition", "diabetes", "heart", "illness", "medical history"],
        "answer": "**Pre-existing diseases** are medical conditions you had before getting insurance.\n\n• Having pre-existing conditions adds approximately ₹7,000 to your premium\n• Common ones include diabetes, hypertension, heart disease\n• Some insurers have a waiting period before covering pre-existing conditions\n• Always disclose pre-existing conditions honestly to avoid claim rejection"
    },
    {
        "keywords": ["deductible", "deduct", "out of pocket", "copay", "co-pay"],
        "answer": "**Deductible** is the amount you pay before your insurance starts covering costs.\n\n• Higher deductible = Lower premium\n• Lower deductible = Higher premium\n• Example: If deductible is ₹5,000 and your bill is ₹20,000, you pay ₹5,000 and insurance covers ₹15,000\n• Choose a deductible you can comfortably afford"
    },
    {
        "keywords": ["reduce", "lower", "decrease", "save", "cheaper", "less"],
        "answer": "**Ways to reduce your insurance premium:**\n\n• Quit smoking — saves up to ₹10,000/year\n• Maintain a healthy BMI through diet and exercise\n• Buy insurance at a younger age\n• Choose a higher deductible\n• Opt for a no-claim bonus by staying healthy\n• Compare multiple plans before choosing\n• Avoid unnecessary claims for minor expenses"
    },
    {
        "keywords": ["predict", "calculate", "estimate", "how to use", "platform", "website"],
        "answer": "**How to use this platform:**\n\n• Go to the **Predict** page from the navigation menu\n• Fill in your details — age, gender, BMI, smoking status, etc.\n• Click **Predict Premium** to get your estimated premium\n• View your history on the **Dashboard** page\n• Use this chatbot anytime for help and queries\n\nYour data is securely stored and you can check your premium anytime!"
    },
    {
        "keywords": ["region", "location", "area", "city", "state"],
        "answer": "**Region** can affect your insurance premium based on:\n\n• Cost of healthcare in your area\n• Risk factors associated with your region\n• Urban vs rural healthcare availability\n• Different regions may have different base rates\n\nOur platform currently considers your region as one of the factors in premium calculation."
    },
    {
        "keywords": ["income", "salary", "earning", "annual"],
        "answer": "**Annual Income** affects your premium in the following way:\n\n• Higher income (above ₹10,00,000/year) may result in a higher premium\n• This reflects ability to pay and sometimes lifestyle factors\n• Income helps insurers assess the appropriate coverage level\n• Higher income earners often opt for more comprehensive plans"
    },
    {
        "keywords": ["coverage", "cover", "what is covered", "benefits", "policy"],
        "answer": "**Health Insurance Coverage** typically includes:\n\n• Hospitalization expenses\n• Pre and post hospitalization costs\n• Day care procedures\n• Ambulance charges\n• Sometimes dental and vision care\n\nAlways read your policy document carefully to understand what is and isn't covered. Coverage varies by plan and insurer."
    },
    {
        "keywords": ["claim", "claims", "file claim", "how to claim"],
        "answer": "**How to file an insurance claim:**\n\n• **Cashless claim** — Get treated at a network hospital, insurer pays directly\n• **Reimbursement claim** — Pay yourself, then submit bills to insurer\n• Keep all medical bills and documents\n• Inform your insurer within the specified time period\n• Fill the claim form accurately and honestly\n\nAlways keep copies of all submitted documents!"
    },
    {
        "keywords": ["hello", "hi", "hey", "good morning", "good evening", "greet"],
        "answer": "👋 **Hello! Welcome to HealthBot!**\n\nI'm here to help you with all your health insurance queries.\n\nYou can ask me about:\n• Insurance premiums and costs\n• BMI and health factors\n• How to use this platform\n• Insurance terms and concepts\n\nWhat would you like to know?"
    },
    {
        "keywords": ["thank", "thanks", "thank you", "helpful", "great"],
        "answer": "You're welcome! 😊\n\nI'm always here to help with your health insurance queries. Feel free to ask anything else!\n\n• Use the **Predict** page to calculate your premium\n• Check **Dashboard** for your history\n• Come back anytime you have questions!"
    },
    {
        "keywords": ["gender", "male", "female", "sex"],
        "answer": "**Gender** is one of the factors considered in health insurance.\n\n• Some insurers may have slight differences in premiums based on gender\n• Women may have specific coverage for maternity benefits\n• Our platform uses gender as one of many factors in premium calculation\n• Overall, other factors like smoking and BMI have a much larger impact on your premium"
    }
]

def preload_qa():
    if qa_collection.count_documents({}) == 0:
        qa_collection.insert_many(DEFAULT_QA)
        print("✅ Default Q&As loaded into MongoDB!")
    else:
        print("✅ Q&As already exist in MongoDB.")

preload_qa()


# ======================================================
#                   AUTH ROUTES
# ======================================================

@app.route("/signup", methods=["POST"])
def signup():
    try:
        data = request.json
        if not all(k in data for k in ["fullName", "email", "password"]):
            return jsonify({"message": "Missing required fields"}), 400

        full_name = data.get("fullName", "").strip()
        email     = data.get("email", "").lower().strip()
        password  = data.get("password", "")

        if len(full_name) < 2:
            return jsonify({"message": "Full name must be at least 2 characters"}), 400
        if len(password) < 6:
            return jsonify({"message": "Password must be at least 6 characters"}), 400
        if users_collection.find_one({"email": email}):
            return jsonify({"message": "User already exists"}), 400

        users_collection.insert_one({
            "fullName":   full_name,
            "email":      email,
            "password":   generate_password_hash(password),
            "role":       "user",
            "created_at": datetime.now(timezone.utc)
        })
        return jsonify({"message": "Signup successful"}), 201

    except Exception as e:
        print(f"Signup error: {e}")
        return jsonify({"message": "Server error during signup"}), 500


@app.route("/login", methods=["POST"])
def login():
    try:
        data = request.json
        if not data or not data.get("email") or not data.get("password"):
            return jsonify({"message": "Email and password required"}), 400

        email    = data["email"].lower().strip()
        password = data["password"]

        user = users_collection.find_one({"email": email})
        if not user or not check_password_hash(user["password"], password):
            return jsonify({"message": "Invalid email or password"}), 401

        role  = user.get("role", "user")
        token = generate_token(email, role)
        if not token:
            return jsonify({"message": "Failed to generate token"}), 500

        return jsonify({
            "message":  "Login successful",
            "token":    token,
            "email":    email,
            "role":     role,
            "fullName": user.get("fullName")
        }), 200

    except Exception as e:
        print(f"Login error: {e}")
        return jsonify({"message": "Server error during login"}), 500


# ======================================================
#                   PREDICTION ROUTES
# ======================================================

@app.route("/predict-premium", methods=["POST"])
@token_required
def predict_premium():
    try:
        data = request.json
        if not all(k in data for k in ["age", "sex", "bmi", "children", "smoker", "region"]):
            return jsonify({"message": "Missing required fields"}), 400

        try:
            age      = int(data["age"])
            bmi      = float(data["bmi"])
            children = int(data["children"])
            smoker   = int(data["smoker"])
            sex      = int(data["sex"])
            region   = str(data["region"])
        except (ValueError, TypeError):
            return jsonify({"message": "Invalid input format"}), 400

        if not (18 <= age <= 100):
            return jsonify({"message": "Age must be an integer between 18 and 100"}), 400
        if not (10 <= bmi <= 60):
            return jsonify({"message": "BMI must be a number between 10 and 60"}), 400
        if not (0 <= children <= 10):
            return jsonify({"message": "Children must be an integer between 0 and 10"}), 400
        if smoker not in [0, 1]:
            return jsonify({"message": "Smoker must be 0 or 1"}), 400
        if sex not in [0, 1]:
            return jsonify({"message": "Sex must be 0 (female) or 1 (male)"}), 400
        if region not in ['northeast', 'northwest', 'southeast', 'southwest']:
            return jsonify({"message": "Invalid region"}), 400

        premium    = predict_premium_ml(age=age, sex=sex, bmi=bmi, children=children, smoker=smoker, region=region)
        user_email = request.current_user.get('email')
        now        = datetime.now(timezone.utc)

        record = {
            "age":               age,
            "gender":            "male" if sex == 1 else "female",
            "bmi":               bmi,
            "children":          children,
            "smoker":            smoker == 1,
            "region":            region,
            "predicted_premium": premium,
            "last_checked_at":   now
        }

        premium_collection.update_one(
            {"email": user_email},
            {"$set": record, "$setOnInsert": {"email": user_email}},
            upsert=True
        )
        prediction_logs.insert_one({"email": user_email, **record})

        return jsonify({"predicted_premium": premium}), 200

    except Exception as e:
        print(f"Premium prediction error: {e}")
        return jsonify({"message": "Server error during prediction"}), 500


@app.route("/premium-history", methods=["GET"])
@token_required
def premium_history():
    try:
        user_email = request.current_user.get('email')
        record     = premium_collection.find_one({"email": user_email}, {"_id": 0})
        if not record:
            return jsonify(None), 200
        if "last_checked_at" in record:
            record["last_checked_at"] = record["last_checked_at"].isoformat()
        return jsonify(record), 200
    except Exception as e:
        print(f"Premium history error: {e}")
        return jsonify({"message": "Server error fetching history"}), 500


@app.route("/my-predictions", methods=["GET"])
@token_required
def my_predictions():
    try:
        user_email = request.current_user.get('email')
        logs = list(
            prediction_logs.find({"email": user_email}, {"_id": 0})
            .sort("last_checked_at", -1).limit(20)
        )
        for log in logs:
            if "last_checked_at" in log:
                log["last_checked_at"] = log["last_checked_at"].isoformat()
        return jsonify(logs), 200
    except Exception as e:
        print(f"Prediction logs error: {e}")
        return jsonify({"message": "Server error fetching prediction history"}), 500


# ======================================================
#                   CONTACT ROUTE
# ======================================================

@app.route("/contact", methods=["POST"])
def contact():
    try:
        data = request.json
        if not all(k in data for k in ["name", "email", "subject", "message"]):
            return jsonify({"message": "Missing required fields"}), 400

        name    = data["name"].strip()
        email   = data["email"].strip().lower()
        subject = data["subject"].strip()
        message = data["message"].strip()

        if len(name) < 3 or len(name) > 100:
            return jsonify({"message": "Name must be between 3 and 100 characters"}), 400
        if len(subject) < 5 or len(subject) > 100:
            return jsonify({"message": "Subject must be between 5 and 100 characters"}), 400
        if len(message) < 20 or len(message) > 1000:
            return jsonify({"message": "Message must be between 20 and 1000 characters"}), 400

        contacts_collection.insert_one({
            "name":       name,
            "email":      email,
            "subject":    subject,
            "message":    message,
            "status":     "unread",
            "created_at": datetime.now(timezone.utc)
        })
        return jsonify({"message": "Message received. We'll get back to you within 24-48 hours."}), 201

    except Exception as e:
        print(f"Contact error: {e}")
        return jsonify({"message": "Server error. Please try again."}), 500


# ======================================================
#                   HELPDESK ROUTES
# ======================================================

@app.route("/create-ticket", methods=["POST"])
@token_required
def create_ticket():
    try:
        data       = request.json
        user_email = request.current_user.get('email')

        if not all(k in data for k in ["subject", "description", "category", "priority"]):
            return jsonify({"message": "Missing required fields"}), 400

        subject     = data["subject"].strip()
        description = data["description"].strip()
        category    = data["category"].strip()
        priority    = data["priority"].strip()

        if len(subject) < 5 or len(subject) > 100:
            return jsonify({"message": "Subject must be between 5 and 100 characters"}), 400
        if len(description) < 10 or len(description) > 1000:
            return jsonify({"message": "Description must be between 10 and 1000 characters"}), 400
        if category not in ["Model Issue", "Prediction Error", "Technical Problem", "Account Problem", "Other"]:
            return jsonify({"message": "Invalid category"}), 400
        if priority not in ["Low", "Medium", "High"]:
            return jsonify({"message": "Priority must be Low, Medium, or High"}), 400

        ticket = {
            "ticket_id":        "TICK-" + str(uuid.uuid4())[:8].upper(),
            "email":            user_email,
            "subject":          subject,
            "description":      description,
            "category":         category,
            "priority":         priority,
            "status":           "Open",
            "assigned_to":      None,
            "assigned_role":    None,
            "admin_response":   "",
            "manager_response": "",
            "created_at":       datetime.now(timezone.utc)
        }

        tickets_collection.insert_one(ticket)
        return jsonify({"message": "Ticket created successfully", "ticket_id": ticket["ticket_id"]}), 201

    except Exception as e:
        print(f"Create ticket error: {e}")
        return jsonify({"message": "Server error creating ticket"}), 500


@app.route("/my-tickets", methods=["GET"])
@token_required
def get_user_tickets():
    try:
        user_email = request.current_user.get('email')
        tickets    = list(
            tickets_collection.find({"email": user_email}, {"_id": 0})
            .sort("created_at", -1)
        )
        for ticket in tickets:
            ticket["created_at"] = ticket["created_at"].isoformat()
        return jsonify(tickets), 200
    except Exception as e:
        print(f"Get tickets error: {e}")
        return jsonify({"message": "Server error fetching tickets"}), 500


# ======================================================
#                   ADMIN ROUTES
# ======================================================

@app.route("/admin/all-tickets", methods=["GET"])
@token_required
@role_required(['admin'])
def admin_all_tickets():
    try:
        tickets = list(tickets_collection.find({}, {"_id": 0}).sort("created_at", -1))
        for ticket in tickets:
            ticket["created_at"] = ticket["created_at"].isoformat()
        return jsonify(tickets), 200
    except Exception as e:
        print(f"Admin get tickets error: {e}")
        return jsonify({"message": "Server error fetching tickets"}), 500


@app.route("/admin/assign-ticket", methods=["POST"])
@token_required
@role_required(['admin'])
def assign_ticket():
    try:
        data = request.json
        if not all(k in data for k in ["ticket_id", "assigned_to", "assigned_role"]):
            return jsonify({"message": "Missing required fields"}), 400

        result = tickets_collection.update_one(
            {"ticket_id": data["ticket_id"]},
            {"$set": {"assigned_to": data["assigned_to"], "assigned_role": data["assigned_role"]}}
        )
        if result.matched_count == 0:
            return jsonify({"message": "Ticket not found"}), 404
        return jsonify({"message": "Ticket assigned successfully"}), 200
    except Exception as e:
        print(f"Assign ticket error: {e}")
        return jsonify({"message": "Server error assigning ticket"}), 500


@app.route("/admin/update-ticket/<ticket_id>", methods=["PUT"])
@token_required
@role_required(['admin'])
def admin_update_ticket(ticket_id):
    try:
        data          = request.json
        update_fields = {}

        if "status" in data:
            update_fields["status"] = data["status"]
        if "admin_response" in data:
            response_text = data["admin_response"].strip()
            if len(response_text) < 10:
                return jsonify({"message": "Response must be at least 10 characters"}), 400
            update_fields["admin_response"] = response_text

        if not update_fields:
            return jsonify({"message": "No fields to update"}), 400

        result = tickets_collection.update_one({"ticket_id": ticket_id}, {"$set": update_fields})
        if result.matched_count == 0:
            return jsonify({"message": "Ticket not found"}), 404
        return jsonify({"message": "Ticket updated successfully"}), 200
    except Exception as e:
        print(f"Update ticket error: {e}")
        return jsonify({"message": "Server error updating ticket"}), 500


@app.route("/admin/create-manager", methods=["POST"])
@token_required
@role_required(['admin'])
def create_manager():
    try:
        data = request.json
        if not all(k in data for k in ["fullName", "email", "password"]):
            return jsonify({"message": "Missing required fields"}), 400

        full_name = data["fullName"].strip()
        email     = data["email"].lower().strip()
        password  = data["password"]

        if len(full_name) < 3:
            return jsonify({"message": "Full name must be at least 3 characters"}), 400
        if len(password) < 6:
            return jsonify({"message": "Password must be at least 6 characters"}), 400
        if users_collection.find_one({"email": email}):
            return jsonify({"message": "User already exists"}), 400

        users_collection.insert_one({
            "fullName":   full_name,
            "email":      email,
            "password":   generate_password_hash(password),
            "role":       "manager",
            "created_at": datetime.now(timezone.utc)
        })
        return jsonify({"message": "Manager account created"}), 201
    except Exception as e:
        print(f"Create manager error: {e}")
        return jsonify({"message": "Server error creating manager"}), 500


@app.route("/admin/managers", methods=["GET"])
@token_required
@role_required(['admin'])
def get_managers():
    try:
        managers = list(users_collection.find({"role": "manager"}, {"_id": 0, "email": 1, "fullName": 1}))
        return jsonify(managers), 200
    except Exception as e:
        print(f"Get managers error: {e}")
        return jsonify({"message": "Server error fetching managers"}), 500


@app.route("/admin/contacts", methods=["GET"])
@token_required
@role_required(['admin'])
def admin_contacts():
    try:
        messages = list(
            contacts_collection.find(
                {},
                {"_id": 1, "name": 1, "email": 1, "subject": 1, "message": 1, "status": 1, "created_at": 1}
            ).sort("created_at", -1)
        )
        for msg in messages:
            msg["_id"]        = str(msg["_id"])
            msg["created_at"] = msg["created_at"].isoformat()
        return jsonify(messages), 200
    except Exception as e:
        print(f"Admin contacts error: {e}")
        return jsonify({"message": "Server error fetching contacts"}), 500


@app.route("/admin/contacts/<contact_id>/read", methods=["PUT"])
@token_required
@role_required(['admin'])
def mark_contact_read(contact_id):
    try:
        result = contacts_collection.update_one(
            {"_id": ObjectId(contact_id)},
            {"$set": {"status": "read"}}
        )
        if result.matched_count == 0:
            return jsonify({"message": "Message not found"}), 404
        return jsonify({"message": "Marked as read"}), 200
    except Exception as e:
        print(f"Mark contact read error: {e}")
        return jsonify({"message": "Server error"}), 500


# ======================================================
#                   MANAGER ROUTES
# ======================================================

@app.route("/manager/my-tickets", methods=["GET"])
@token_required
@role_required(['manager', 'admin'])
def manager_tickets():
    try:
        user_email   = request.current_user.get('email')
        user_role    = request.current_user.get('role')
        target_email = request.args.get('email', user_email)

        if user_role == 'manager' and target_email != user_email:
            return jsonify({"message": "Unauthorized access"}), 403

        tickets = list(
            tickets_collection.find({"assigned_to": target_email}, {"_id": 0})
            .sort("created_at", -1)
        )
        for ticket in tickets:
            ticket["created_at"] = ticket["created_at"].isoformat()
        return jsonify(tickets), 200
    except Exception as e:
        print(f"Manager tickets error: {e}")
        return jsonify({"message": "Server error fetching tickets"}), 500


@app.route("/manager/update-ticket/<ticket_id>", methods=["PUT"])
@token_required
@role_required(['manager', 'admin'])
def manager_update_ticket(ticket_id):
    try:
        data          = request.json
        update_fields = {}

        if "status" in data:
            update_fields["status"] = data["status"]
        if "manager_response" in data:
            response_text = data["manager_response"].strip()
            if len(response_text) < 10:
                return jsonify({"message": "Response must be at least 10 characters"}), 400
            update_fields["manager_response"] = response_text

        if not update_fields:
            return jsonify({"message": "No fields to update"}), 400

        result = tickets_collection.update_one({"ticket_id": ticket_id}, {"$set": update_fields})
        if result.matched_count == 0:
            return jsonify({"message": "Ticket not found"}), 404
        return jsonify({"message": "Ticket updated by manager"}), 200
    except Exception as e:
        print(f"Manager update error: {e}")
        return jsonify({"message": "Server error updating ticket"}), 500


# ======================================================
#                   CHATBOT ROUTES
# ======================================================

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data     = request.get_json()
        messages = data.get("messages", [])

        if not messages:
            return jsonify({"error": "No messages provided"}), 400

        user_message = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                user_message = msg["content"].lower()
                break

        all_qa     = list(qa_collection.find({}, {"_id": 0}))
        best_match = None
        best_score = 0

        for qa in all_qa:
            score = sum(1 for keyword in qa["keywords"] if keyword.lower() in user_message)
            if score > best_score:
                best_score = score
                best_match = qa

        if best_match and best_score > 0:
            return jsonify({"reply": best_match["answer"]}), 200

        default_reply = (
            "I'm not sure about that specific query. 🤔\n\n"
            "I can help you with:\n"
            "• **Insurance premiums** and what affects them\n"
            "• **BMI** and health factors\n"
            "• **Smoking** and its impact on costs\n"
            "• **How to use** this platform\n"
            "• Insurance terms like **deductibles** and **coverage**\n\n"
            "Please try rephrasing your question or ask about one of the topics above!"
        )
        return jsonify({"reply": default_reply}), 200

    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/add-qa", methods=["POST"])
@token_required
@role_required(['admin'])
def add_qa():
    try:
        data     = request.get_json()
        keywords = data.get("keywords", [])
        answer   = data.get("answer", "")

        if not keywords or not answer:
            return jsonify({"message": "keywords and answer are required"}), 400

        qa_collection.insert_one({"keywords": keywords, "answer": answer})
        return jsonify({"message": "Q&A added successfully!"}), 201
    except Exception as e:
        print(f"Add Q&A error: {e}")
        return jsonify({"message": "Server error adding Q&A"}), 500


# ==============================
#           HEALTH CHECK
# ==============================

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status":       "healthy",
        "message":      "API is running",
        "model_loaded": ml_model is not None
    }), 200


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)