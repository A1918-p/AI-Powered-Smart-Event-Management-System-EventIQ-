import streamlit as st
import pandas as pd
import numpy as np
import os
import qrcode
from io import BytesIO
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from textblob import TextBlob  # <-- NEW IMPORT FOR AI SENTIMENT ANALYSIS

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="EventIQ", layout="wide")

# --- INJECT FONTAWESOME CSS ---
st.markdown(
    '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">',
    unsafe_allow_html=True
)

# --- CONSTANTS & SETUP ---
DATA_FILE = "data/registrations.csv"
FEEDBACK_FILE = "data/feedback.csv" # <-- NEW FILE FOR FEEDBACK
EVENT_TYPES = ["Tech Event", "Cultural", "Workshop", "Seminar", "Hackathon"]

if not os.path.exists("data"):
    os.makedirs("data")

if not os.path.exists(DATA_FILE):
    pd.DataFrame(columns=["Name", "Email", "Event_Type"]).to_csv(DATA_FILE, index=False)
    
if not os.path.exists(FEEDBACK_FILE):
    pd.DataFrame(columns=["Event_Type", "Feedback", "Sentiment", "Polarity"]).to_csv(FEEDBACK_FILE, index=False)

def generate_qr_code(data_string):
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(data_string)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf

# --- MACHINE LEARNING MODEL SETUP ---
@st.cache_resource
def train_attendance_model():
    np.random.seed(42)
    hist_events = np.random.choice(EVENT_TYPES, 200)
    hist_past_attendance = np.random.randint(50, 500, 200)
    growth_factors = {"Tech Event": 1.2, "Cultural": 1.5, "Workshop": 1.05, "Seminar": 1.0, "Hackathon": 1.3}
    
    target_attendance = []
    for i in range(200):
        base = hist_past_attendance[i] * growth_factors[hist_events[i]]
        noise = np.random.randint(-20, 30)
        target_attendance.append(int(base + noise))
        
    df_ml = pd.DataFrame({"Event_Type": hist_events, "Past_Attendance": hist_past_attendance, "Actual_Attendance": target_attendance})
    
    le = LabelEncoder()
    df_ml['Event_Encoded'] = le.fit_transform(df_ml['Event_Type'])
    
    X = df_ml[['Event_Encoded', 'Past_Attendance']]
    y = df_ml['Actual_Attendance']
    
    model = LinearRegression()
    model.fit(X, y)
    
    return model, le, df_ml

ml_model, label_encoder, sample_data = train_attendance_model()

# --- SIDEBAR NAVIGATION ---
st.sidebar.markdown('<h2><i class="fa-solid fa-calendar-check"></i> EventIQ Navigation</h2>', unsafe_allow_html=True)
st.sidebar.markdown("### DEXTERITY 2K26 Management")

menu = ["1. Registration", "2. Dashboard", "3. Attendance Predictor", "4. Feedback Analysis"]
choice = st.sidebar.radio("Go to:", menu)

# --- MODULE 1: REGISTRATION SYSTEM ---
if choice == "1. Registration":
    st.markdown('<h1><i class="fa-solid fa-address-card"></i> Attendee Registration</h1>', unsafe_allow_html=True)
    st.write("Register participants and generate their entry QR codes.")
    
    with st.form(key="registration_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Full Name")
        with col2:
            email = st.text_input("Email Address")
        event_choice = st.selectbox("Select Event Type", EVENT_TYPES)
        submit_button = st.form_submit_button(label="Register")
        
    if submit_button:
        if name and email:
            new_data = pd.DataFrame({"Name": [name], "Email": [email], "Event_Type": [event_choice]})
            new_data.to_csv(DATA_FILE, mode='a', header=False, index=False)
            st.success(f"Successfully registered {name} for {event_choice}!")
            
            qr_data = f"Name:{name}|Event:{event_choice}"
            qr_image_buffer = generate_qr_code(qr_data)
            
            st.markdown('<h3><i class="fa-solid fa-qrcode"></i> Your Entry Ticket</h3>', unsafe_allow_html=True)
            st.image(qr_image_buffer, width=200)
            st.download_button("Download QR Ticket", qr_image_buffer, f"{name.replace(' ', '_')}_ticket.png", "image/png")
        else:
            st.error("Please fill in both Name and Email.")
            
    st.divider()
    st.markdown('<h3><i class="fa-solid fa-users"></i> Current Registrations</h3>', unsafe_allow_html=True)
    try:
        df = pd.read_csv(DATA_FILE)
        st.dataframe(df, use_container_width=True)
    except:
        st.write("No registrations yet.")

# --- MODULE 2: DASHBOARD ---
elif choice == "2. Dashboard":
    st.markdown('<h1><i class="fa-solid fa-chart-pie"></i> Live Dashboard</h1>', unsafe_allow_html=True)
    try:
        df = pd.read_csv(DATA_FILE)
        if df.empty:
            st.info("No registrations found.")
        else:
            st.metric(label="Total Registrations", value=len(df))
            st.divider()
            
            event_counts = df['Event_Type'].value_counts().reset_index()
            event_counts.columns = ['Event_Type', 'Count']
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('### Registrations per Event')
                fig_bar = px.bar(event_counts, x='Event_Type', y='Count', color='Event_Type', text='Count', template='plotly_white')
                st.plotly_chart(fig_bar, use_container_width=True)
            with col2:
                st.markdown('### Event Distribution')
                fig_pie = px.pie(event_counts, names='Event_Type', values='Count', hole=0.4, template='plotly_white')
                st.plotly_chart(fig_pie, use_container_width=True)
    except:
        st.warning("Data file not found.")

# --- MODULE 3: ATTENDANCE PREDICTOR (ML) ---
elif choice == "3. Attendance Predictor":
    st.markdown('<h1><i class="fa-solid fa-brain"></i> AI Attendance Predictor</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        with st.form(key="prediction_form"):
            pred_event_type = st.selectbox("Select Event Type", EVENT_TYPES)
            pred_past_attendance = st.number_input("Last Year's Attendance / Expected Base", min_value=10, max_value=5000, value=100)
            predict_button = st.form_submit_button("Predict Turnout")
            
        if predict_button:
            encoded_event = label_encoder.transform([pred_event_type])[0]
            input_features = pd.DataFrame([[encoded_event, pred_past_attendance]], columns=['Event_Encoded', 'Past_Attendance'])
            prediction = int(ml_model.predict(input_features)[0])
            
            st.success("Prediction Complete!")
            st.metric(label=f"Expected Turnout for {pred_event_type}", value=f"{prediction} Attendees", delta=f"{prediction - pred_past_attendance} from last year")

    with col2:
        st.info("**Algorithm:** Linear Regression")
        st.write("Trained on a synthetic dataset of 200 past events.")

# --- MODULE 4: FEEDBACK ANALYSIS (AI) ---
elif choice == "4. Feedback Analysis":
    st.markdown('<h1><i class="fa-solid fa-comments"></i> AI Feedback Sentiment Analysis</h1>', unsafe_allow_html=True)
    st.write("Submit attendee feedback and let NLP automatically determine the sentiment.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Submit Feedback")
        with st.form(key="feedback_form"):
            fb_event = st.selectbox("Which event are you reviewing?", EVENT_TYPES)
            fb_text = st.text_area("Write your feedback here:")
            fb_submit = st.form_submit_button("Analyze & Submit")
            
        if fb_submit:
            if fb_text:
                # --- NLP SENTIMENT ANALYSIS CORE ---
                analysis = TextBlob(fb_text)
                polarity = analysis.sentiment.polarity
                
                # Determine category based on polarity score
                if polarity > 0.1:
                    sentiment_label = "Positive 😃"
                elif polarity < -0.1:
                    sentiment_label = "Negative 😞"
                else:
                    sentiment_label = "Neutral 😐"
                    
                # Save to CSV
                new_fb = pd.DataFrame({"Event_Type": [fb_event], "Feedback": [fb_text], "Sentiment": [sentiment_label], "Polarity": [round(polarity, 2)]})
                new_fb.to_csv(FEEDBACK_FILE, mode='a', header=False, index=False)
                
                st.success(f"Feedback recorded as **{sentiment_label}** (Score: {round(polarity, 2)})")
            else:
                st.error("Please write some feedback before submitting.")

    with col2:
        st.markdown("### Recent Feedback Logs")
        try:
            df_fb = pd.read_csv(FEEDBACK_FILE)
            if df_fb.empty:
                st.info("No feedback recorded yet.")
            else:
                # Display the data visually
                st.dataframe(df_fb.tail(10).iloc[::-1], use_container_width=True)
                
                # Show a quick pie chart of overall sentiment
                sentiment_counts = df_fb['Sentiment'].value_counts().reset_index()
                sentiment_counts.columns = ['Sentiment', 'Count']
                fig_sentiment = px.pie(sentiment_counts, names='Sentiment', values='Count', title="Overall Event Sentiment", hole=0.5)
                st.plotly_chart(fig_sentiment, use_container_width=True)
                
        except FileNotFoundError:
            st.warning("Feedback database not initialized yet.")