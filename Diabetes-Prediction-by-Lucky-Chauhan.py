import streamlit as st
import base64
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO, StringIO
import datetime

st.set_page_config(page_title="Diabetes Prediction and Dashboard", page_icon="🩺", layout="wide")
#for backgrounf image
def set_background_image():
    with open("COLOURlovers.com-Grey_Background.jpg", "rb") as img_file:
        b64_string = base64.b64encode(img_file.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{b64_string}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: #FFFFFF
    }}
    .dataframe {{
        color: #00DDFF;
    }}
 
    .main .block-container {{
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
        padding-left: 0rem !important;
        padding-right: 0rem !important;
        max-width: 100% !important;
    }}
    header[data-testid="stHeader"] {{
        height: 0px !important;
        background: rgba(0,0,0,0) !important;
    }}
    .main {{
        padding-top: 0rem !important;
    }}
    .stApp > footer {{
        visibility: hidden;
    }}
    .css-1d391kg {{
        padding-top: 0rem !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
set_background_image()

@st.cache_data
def load_dataset():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    df = pd.read_csv(url, header=None, names=columns)
    df['Outcome'] = df['Outcome'].astype(int)
    return df

data = load_dataset()

model = joblib.load("diabetes_prediction_model.pkl")
scaler = joblib.load("scaler.pkl")

if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

#sidebar.
with st.sidebar:
    st.header("📊 Dataset Analysis")
    
    st.subheader("Dataset Overview")
    st.write(f"**Total Records:** {len(data)}")
    st.write(f"**Features:** {len(data.columns)-1}")
    st.write(f"**Diabetic Cases:** {len(data[data['Outcome']==1])}")
    st.write(f"**Non-Diabetic Cases:** {len(data[data['Outcome']==0])}")
    st.write(f"**Diabetes Rate:** {(len(data[data['Outcome']==1])/len(data)*100):.1f}%")
    
    st.subheader("Feature Statistics")
    if st.checkbox("Show Detailed Stats"):
        st.dataframe(data.describe())
    
    st.subheader("Data Distribution")
    if st.checkbox("Show Feature Distributions"):
        feature = st.selectbox("Select Feature", 
                              ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                               'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
        
        fig, ax = plt.subplots(figsize=(8, 4))
        data[data['Outcome']==0][feature].hist(alpha=0.7, label='Non-Diabetic', bins=20)
        data[data['Outcome']==1][feature].hist(alpha=0.7, label='Diabetic', bins=20)
        ax.legend()
        ax.set_title(f'{feature} Distribution')
        st.pyplot(fig)
    
    st.subheader("Correlation Matrix")
    if st.checkbox("Show Correlations"):
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm', center=0, ax=ax)
        st.pyplot(fig)

st.markdown("""
<div style="padding: 2rem 0; text-align: center; background: #000000 ; border-radius: 10px; margin-bottom: 2rem;">
    <h1 style="color: #EB4F4F; margin: 0; font-size: 3.5rem; font-weight: bold;">
          ᴅɪᴀʙᴇᴛᴇꜱ ᴘʀᴇᴅɪᴄᴛɪᴏɴ ᴅᴀꜱʜʙᴏᴀʀᴅ 🩺  
    </h1>
    <p style="color: #B18A72; font-size: 1.2rem; margin: 0.5rem 0 0 0;">
        𝐀𝐈 𝐩𝐨𝐰𝐞𝐫𝐞𝐝 𝐝𝐢𝐚𝐛𝐞𝐭𝐞𝐬 𝐫𝐢𝐬𝐤 𝐚𝐬𝐬𝐞𝐬𝐬𝐦𝐞𝐧𝐭 𝐚𝐧𝐝 𝐝𝐚𝐭𝐚 𝐯𝐢𝐬𝐮𝐚𝐥𝐢𝐳𝐚𝐭𝐢𝐨𝐧
    </p>
</div>
""", unsafe_allow_html=True)

FEATURES = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin',
            'BMI','DiabetesPedigreeFunction','Age']

HEALTHY_RANGES = {
    'Pregnancies': (0, 6),
    'Glucose': (70, 99),
    'BloodPressure': (60, 80),
    'SkinThickness': (10, 30),
    'Insulin': (15, 200),
    'BMI': (18.5, 24.9),
    'DiabetesPedigreeFunction': (0.0, 0.5),
    'Age': (18, 65)
}

@st.cache_data
def compute_average_stats(df: pd.DataFrame):
    """Compute average statistics for comparison."""
    healthy = df[df['Outcome'] == 0]
    diabetic = df[df['Outcome'] == 1]
    stats = {}
    for col in FEATURES:
        stats[col] = {
            'healthy_avg': float(healthy[col].mean()),
            'diabetic_avg': float(diabetic[col].mean()),
            'overall_avg': float(df[col].mean())
        }
    return stats

def risk_bucket(p):
    """Risk categorization based on probability."""
    if p >= 0.75:
        return "High Risk", "🔴"
    elif p >= 0.40:
        return "Medium Risk", "🟡"
    else:
        return "Low Risk", "🟢"

def predict_patient(df_row: pd.DataFrame):
    """Scale, predict class and probability for a single patient."""
    Xs = scaler.transform(df_row[FEATURES])
    pred = model.predict(Xs)[0]
    proba = float(model.predict_proba(Xs)[0,1])
    return int(pred), proba

def create_patient_comparison_chart(df, patient_df):
    """Create comparison chart showing patient vs average values."""
    stats = compute_average_stats(df)
    #pepare data for ploting
    features = []
    patient_values = []
    healthy_avgs = []
    diabetic_avgs = []
    healthy_min = []
    healthy_max = []
    
    for feature in FEATURES:
        features.append(feature)
        patient_values.append(float(patient_df[feature].iloc[0]))
        healthy_avgs.append(stats[feature]['healthy_avg'])
        diabetic_avgs.append(stats[feature]['diabetic_avg'])
        healthy_min.append(HEALTHY_RANGES[feature][0])
        healthy_max.append(HEALTHY_RANGES[feature][1])
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    axes = axes.flatten()
    
    for i, feature in enumerate(FEATURES):
        ax = axes[i]
        x_pos = [0, 1, 2, 3]
        values = [
            patient_values[i],
            healthy_avgs[i],
            diabetic_avgs[i],
            (healthy_min[i] + healthy_max[i]) / 2
        ]
        labels = ['Patient', 'Healthy\nAvg', 'Diabetic\nAvg', 'Healthy\nRange']
        colors = ['red', 'green', 'orange', 'lightblue']
        #bar creatin7
        bars = ax.bar(x_pos, values, color=colors, alpha=0.7, edgecolor='black')
        
        ax.errorbar(3, values[3], yerr=[[values[3] - healthy_min[i]], [healthy_max[i] - values[3]]], 
                   fmt='none', color='blue', capsize=5, capthick=2, linewidth=2)
        
        for j, (bar, value) in enumerate(zip(bars, values)):
            height = bar.get_height()
            if j == 3: 
                ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.05,
                        f'{healthy_min[i]:.1f}-{healthy_max[i]:.1f}', 
                        ha='center', va='bottom', fontweight='bold', fontsize=8)
            else:
                ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                        f'{value:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
        patient_val = patient_values[i]
        if patient_val < healthy_min[i] or patient_val > healthy_max[i]:
            ax.set_facecolor('#ffeeee')  # Light red background
        ax.set_title(f'{feature}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Value', fontsize=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=9, rotation=45)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_healthy_ranges_chart():
    """Create a chart showing healthy value ranges."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    features = list(HEALTHY_RANGES.keys())
    ranges = list(HEALTHY_RANGES.values())
    
    y_pos = np.arange(len(features))
    
    for i, (feature, (min_val, max_val)) in enumerate(HEALTHY_RANGES.items()):
        ax.barh(i, max_val - min_val, left=min_val, alpha=0.7, 
               color='lightgreen', edgecolor='darkgreen')
        
        ax.text(min_val + (max_val - min_val)/2, i, 
               f'{min_val} - {max_val}', 
               ha='center', va='center', fontweight='bold')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.set_xlabel('Value Range')
    ax.set_title('Healthy Value Ranges for Non-Diabetic Individuals', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig

def add_to_history(patient_data, prediction, probability, risk_label):
    """Add prediction to history."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    history_entry = {
        'Timestamp': timestamp,
        'Pregnancies': patient_data['Pregnancies'].iloc[0],
        'Glucose': patient_data['Glucose'].iloc[0],
        'BloodPressure': patient_data['BloodPressure'].iloc[0],
        'SkinThickness': patient_data['SkinThickness'].iloc[0],
        'Insulin': patient_data['Insulin'].iloc[0],
        'BMI': patient_data['BMI'].iloc[0],
        'DiabetesPedigreeFunction': patient_data['DiabetesPedigreeFunction'].iloc[0],
        'Age': patient_data['Age'].iloc[0],
        'Prediction': 'Diabetic' if prediction == 1 else 'Non-Diabetic',
        'Probability': f"{probability:.1%}",
        'Risk_Level': risk_label
    }
    st.session_state.prediction_history.append(history_entry)

def convert_to_csv(df):
    """Convert dataframe to CSV."""
    return df.to_csv(index=False).encode('utf-8')

def convert_to_excel(df):
    """Convert dataframe to Excel."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Predictions', index=False)
    return output.getvalue()

st.markdown("---")
st.header("🩺 Patient Health Assessment")

with st.expander("📝 Enter Patient Information", expanded=True):
    st.write("Enter the patient's clinical measurements for diabetes risk assessment:")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=2, step=1,
                                    help="Number of times pregnant")
        glucose = st.number_input("Glucose (mg/dL)", min_value=50, max_value=250, value=120, step=1,
                                help="Plasma glucose concentration (Normal: 70-99)")
    with col2:
        blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=40, max_value=150, value=70, step=1,
                                       help="Diastolic blood pressure (Normal: 60-80)")
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=60, value=20, step=1,
                                       help="Triceps skin fold thickness")
    with col3:
        insulin = st.number_input("Insulin (μU/mL)", min_value=0, max_value=600, value=80, step=1,
                                help="2-Hour serum insulin (Normal: 15-200)")
        bmi = st.number_input("BMI (kg/m²)", min_value=15.0, max_value=60.0, value=25.0, step=0.1,
                            help="Body mass index (Normal: 18.5-24.9)", format="%.1f")
    with col4:
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.50, step=0.01,
                            help="Genetic diabetes risk factor", format="%.3f")
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=30, step=1,
                            help="Age in years")

patient_df = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness,
                           insulin, bmi, dpf, age]], columns=FEATURES)

pred, proba = predict_patient(patient_df)
risk_label, risk_emoji = risk_bucket(proba)

st.markdown("### 📋 Prediction Results")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Prediction", f"{'Diabetic' if pred==1 else 'Non-Diabetic'}")
with col2:
    st.metric("Probability", f"{proba:.1%}")
with col3:
    st.metric("Risk Level", f"{risk_emoji} {risk_label}")
with col4:
    if st.button("💾 Save to History", type="primary"):
        add_to_history(patient_df, pred, proba, risk_label)
        st.success("Prediction saved to history!")
        st.rerun()

st.markdown("---")
st.subheader("📊 Patient vs Average Values Comparison")
st.info("This chart compares patient values with population averages and healthy ranges.")
fig1 = create_patient_comparison_chart(data, patient_df)
st.pyplot(fig1)
st.markdown("---")
st.subheader("💚 Healthy Value Ranges Reference")
st.info("Reference chart showing normal/healthy ranges for each health parameter.")
fig2 = create_healthy_ranges_chart()
st.pyplot(fig2)
st.markdown("### 💡 Quick Interpretation")
stats = compute_average_stats(data)
interpretation_points = []

for feature in FEATURES:
    patient_val = float(patient_df[feature].iloc[0])
    healthy_avg = stats[feature]['healthy_avg']
    diabetic_avg = stats[feature]['diabetic_avg']
    min_healthy, max_healthy = HEALTHY_RANGES[feature]
    
    if patient_val < min_healthy:
        interpretation_points.append(f"⚠️ **{feature}**: Your value ({patient_val:.1f}) is below the healthy range ({min_healthy}-{max_healthy})")
    elif patient_val > max_healthy:
        interpretation_points.append(f"⚠️ **{feature}**: Your value ({patient_val:.1f}) is above the healthy range ({min_healthy}-{max_healthy})")

    diff_to_healthy = abs(patient_val - healthy_avg)
    diff_to_diabetic = abs(patient_val - diabetic_avg)
    
    if diff_to_diabetic < diff_to_healthy and patient_val >= min_healthy and patient_val <= max_healthy:
        interpretation_points.append(f"⚡ **{feature}**: Your value ({patient_val:.1f}) is within healthy range but closer to diabetic population average ({diabetic_avg:.1f})")

if interpretation_points:
    st.write("**Areas of Concern:**")
    for point in interpretation_points[:4]:  
        st.write(f"- {point}")
else:
    st.success("✅ All your values are within healthy ranges and align with the healthy population averages.")


# priDICTION HIRSTOY= EVALUATIONN
st.markdown("---")
st.header("📈 Prediction History")

if st.session_state.prediction_history:
    #Convert history to dataframe
    history_df = pd.DataFrame(st.session_state.prediction_history)
    
    #dislay metrices
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Predictions", len(history_df))
    with col2:
        diabetic_count = len(history_df[history_df['Prediction'] == 'Diabetic'])
        st.metric("Diabetic Predictions", diabetic_count)
    with col3:
        non_diabetic_count = len(history_df[history_df['Prediction'] == 'Non-Diabetic'])
        st.metric("Non-Diabetic Predictions", non_diabetic_count)
    with col4:
        high_risk_count = len(history_df[history_df['Risk_Level'] == 'High Risk'])
        st.metric("High Risk Cases", high_risk_count)
    
    #display history table
    st.subheader("📋 Recent Predictions")
    st.dataframe(history_df, use_container_width=True)
    
    #export options
    st.subheader("📥 Export History")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv_data = convert_to_csv(history_df)
        st.download_button(
            label="📄 Download as CSV",
            data=csv_data,
            file_name=f"diabetes_predictions_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        excel_data = convert_to_excel(history_df)
        st.download_button(
            label="📊 Download as Excel",
            data=excel_data,
            file_name=f"diabetes_predictions_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    with col3:
        if st.button("🗑️ Clear History", type="secondary"):
            st.session_state.prediction_history = []
            st.success("History cleared!")
            st.rerun()
    
    #history visulization
    if len(history_df) > 1:
        st.subheader("📊 History Trends")
        #risk leveldistribution
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            risk_counts = history_df['Risk_Level'].value_counts()
            colors = ['green' if x == 'Low Risk' else 'orange' if x == 'Medium Risk' else 'red' for x in risk_counts.index]
            ax.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', colors=colors)
            ax.set_title('Risk Level Distribution')
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            prediction_counts = history_df['Prediction'].value_counts()
            colors = ['lightgreen', 'lightcoral']
            ax.pie(prediction_counts.values, labels=prediction_counts.index, autopct='%1.1f%%', colors=colors)
            ax.set_title('Prediction Distribution')
            st.pyplot(fig)

else:
    st.info("No predictions saved yet. Make a prediction and click 'Save to History' to start tracking your results.")

st.markdown("---")
st.caption("💡 **Note**: This analysis is based on population data and should not replace professional medical consultation.")


