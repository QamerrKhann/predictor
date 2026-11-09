"""
Student Performance Predictor with Advanced ML Pipeline
Author: Iowa State University CS/Math Project
Features: Ensemble models, feature engineering, model persistence, interactive dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
        from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
        from sklearn.preprocessing import StandardScaler, PolynomialFeatures
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
        from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
        from dataclasses import dataclass
from typing import Tuple, Dict, List
import pickle
from pathlib import Path
import io

@dataclass
class ModelMetrics:
        """Data class to store model performance metrics"""
rmse: float
r2: float
mae: float
cv_scores: np.ndarray

def __str__(self) -> str:
        return (f"RMSE: {self.rmse:.4f} | R²: {self.r2:.4f} | "
f"MAE: {self.mae:.4f} | CV Mean: {self.cv_scores.mean():.4f}")

class FeatureEngineer:
        """Advanced feature engineering for student performance data"""

def __init__(self):
self.scaler = StandardScaler()
self.poly = PolynomialFeatures(degree=2, include_bias=False)
self.feature_names = []

def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features from raw data"""
df = df.copy()
        
        # Study efficiency metrics
        if 'study_hours' in df.columns and 'attendance_rate' in df.columns:
df['study_efficiency'] = df['study_hours'] * df['attendance_rate'] / 100
df['engagement_score'] = (df['study_hours'] * 0.6 +
df['attendance_rate'] * 0.4)

        # Academic load indicators
        if 'previous_gpa' in df.columns:
df['gpa_trend'] = df.groupby(level=0)['previous_gpa'].diff().fillna(0)
df['gpa_category'] = pd.cut(df['previous_gpa'],
                            bins=[0, 2.0, 3.0, 3.5, 4.0],
                            labels=['low', 'medium', 'high', 'excellent'])
        
        # Activity balance
        if 'extracurricular_hours' in df.columns and 'study_hours' in df.columns:
df['activity_balance'] = df['study_hours'] / (df['extracurricular_hours'] + 1)
df['total_commitment'] = df['study_hours'] + df['extracurricular_hours']

        # Sleep adequacy indicator
        if 'sleep_hours' in df.columns:
df['sleep_adequate'] = (df['sleep_hours'] >= 7).astype(int)
df['sleep_deviation'] = np.abs(df['sleep_hours'] - 7.5)
        
        return df

def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit scaler and transform features"""
self.feature_names = X.columns.tolist()
X_scaled = self.scaler.fit_transform(X)
        return X_scaled

def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform features using fitted scaler"""
        return self.scaler.transform(X)

class StudentPerformanceModel:
        """Ensemble model for predicting student GPA"""

def __init__(self):
self.feature_engineer = FeatureEngineer()
self.model = None
self.feature_importance = None
self.is_trained = False

def _create_ensemble(self) -> VotingRegressor:
        """Create ensemble of multiple models"""
rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
)

gb = GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=5,
        random_state=42
)

ridge = Ridge(alpha=10.0, random_state=42)

ensemble = VotingRegressor([
                                   ('rf', rf),
            ('gb', gb),
        ('ridge', ridge)
        ])

        return ensemble

def train(self, X: pd.DataFrame, y: pd.Series) -> ModelMetrics:
        """Train the model with hyperparameter tuning"""
        # Feature engineering
X_engineered = self.feature_engineer.create_features(X)
        
        # Select numeric columns only
numeric_cols = X_engineered.select_dtypes(include=[np.number]).columns
        X_numeric = X_engineered[numeric_cols]
        
        # Scale features
X_scaled = self.feature_engineer.fit_transform(X_numeric)
        
        # Split data
X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
)
        
        # Train ensemble
self.model = self._create_ensemble()
        self.model.fit(X_train, y_train)
        
        # Calculate feature importance (from Random Forest component)
rf_model = self.model.named_estimators_['rf']
self.feature_importance = pd.DataFrame({
    'feature': numeric_cols,
            'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        
        # Cross-validation
        cv_scores = cross_val_score(
        self.model, X_scaled, y, cv=5,
        scoring='neg_mean_squared_error'
)

metrics = ModelMetrics(
        rmse=np.sqrt(mean_squared_error(y_test, y_pred)),
r2=r2_score(y_test, y_pred),
mae=mean_absolute_error(y_test, y_pred),
cv_scores=np.sqrt(-cv_scores)
        )

self.is_trained = True
        return metrics

def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data"""
        if not self.is_trained:
raise ValueError("Model must be trained before prediction")

X_engineered = self.feature_engineer.create_features(X)
numeric_cols = X_engineered.select_dtypes(include=[np.number]).columns
        X_numeric = X_engineered[numeric_cols]
X_scaled = self.feature_engineer.transform(X_numeric)
        
        return self.model.predict(X_scaled)

def save(self, path: str):
        """Save model to disk"""
with open(path, 'wb') as f:
        pickle.dump(self, f)

@staticmethod
def load(path: str) -> 'StudentPerformanceModel':
        """Load model from disk"""
with open(path, 'rb') as f:
        return pickle.load(f)

# Streamlit Dashboard
def main():
        st.set_page_config(page_title="Student Performance Predictor", layout="wide")
    
    st.title("🎓 Student Performance Predictor")
    st.markdown("*Advanced ML-based GPA prediction with ensemble models*")
    
    # Sidebar
    st.sidebar.header("Navigation")
page = st.sidebar.radio("Select Page",
                                ["Upload & Train", "Make Predictions", "Model Analysis"])
    
    # Initialize session state
    if 'model' not in st.session_state:
st.session_state.model = None
    if 'metrics' not in st.session_state:
st.session_state.metrics = None
    if 'train_data' not in st.session_state:
st.session_state.train_data = None
    
    if page == "Upload & Train":
upload_and_train_page()
elif page == "Make Predictions":
prediction_page()
    else:
analysis_page()

def upload_and_train_page():
        st.header("Upload Training Data")
    
    # Sample data generator
    if st.button("Generate Sample Dataset"):
sample_df = generate_sample_data(500)
st.session_state.train_data = sample_df
        st.success("Sample data generated! Preview below:")
        st.dataframe(sample_df.head(10))

        # File upload
uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file:
df = pd.read_csv(uploaded_file)
st.session_state.train_data = df
        st.success(f"Loaded {len(df)} records")
        st.dataframe(df.head())

        # Train model
    if st.session_state.train_data is not None:
        if st.button("Train Model", type="primary"):
with st.spinner("Training ensemble model..."):
df = st.session_state.train_data
                
                # Separate features and target
                if 'gpa' not in df.columns:
        st.error("Dataset must contain 'gpa' column as target")
                    return

X = df.drop('gpa', axis=1)
y = df['gpa']

        # Train
        model = StudentPerformanceModel()
metrics = model.train(X, y)

st.session_state.model = model
st.session_state.metrics = metrics
                
                st.success("Model trained successfully!")
                
                # Display metrics
col1, col2, col3, col4 = st.columns(4)
                col1.metric("RMSE", f"{metrics.rmse:.4f}")
                col2.metric("R² Score", f"{metrics.r2:.4f}")
                col3.metric("MAE", f"{metrics.mae:.4f}")
                col4.metric("CV RMSE", f"{metrics.cv_scores.mean():.4f}")

def prediction_page():
        st.header("Make Predictions")
    
    if st.session_state.model is None:
        st.warning("Please train a model first (Upload & Train page)")
        return

                st.subheader("Enter Student Information")

col1, col2 = st.columns(2)

with col1:
study_hours = st.slider("Study Hours per Week", 0, 40, 15)
attendance_rate = st.slider("Attendance Rate (%)", 0, 100, 85)
previous_gpa = st.slider("Previous GPA", 0.0, 4.0, 3.0, 0.1)
sleep_hours = st.slider("Average Sleep Hours", 4, 12, 7)

with col2:
extracurricular_hours = st.slider("Extracurricular Hours/Week", 0, 20, 5)
assignments_completed = st.slider("Assignments Completed (%)", 0, 100, 90)
quiz_average = st.slider("Quiz Average (%)", 0, 100, 80)
participation_score = st.slider("Participation Score", 0, 10, 7)
    
    if st.button("Predict GPA", type="primary"):
        # Create input dataframe
        input_data = pd.DataFrame({
        'study_hours': [study_hours],
        'attendance_rate': [attendance_rate],
        'previous_gpa': [previous_gpa],
        'sleep_hours': [sleep_hours],
        'extracurricular_hours': [extracurricular_hours],
        'assignments_completed': [assignments_completed],
        'quiz_average': [quiz_average],
        'participation_score': [participation_score]
        })

prediction = st.session_state.model.predict(input_data)[0]

        # Display prediction
        st.success(f"Predicted GPA: **{prediction:.2f}**")
        
        # Visualization
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.barh(['Predicted GPA'], [prediction], color='#1f77b4')
        ax.set_xlim(0, 4.0)
        ax.set_xlabel('GPA')
        st.pyplot(fig)

def analysis_page():
        st.header("Model Analysis")
    
    if st.session_state.model is None:
        st.warning("Please train a model first (Upload & Train page)")
        return

model = st.session_state.model
        metrics = st.session_state.metrics
    
    # Metrics
    st.subheader("Performance Metrics")
    st.write(str(metrics))

        # Feature importance
    st.subheader("Feature Importance")
fig, ax = plt.subplots(figsize=(10, 6))
top_features = model.feature_importance.head(10)
    ax.barh(top_features['feature'], top_features['importance'])
    ax.set_xlabel('Importance')
    ax.set_title('Top 10 Most Important Features')
    plt.tight_layout()
    st.pyplot(fig)
    
    # CV scores distribution
    st.subheader("Cross-Validation Scores")
fig, ax = plt.subplots(figsize=(8, 4))
        ax.boxplot(metrics.cv_scores)
    ax.set_ylabel('RMSE')
    ax.set_title('5-Fold Cross-Validation RMSE Distribution')
    st.pyplot(fig)

def generate_sample_data(n_samples: int = 500) -> pd.DataFrame:
        """Generate synthetic student performance data"""
        np.random.seed(42)

data = {
        'study_hours': np.random.normal(15, 5, n_samples).clip(0, 40),
        'attendance_rate': np.random.normal(85, 10, n_samples).clip(50, 100),
        'previous_gpa': np.random.normal(3.0, 0.5, n_samples).clip(0, 4),
        'sleep_hours': np.random.normal(7, 1.5, n_samples).clip(4, 12),
        'extracurricular_hours': np.random.exponential(5, n_samples).clip(0, 20),
        'assignments_completed': np.random.normal(85, 15, n_samples).clip(0, 100),
        'quiz_average': np.random.normal(80, 12, n_samples).clip(0, 100),
        'participation_score': np.random.normal(7, 2, n_samples).clip(0, 10)
    }

df = pd.DataFrame(data)
    
    # Generate target with realistic relationships
        gpa = (
        0.02 * df['study_hours'] +
                0.015 * df['attendance_rate'] +
                0.4 * df['previous_gpa'] +
                0.05 * df['sleep_hours'] +
                0.008 * df['assignments_completed'] +
                0.01 * df['quiz_average'] +
                0.05 * df['participation_score'] +
                np.random.normal(0, 0.2, n_samples)
).clip(0, 4)

df['gpa'] = gpa
    
    return df

if __name__ == "__main__":
main()