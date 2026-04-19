"""
=============================================================================
STREAMLIT: MÜŞTERİ KAYBI (CHURN) TAHMİN ve KARAR DESTEK SİSTEMİ
=============================================================================
Tek dosya, tak-çalıştır yapısı ile profesyonel web arayüzü.
"""

# -----------------------------------------------------------------------------
# KÜTÜPHANE IMPORTLARI
# -----------------------------------------------------------------------------
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from catboost import CatBoostClassifier
import shap
import joblib
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# SAYFA KONFİGÜRASYONU (MUST BE FIRST ST COMMAND)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Churn Tahmin Sistemi | AI Destekli",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# CSS ÖZELLEŞTİRMELERİ (Modern UI)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6b7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .risk-high {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        animation: pulse 2s infinite;
    }
    .risk-low {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
    }
    .action-box {
        background-color: #fef3c7;
        border-left: 5px solid #f59e0b;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# GLOBAL DEĞİŞKENLER (Session State için)
# -----------------------------------------------------------------------------
if 'model' not in st.session_state:
    st.session_state.model = None
if 'label_encoders' not in st.session_state:
    st.session_state.label_encoders = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None
if 'explainer' not in st.session_state:
    st.session_state.explainer = None
if 'df' not in st.session_state:
    st.session_state.df = None

# -----------------------------------------------------------------------------
# 1. SENTETİK VERİ ÜRETİMİ
# -----------------------------------------------------------------------------
@st.cache_data
def generate_synthetic_data(n_samples=2000):
    """Gerçekçi telekom verisi üretir."""
    np.random.seed(42)
    
    data = {
        'Tenure': np.random.randint(1, 73, n_samples),
        'MonthlyCharges': np.random.uniform(18.0, 118.0, n_samples).round(2),
        'Contract': np.random.choice(
            ['Month-to-month', 'One year', 'Two year'], 
            n_samples, 
            p=[0.55, 0.25, 0.20]
        ),
        'TechSupport': np.random.choice(['Yes', 'No'], n_samples, p=[0.35, 0.65]),
        'InternetService': np.random.choice(
            ['DSL', 'Fiber optic', 'No'], 
            n_samples, 
            p=[0.45, 0.40, 0.15]
        )
    }
    
    df = pd.DataFrame(data)
    
    # TotalCharges hesaplama
    noise = np.random.normal(0, 10, n_samples)
    df['TotalCharges'] = (df['Tenure'] * df['MonthlyCharges'] + noise).round(2)
    df['TotalCharges'] = df['TotalCharges'].clip(lower=0)
    
    # Churn skoru hesaplama (gerçekçi mantık)
    churn_score = np.zeros(n_samples)
    churn_score += np.where(df['Contract'] == 'Month-to-month', 0.4, 0)
    churn_score += np.where(df['Contract'] == 'One year', 0.1, 0)
    churn_score += np.where(df['MonthlyCharges'] > 80, 0.25, 0)
    churn_score += np.where((df['MonthlyCharges'] > 60) & (df['MonthlyCharges'] <= 80), 0.15, 0)
    churn_score += np.where(df['Tenure'] <= 12, 0.3, 0)
    churn_score += np.where((df['Tenure'] > 12) & (df['Tenure'] <= 24), 0.15, 0)
    churn_score += np.where(df['TechSupport'] == 'No', 0.2, 0)
    churn_score += np.where(df['InternetService'] == 'Fiber optic', 0.15, 0)
    
    churn_prob = 1 / (1 + np.exp(-(churn_score - 1.2)))
    df['Churn'] = np.random.binomial(1, churn_prob)
    
    return df

# -----------------------------------------------------------------------------
# 2. MODEL EĞİTİMİ
# -----------------------------------------------------------------------------
@st.cache_resource
def train_model(df):
    """CatBoost modelini eğitir ve kaynakları döndürür."""
    try:
        # Kategorik kolonları encode et
        categorical_cols = ['Contract', 'TechSupport', 'InternetService']
        label_encoders = {}
        df_processed = df.copy()
        
        for col in categorical_cols:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])
            label_encoders[col] = le
        
        # Özellikler ve hedef
        X = df_processed[['Tenure', 'MonthlyCharges', 'TotalCharges', 
                         'Contract', 'TechSupport', 'InternetService']]
        y = df_processed['Churn']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )
        
        # Model eğitimi
        model = CatBoostClassifier(
            iterations=300,
            learning_rate=0.08,
            depth=6,
            eval_metric='AUC',
            random_seed=42,
            verbose=False
        )
        
        model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)
        
        # SHAP explainer
        explainer = shap.TreeExplainer(model)
        
        # Performans metrikleri
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        return model, label_encoders, X.columns.tolist(), explainer, auc_score
        
    except Exception as e:
        st.error(f"Model eğitim hatası: {str(e)}")
        return None, None, None, None, 0

# -----------------------------------------------------------------------------
# 3. BAŞLATMA FONKSİYONU (Tak-Çalıştır)
# -----------------------------------------------------------------------------
def initialize_app():
    """Uygulama başlatıldığında modeli ve veriyi hazırlar."""
    try:
        with st.spinner("🔄 Yapay Zeka modeli hazırlanıyor... Lütfen bekleyin."):
            # Veri üret
            df = generate_synthetic_data(2000)
            
            # Model eğit
            model, label_encoders, feature_names, explainer, auc_score = train_model(df)
            
            if model is not None:
                st.session_state.model = model
                st.session_state.label_encoders = label_encoders
                st.session_state.feature_names = feature_names
                st.session_state.explainer = explainer
                st.session_state.df = df
                st.session_state.auc_score = auc_score
                return True
            else:
                return False
                
    except Exception as e:
        st.error(f"Başlatma hatası: {str(e)}")
        return False

# -----------------------------------------------------------------------------
# 4. TAHMİN FONKSİYONU
# -----------------------------------------------------------------------------
def predict_churn(customer_data):
    """Tek müşteri için churn tahmini yapar."""
    try:
        model = st.session_state.model
        label_encoders = st.session_state.label_encoders
        
        # DataFrame oluştur
        input_df = pd.DataFrame([customer_data])
        
        # Encode kategorik değişkenler
        for col in ['Contract', 'TechSupport', 'InternetService']:
            if col in input_df.columns:
                le = label_encoders[col]
                val = input_df[col].iloc[0]
                if val in le.classes_:
                    input_df[col] = le.transform([val])[0]
                else:
                    input_df[col] = 0  # Unknown için default
        
        # Sadece modeldeki özellikleri seç
        features = input_df[st.session_state.feature_names]
        
        # Tahmin
        churn_proba = model.predict_proba(features)[0, 1]
        prediction = 1 if churn_proba >= 0.5 else 0
        
        return prediction, churn_proba
        
    except Exception as e:
        st.error(f"Tahmin hatası: {str(e)}")
        return None, 0

# -----------------------------------------------------------------------------
# 5. SHAP AÇIKLAMA FONKSİYONU
# -----------------------------------------------------------------------------
def get_shap_explanation(customer_data):
    """Müşteri için SHAP değerlerini hesaplar."""
    try:
        model = st.session_state.model
        label_encoders = st.session_state.label_encoders
        explainer = st.session_state.explainer
        
        # DataFrame oluştur ve encode et
        input_df = pd.DataFrame([customer_data])
        for col in ['Contract', 'TechSupport', 'InternetService']:
            if col in input_df.columns:
                le = label_encoders[col]
                val = input_df[col].iloc[0]
                input_df[col] = le.transform([val])[0] if val in le.classes_ else 0
        
        features = input_df[st.session_state.feature_names]
        
        # SHAP değerleri
        shap_values = explainer.shap_values(features)
        
        return shap_values, features
        
    except Exception as e:
        st.error(f"SHAP hatası: {str(e)}")
        return None, None

# -----------------------------------------------------------------------------
# 6. AKSIYON MOTORU
# -----------------------------------------------------------------------------
def get_action_recommendation(churn_proba, monthly_charges):
    """Risk seviyesine göre aksiyon önerisi üretir."""
    is_premium = monthly_charges > 70
    risk_level = "Yüksek" if churn_proba >= 0.7 else ("Orta" if churn_proba >= 0.3 else "Düşük")
    
    if churn_proba >= 0.7 and is_premium:
        return {
            'action': '🎯 Özel İndirim Teklifi',
            'discount': 20,
            'color': 'red',
            'message': f"Churn olasılığı %{churn_proba*100:.1f} ve aylık ücret ${monthly_charges}. "
                      f"KRİTİK MÜŞTERİ! Hemen %20 indirim + yeni kontrat teklifi sunulmalı.",
            'icon': '⚠️'
        }
    elif churn_proba >= 0.7:
        return {
            'action': '📞 Acil Retansiyon Araması',
            'discount': 15,
            'color': 'orange',
            'message': f"Yüksek churn riski (%{churn_proba*100:.1f}). "
                      f"Standart %15 indirim ve teknik destek paketi önerilir.",
            'icon': '📢'
        }
    elif churn_proba >= 0.3:
        return {
            'action': '📋 Proaktif Anket',
            'discount': 5,
            'color': 'yellow',
            'message': f"Orta risk seviyesi (%{churn_proba*100:.1f}). "
                      f"Memnuniyet anketi gönder ve kontrat yenileme hatırlatması yap.",
            'icon': '📊'
        }
    else:
        return {
            'action': '✅ Sadakat Programı',
            'discount': 0,
            'color': 'green',
            'message': f"Düşük risk (%{churn_proba*100:.1f}). "
                      f"Mevcut sadakat programları ile ilişkiyi güçlendir.",
            'icon': '🌟'
        }

# -----------------------------------------------------------------------------
# 7. ANA UYGULAMA - SIDEBAR
# -----------------------------------------------------------------------------
def render_sidebar():
    """Sol menüyü render eder."""
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2922/2922506.png", width=100)
        st.title("🎛️ Müşteri Paneli")
        st.markdown("---")
        
        st.subheader("📊 Müşteri Bilgileri")
        
        # Tenure slider
        tenure = st.slider(
            "⏱️ Müşteri Süresi (Ay)", 
            min_value=1, 
            max_value=72, 
            value=12,
            help="Müşterinin şirketteki kaçıncı ayı"
        )
        
        # Monthly Charges slider
        monthly_charges = st.slider(
            "💰 Aylık Ücret ($)", 
            min_value=18.0, 
            max_value=118.0, 
            value=65.0,
            step=0.5,
            help="Müşterinin aylık ödediği ücret"
        )
        
        # Contract type
        contract = st.selectbox(
            "📄 Kontrat Tipi",
            options=['Month-to-month', 'One year', 'Two year'],
            index=0,
            help="Mevcut kontrat süresi"
        )
        
        # Tech Support
        tech_support = st.selectbox(
            "🔧 Teknik Destek",
            options=['Yes', 'No'],
            index=1,
            help="Teknik destek alıyor mu?"
        )
        
        # Internet Service
        internet_service = st.selectbox(
            "🌐 İnternet Servisi",
            options=['DSL', 'Fiber optic', 'No'],
            index=1,
            help="Kullandığı internet servisi tipi"
        )
        
        # Total Charges (auto-calculated)
        total_charges = round(tenure * monthly_charges + np.random.normal(0, 10), 2)
        
        st.markdown("---")
        
        # Tahmin butonu
        predict_btn = st.button(
            "🚀 Tahmin Et", 
            type="primary",
            use_container_width=True
        )
        
        st.markdown("---")
        st.caption("v1.0 | AI Destekli Churn Tahmin Sistemi")
        
        return {
            'Tenure': tenure,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': max(0, total_charges),
            'Contract': contract,
            'TechSupport': tech_support,
            'InternetService': internet_service
        }, predict_btn

# -----------------------------------------------------------------------------
# 8. ANA UYGULAMA - TABS
# -----------------------------------------------------------------------------
def render_tabs():
    """3 ana sekmeyi oluşturur."""
    return st.tabs(["🎯 Tahmin Paneli", "📈 Model Analitiği", "📋 Veri Seti"])

# -----------------------------------------------------------------------------
# 9. TAHMİN PANELİ İÇERİĞİ
# -----------------------------------------------------------------------------
def render_prediction_tab(tab, customer_data, predict_btn):
    """Tahmin sekmesinin içeriğini render eder."""
    with tab:
        st.markdown('<h2 class="main-header">Müşteri Kaybı Tahmini</h2>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Yapay zeka destekli churn risk analizi ve otomatik aksiyon önerileri</p>', 
                   unsafe_allow_html=True)
        
        if predict_btn:
            with st.spinner("🤖 Yapay zeka analiz ediyor..."):
                # Tahmin yap
                prediction, churn_proba = predict_churn(customer_data)
                
                if prediction is not None:
                    # Sonuç kartları
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        risk_color = "🔴 Yüksek" if churn_proba >= 0.7 else ("🟠 Orta" if churn_proba >= 0.3 else "🟢 Düşük")
                        st.metric(
                            label="Risk Seviyesi",
                            value=risk_color,
                            delta=f"%{churn_proba*100:.1f} Olasılık"
                        )
                    
                    with col2:
                        st.metric(
                            label="Churn Olasılığı",
                            value=f"%{churn_proba*100:.1f}",
                            delta=f"{'Artı' if churn_proba > 0.5 else 'Eksi'} {abs(churn_proba-0.5)*100:.1f}%"
                        )
                    
                    with col3:
                        result_text = "⚠️ AYRILMA RİSKİ" if prediction == 1 else "✅ MÜŞTERİ KALIYOR"
                        st.metric(
                            label="Tahmin Sonucu",
                            value=result_text
                        )
                    
                    # Görsel risk göstergesi
                    st.markdown("---")
                    
                    if churn_proba >= 0.7:
                        st.markdown(f"""
                        <div class="risk-high">
                            <h3>🚨 KRİTİK RİSK TESPİT EDİLDİ</h3>
                            <p>Bu müşterinin churn olasılığı %{churn_proba*100:.1f} - Acil aksiyon gerekiyor!</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif churn_proba >= 0.3:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #f6d365 0%, #fda085 100%); padding: 1.5rem; border-radius: 1rem; color: white;">
                            <h3>⚡ ORTA RİSK SEVİYESİ</h3>
                            <p>Churn olasılığı %{churn_proba*100:.1f} - Proaktif tedbirler alınmalı.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="risk-low">
                            <h3>✅ DÜŞÜK RİSK</h3>
                            <p>Churn olasılığı %{churn_proba*100:.1f} - Müşteri sadakati yüksek.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Aksiyon önerisi
                    st.markdown("---")
                    st.subheader("🎯 Önerilen Aksiyon")
                    
                    action = get_action_recommendation(churn_proba, customer_data['MonthlyCharges'])
                    
                    with st.container():
                        col_icon, col_content = st.columns([1, 10])
                        with col_icon:
                            st.markdown(f"<h1 style='text-align: center;'>{action['icon']}</h1>", 
                                      unsafe_allow_html=True)
                        with col_content:
                            st.markdown(f"""
                            <div class="action-box">
                                <h4>{action['action']}</h4>
                                <p>{action['message']}</p>
                                {f"<strong>Önerilen İndirim: %{action['discount']}</strong>" if action['discount'] > 0 else ""}
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # SHAP Açıklaması
                    st.markdown("---")
                    st.subheader("🔍 Model Kararı Nasıl Verdi? (SHAP Analizi)")
                    
                    with st.spinner("SHAP değerleri hesaplanıyor..."):
                        shap_values, features = get_shap_explanation(customer_data)
                        
                        if shap_values is not None:
                            # SHAP Bar Plot
                            fig, ax = plt.subplots(figsize=(10, 6))
                            
                            # Özellik isimlerini düzgün göster
                            feature_names_display = ['Tenure', 'MonthlyCharges', 'TotalCharges', 
                                                   'Contract', 'TechSupport', 'InternetService']
                            
                            # SHAP değerlerini görselleştir
                            shap.summary_plot(
                                shap_values, 
                                features,
                                feature_names=feature_names_display,
                                plot_type="bar",
                                show=False
                            )
                            
                            plt.title("Bu Müşteri İçin Özellik Etkileri", fontsize=14, fontweight='bold')
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)
                            
                            # Yorum
                            st.info("""
                            **📊 SHAP Yorumu:** Grafikte pozitif değerler churn olasılığını artırırken, 
                            negatif değerler azaltıyor. Örneğin 'Month-to-month' kontrat ve yüksek aylık 
                            ücret churn riskini artıran faktörlerdir.
                            """)
        
        else:
            # İlk açılış durumu
            st.info("👈 Sol panelden müşteri bilgilerini girin ve 'Tahmin Et' butonuna basın.")

# -----------------------------------------------------------------------------
# 10. MODEL ANALİTİĞİ SEKMESİ
# -----------------------------------------------------------------------------
def render_analytics_tab(tab):
    """Model analitiği sekmesini render eder."""
    with tab:
        st.markdown('<h2 class="main-header">Model Performans Analitiği</h2>', unsafe_allow_html=True)
        
        try:
            df = st.session_state.df
            model = st.session_state.model
            label_encoders = st.session_state.label_encoders
            
            # Model metrikleri
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Model AUC Skoru", f"{st.session_state.get('auc_score', 0):.3f}", "Mükemmel")
            with col2:
                st.metric("Eğitim Verisi", "1,600 örnek", "Stratified")
            with col3:
                st.metric("Test Verisi", "400 örnek", "%20")
            
            st.markdown("---")
            
            # Veri dağılımı
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.subheader("📊 Churn Dağılımı")
                fig, ax = plt.subplots(figsize=(8, 6))
                churn_counts = df['Churn'].value_counts()
                colors = ['#2ecc71', '#e74c3c']
                labels = ['Müşteri Kalıyor', 'Müşteri Ayrılıyor']
                ax.pie(churn_counts, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
                st.pyplot(fig)
                plt.close(fig)
            
            with col_right:
                st.subheader("📈 Kontrat Tipine Göre Churn")
                fig, ax = plt.subplots(figsize=(8, 6))
                contract_churn = pd.crosstab(df['Contract'], df['Churn'], normalize='index') * 100
                contract_churn.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'])
                ax.set_ylabel('Yüzde (%)')
                ax.set_xlabel('Kontrat Tipi')
                ax.legend(['Kalıyor', 'Ayrılıyor'])
                ax.tick_params(axis='x', rotation=45)
                st.pyplot(fig)
                plt.close(fig)
            
            # Özellik önemi
            st.markdown("---")
            st.subheader("🎯 Özellik Önem Düzeyleri")
            
            feature_importance = pd.DataFrame({
                'feature': st.session_state.feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(feature_importance['feature'], feature_importance['importance'], 
                   color='steelblue')
            ax.set_xlabel('Önem Skoru')
            ax.set_title('CatBoost Özellik Önem Düzeyleri')
            st.pyplot(fig)
            plt.close(fig)
            
            # Global SHAP
            st.markdown("---")
            st.subheader("🔮 Global Model Açıklanabilirliği (SHAP)")
            
            with st.spinner("Global SHAP analizi hesaplanıyor..."):
                # Sample üzerinden global analiz
                sample_df = df.sample(n=min(100, len(df)), random_state=42)
                for col in ['Contract', 'TechSupport', 'InternetService']:
                    le = label_encoders[col]
                    sample_df[col] = le.transform(sample_df[col])
                
                X_sample = sample_df[st.session_state.feature_names]
                explainer = st.session_state.explainer
                shap_values = explainer.shap_values(X_sample)
                
                fig, ax = plt.subplots(figsize=(12, 8))
                shap.summary_plot(shap_values, X_sample, 
                               feature_names=st.session_state.feature_names,
                               show=False)
                plt.title("Tüm Veri Seti İçin SHAP Özet (Örneklem)", fontsize=14, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                
        except Exception as e:
            st.error(f"Analitik yükleme hatası: {str(e)}")

# -----------------------------------------------------------------------------
# 11. VERİ SETİ SEKMESİ
# -----------------------------------------------------------------------------
def render_data_tab(tab):
    """Veri seti sekmesini render eder."""
    with tab:
        st.markdown('<h2 class="main-header">Eğitim Veri Seti</h2>', unsafe_allow_html=True)
        
        try:
            df = st.session_state.df
            
            # Filtreler
            col1, col2, col3 = st.columns(3)
            with col1:
                contract_filter = st.multiselect(
                    "Kontrat Tipi",
                    options=df['Contract'].unique(),
                    default=df['Contract'].unique()
                )
            with col2:
                churn_filter = st.multiselect(
                    "Churn Durumu",
                    options=[0, 1],
                    format_func=lambda x: "Kalıyor" if x == 0 else "Ayrılıyor",
                    default=[0, 1]
                )
            with col3:
                min_charge, max_charge = st.slider(
                    "Aylık Ücret Aralığı",
                    min_value=float(df['MonthlyCharges'].min()),
                    max_value=float(df['MonthlyCharges'].max()),
                    value=(float(df['MonthlyCharges'].min()), float(df['MonthlyCharges'].max()))
                )
            
            # Filtrele
            filtered_df = df[
                (df['Contract'].isin(contract_filter)) &
                (df['Churn'].isin(churn_filter)) &
                (df['MonthlyCharges'] >= min_charge) &
                (df['MonthlyCharges'] <= max_charge)
            ]
            
            st.markdown(f"**Gösterilen kayıt sayısı:** {len(filtered_df)} / {len(df)}")
            
            # DataFrame göster
            st.dataframe(
                filtered_df.style.background_gradient(subset=['MonthlyCharges', 'TotalCharges'], cmap='Blues'),
                use_container_width=True,
                height=400
            )
            
            # İstatistikler
            st.markdown("---")
            st.subheader("📊 İstatistiksel Özet")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Ortalama Tenure", f"{filtered_df['Tenure'].mean():.1f} ay")
            with col2:
                st.metric("Ortalama Aylık Ücret", f"${filtered_df['MonthlyCharges'].mean():.2f}")
            with col3:
                st.metric("Ortalama Toplam Ücret", f"${filtered_df['TotalCharges'].mean():.2f}")
            with col4:
                churn_rate = filtered_df['Churn'].mean() * 100
                st.metric("Churn Oranı", f"%{churn_rate:.1f}")
            
            # CSV indirme
            st.markdown("---")
            csv = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Veriyi CSV Olarak İndir",
                data=csv,
                file_name='churn_data.csv',
                mime='text/csv',
                use_container_width=True
            )
            
        except Exception as e:
            st.error(f"Veri yükleme hatası: {str(e)}")

# -----------------------------------------------------------------------------
# 12. ANA FONKSİYON
# -----------------------------------------------------------------------------
def main():
    """Ana uygulama akışı."""
    # Başlık
    st.markdown('<h1 class="main-header">🎯 Churn Tahmin Sistemi</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI Destekli Müşteri Kaybı Önleme ve Karar Destek Platformu</p>', 
                unsafe_allow_html=True)
    
    # Başlatma
    if st.session_state.model is None:
        success = initialize_app()
        if not success:
            st.error("Uygulama başlatılamadı. Lütfen sayfayı yenileyin.")
            return
    
    # Başarı mesajı (ilk yüklemede)
    if 'initialized' not in st.session_state:
        st.success(f"✅ Model hazır! AUC Skoru: {st.session_state.auc_score:.3f}")
        st.session_state.initialized = True
    
    # Sidebar
    customer_data, predict_btn = render_sidebar()
    
    # Tabs
    tab1, tab2, tab3 = render_tabs()
    
    # Tab içerikleri
    render_prediction_tab(tab1, customer_data, predict_btn)
    render_analytics_tab(tab2)
    render_data_tab(tab3)

# -----------------------------------------------------------------------------
# UYGULAMA BAŞLATMA
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()