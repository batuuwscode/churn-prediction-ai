"""
=============================================================================
MÜŞTERİ KAYBI (CHURN) TAHMİNİ ve KARAR DESTEK SİSTEMİ
=============================================================================
Rol: Kıdemli Veri Bilimci & ML Mühendisi
Amaç: Telekomünikasyon sektöründe müşteri kaybını tahmin etmek ve 
      aksiyon alınabilir insight'lar üretmek.
Teknolojiler: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, 
              CatBoost, SHAP
=============================================================================
"""

# -----------------------------------------------------------------------------
# 1. KÜTÜPHANE IMPORTLARI
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from catboost import CatBoostClassifier
import shap
import warnings
warnings.filterwarnings('ignore')

# Görselleştirme ayarları
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
np.random.seed(42)

# -----------------------------------------------------------------------------
# 2. SENTETİK VERİ ÜRETİMİ (2000 Satır, Gerçekçi Korelasyonlar)
# -----------------------------------------------------------------------------
def generate_synthetic_telco_data(n_samples=2000):
    """
    Gerçekçi telekomünikasyon verisi üretir.
    Churn mantığı: Yüksek ücret + Kısa süreli müşteri + Kontratsız + Destek yok
    """
    data = {
        'CustomerID': [f'CUST_{str(i).zfill(5)}' for i in range(n_samples)],
        'Tenure': np.random.randint(1, 73, n_samples),  # 1-72 ay
        'MonthlyCharges': np.random.uniform(18.0, 118.0, n_samples).round(2),
        'Contract': np.random.choice(
            ['Month-to-month', 'One year', 'Two year'], 
            n_samples, 
            p=[0.55, 0.25, 0.20]  # Kontratsız müşteriler daha fazla
        ),
        'TechSupport': np.random.choice(['Yes', 'No'], n_samples, p=[0.35, 0.65]),
        'InternetService': np.random.choice(
            ['DSL', 'Fiber optic', 'No'], 
            n_samples, 
            p=[0.45, 0.40, 0.15]
        )
    }
    
    df = pd.DataFrame(data)
    
    # TotalCharges: Tenure * MonthlyCharges + küçük gürültü
    noise = np.random.normal(0, 10, n_samples)
    df['TotalCharges'] = (df['Tenure'] * df['MonthlyCharges'] + noise).round(2)
    df['TotalCharges'] = df['TotalCharges'].clip(lower=0)
    
    # CHURN HEDEF DEĞİŞKENİ (Gerçekçi Mantık)
    # Temel skor hesaplama
    churn_score = np.zeros(n_samples)
    
    # 1. Kontrat tipi etkisi (Month-to-month = yüksek risk)
    churn_score += np.where(df['Contract'] == 'Month-to-month', 0.4, 0)
    churn_score += np.where(df['Contract'] == 'One year', 0.1, 0)
    
    # 2. Ücret etkisi (Yüksek ücret = yüksek risk)
    churn_score += np.where(df['MonthlyCharges'] > 80, 0.25, 0)
    churn_score += np.where((df['MonthlyCharges'] > 60) & (df['MonthlyCharges'] <= 80), 0.15, 0)
    
    # 3. Tenure etkisi (Yeni müşteriler daha kolay ayrılır)
    churn_score += np.where(df['Tenure'] <= 12, 0.3, 0)
    churn_score += np.where((df['Tenure'] > 12) & (df['Tenure'] <= 24), 0.15, 0)
    
    # 4. Teknik destek etkisi
    churn_score += np.where(df['TechSupport'] == 'No', 0.2, 0)
    
    # 5. İnternet servisi etkisi (Fiber kullanıcıları daha hassas)
    churn_score += np.where(df['InternetService'] == 'Fiber optic', 0.15, 0)
    
    # Olasılığa dönüştürme (Sigmoid benzeri)
    churn_prob = 1 / (1 + np.exp(-(churn_score - 1.2)))
    
    # Binary churn üretimi
    df['Churn'] = np.random.binomial(1, churn_prob)
    
    return df

# Veriyi üret
print("=" * 60)
print("SENTETİK TELEKOM VERİSİ ÜRETİLİYOR...")
print("=" * 60)
df = generate_synthetic_telco_data(2000)
print(f"Veri seti boyutu: {df.shape}")
print(f"\nİlk 5 satır:\n{df.head()}\n")

# -----------------------------------------------------------------------------
# 3. KAPSAMLI VERİ ANALİZİ (EDA)
# -----------------------------------------------------------------------------
print("=" * 60)
print("KEŞİFSEL VERİ ANALİZİ (EDA)")
print("=" * 60)

# 3.1 Eksik Değer Kontrolü
print("\n--- Eksik Değer Analizi ---")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0] if missing_values.sum() > 0 else "Eksik değer bulunmamaktadır. ✓")

# 3.2 Hedef Değişken Dağılımı
print("\n--- Hedef Değişken (Churn) Dağılımı ---")
churn_dist = df['Churn'].value_counts(normalize=True) * 100
print(f"Churn=0 (Müşteri Kalıyor): %{churn_dist[0]:.1f}")
print(f"Churn=1 (Müşteri Ayrılıyor): %{churn_dist[1]:.1f}")

# Görselleştirme: Hedef Değişken
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Churn dağılımı pie chart
axes[0, 0].pie(
    df['Churn'].value_counts(), 
    labels=['Müşteri Kalıyor', 'Müşteri Ayrılıyor'],
    autopct='%1.1f%%', 
    startangle=90,
    colors=['#2ecc71', '#e74c3c']
)
axes[0, 0].set_title('Churn Dağılımı', fontsize=14, fontweight='bold')

# Tenure vs Churn
sns.boxplot(data=df, x='Churn', y='Tenure', ax=axes[0, 1])
axes[0, 1].set_title('Tenure vs Churn', fontsize=14, fontweight='bold')
axes[0, 1].set_xticklabels(['Kalıyor', 'Ayrılıyor'])

# MonthlyCharges vs Churn
sns.boxplot(data=df, x='Churn', y='MonthlyCharges', ax=axes[1, 0])
axes[1, 0].set_title('Aylık Ücret vs Churn', fontsize=14, fontweight='bold')
axes[1, 0].set_xticklabels(['Kalıyor', 'Ayrılıyor'])

# Contract vs Churn
contract_churn = pd.crosstab(df['Contract'], df['Churn'], normalize='index') * 100
contract_churn.plot(kind='bar', ax=axes[1, 1], color=['#2ecc71', '#e74c3c'])
axes[1, 1].set_title('Kontrat Tipine Göre Churn Oranı (%)', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Kontrat Tipi')
axes[1, 1].legend(['Kalıyor', 'Ayrılıyor'])
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('eda_analysis.png', dpi=300, bbox_inches='tight')
print("\nEDA görselleştirmesi 'eda_analysis.png' olarak kaydedildi.")
plt.show()

# 3.3 Korelasyon Heatmap'i (Sayısal Değişkenler)
print("\n--- Sayısal Değişkenler Korelasyon Matrisi ---")
numeric_cols = ['Tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']
correlation_matrix = df[numeric_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(
    correlation_matrix, 
    annot=True, 
    cmap='RdBu_r', 
    center=0,
    square=True,
    fmt='.2f',
    cbar_kws={'shrink': 0.8}
)
plt.title('Sayısal Değişkenler Korelasyon Heatmap', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("Korelasyon heatmap'i 'correlation_heatmap.png' olarak kaydedildi.")
plt.show()

# -----------------------------------------------------------------------------
# 4. VERİ ÖN İŞLEME (PREPROCESSING)
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("VERİ ÖN İŞLEME")
print("=" * 60)

# Kopya oluştur
df_processed = df.copy()

# CustomerID'yi ayır (Model için gerekli değil)
customer_ids = df_processed['CustomerID']
df_processed = df_processed.drop('CustomerID', axis=1)

# Kategorik ve sayısal kolonları belirle
categorical_cols = ['Contract', 'TechSupport', 'InternetService']
numerical_cols = ['Tenure', 'MonthlyCharges', 'TotalCharges']

# Label Encoding (Kategorik değişkenler)
# Not: CatBoost kategorik değişkenleri doğal olarak destekler ancak 
# gösterim amacıyla ön işleme adımını ayrı tutuyoruz.
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_processed[col] = le.fit_transform(df_processed[col])
    label_encoders[col] = le
    print(f"'{col}' LabelEncoder ile dönüştürüldü. Sınıflar: {le.classes_}")

# Özellikler ve Hedef
X = df_processed.drop('Churn', axis=1)
y = df_processed['Churn']

# %80 Eğitim - %20 Test Ayrımı
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.20, 
    random_state=42, 
    stratify=y  # Dengesiz dağılımı korumak için
)

print(f"\nEğitim seti boyutu: {X_train.shape}")
print(f"Test seti boyutu: {X_test.shape}")
print(f"Eğitim seti churn oranı: %{y_train.mean()*100:.1f}")
print(f"Test seti churn oranı: %{y_test.mean()*100:.1f}")

# -----------------------------------------------------------------------------
# 5. MODEL EĞİTİMİ (CatBoostClassifier)
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("CATBOOST MODEL EĞİTİMİ")
print("=" * 60)

# CatBoost Classifier
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    eval_metric='AUC',
    random_seed=42,
    verbose=50,
    early_stopping_rounds=50
)

# Model eğitimi
model.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
    verbose=50
)

# Tahminler
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Performans Metrikleri
print("\n--- Model Performansı ---")
print(f"AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Kalıyor', 'Ayrılıyor']))

# Confusion Matrix
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('Gerçek Değer')
plt.xlabel('Tahmin Edilen Değer')
plt.xticks([0.5, 1.5], ['Kalıyor', 'Ayrılıyor'])
plt.yticks([0.5, 1.5], ['Kalıyor', 'Ayrılıyor'], rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Özellik Önem Düzeyleri
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n--- Özellik Önem Düzeyleri ---")
print(feature_importance)

plt.figure(figsize=(8, 5))
sns.barplot(data=feature_importance, x='importance', y='feature', palette='viridis')
plt.title('CatBoost Özellik Önem Düzeyleri', fontsize=14, fontweight='bold')
plt.xlabel('Önem Skoru')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# -----------------------------------------------------------------------------
# 6. MODEL AÇIKLANABİLİRLİĞİ (SHAP)
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SHAP ile MODEL AÇIKLANABİLİRLİĞİ")
print("=" * 60)

# SHAP Explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary Plot (Global Açıklanabilirlik)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title('SHAP Özellik Önemi (Global)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('shap_summary_bar.png', dpi=300, bbox_inches='tight')
print("\nSHAP bar plot 'shap_summary_bar.png' olarak kaydedildi.")
plt.show()

# Detaylı SHAP Summary (Beeswarm plot)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, show=False)
plt.title('SHAP Özellik Etkisi (Detaylı)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('shap_summary_beeswarm.png', dpi=300, bbox_inches='tight')
print("SHAP beeswarm plot 'shap_summary_beeswarm.png' olarak kaydedildi.")
plt.show()

# -----------------------------------------------------------------------------
# 7. DİNAMİK AKSIYON MOTORU (BUSINESS LOGIC)
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("DİNAMİK AKSIYON MOTORU")
print("=" * 60)

class CustomerRetentionEngine:
    """
    Müşteri kaybı riskine göre otomatik aksiyon önerileri üreten
    iş mantığı motoru.
    """
    
    def __init__(self, model, label_encoders, base_threshold=0.70):
        self.model = model
        self.label_encoders = label_encoders
        self.base_threshold = base_threshold
        
    def is_premium(self, monthly_charges):
        """Premium müşteri tanımı: Aylık ücret > 70"""
        return monthly_charges > 70
    
    def predict_churn_probability(self, customer_data):
        """Müşteri için churn olasılığı hesaplar"""
        # DataFrame kontrolü
        if isinstance(customer_data, dict):
            customer_data = pd.DataFrame([customer_data])
        
        # Kategorik değişkenleri dönüştür
        processed_data = customer_data.copy()
        for col in ['Contract', 'TechSupport', 'InternetService']:
            if col in processed_data.columns:
                le = self.label_encoders[col]
                # Bilinmeyen kategorileri handled etmek için
                processed_data[col] = processed_data[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
        
        # Gerekli kolonları seç
        features = processed_data[['Tenure', 'MonthlyCharges', 'TotalCharges', 
                                   'Contract', 'TechSupport', 'InternetService']]
        
        # Tahmin
        proba = self.model.predict_proba(features)[0, 1]
        return proba
    
    def generate_action(self, customer_data):
        """
        Müşteri verisine göre aksiyon önerisi üretir.
        Returns: dict -> aksiyon detayları
        """
        proba = self.predict_churn_probability(customer_data)
        monthly_charges = customer_data['MonthlyCharges']
        
        result = {
            'customer_id': customer_data.get('CustomerID', 'UNKNOWN'),
            'churn_probability': round(proba, 4),
            'risk_level': 'Düşük' if proba < 0.3 else ('Orta' if proba < 0.7 else 'Yüksek'),
            'is_premium': self.is_premium(monthly_charges),
            'recommended_action': 'Beklemede',
            'discount_rate': 0
        }
        
        # YÜKSEK RİSK + PREMIUM = Özel İndirim Teklifi
        if proba >= self.base_threshold and result['is_premium']:
            result['recommended_action'] = 'Özel İndirim Teklifi'
            result['discount_rate'] = 20  # %20 indirim
            result['message'] = (
                f"Müşteri {result['customer_id']} için KRİTİK RISK tespit edildi. "
                f"Churn olasılığı: %{proba*100:.1f}. "
                f"Aylık ücret: ${monthly_charges}. "
                f"ÖNERİLEN AKSIYON: %20 Özel İndirim + Yeni Kontrat Teklifi."
            )
        
        # Yüksek risk ama premium değil = Standart Retansiyon
        elif proba >= self.base_threshold and not result['is_premium']:
            result['recommended_action'] = 'Standart Retansiyon Paketi'
            result['discount_rate'] = 10
            result['message'] = (
                f"Müşteri {result['customer_id']} için yüksek risk. "
                f"Standart %10 indirim ve ücretsiz teknik destek önerilir."
            )
            
        # Orta risk = Proaktif iletişim
        elif 0.3 <= proba < 0.7:
            result['recommended_action'] = 'Proaktif Memnuniyet Anketi'
            result['message'] = (
                f"Müşteri {result['customer_id']} için orta risk seviyesi. "
                f"Memnuniyet anketi gönder ve kontrat yenileme hatırlatması yap."
            )
            
        # Düşük risk = Sadakat programı
        else:
            result['recommended_action'] = 'Sadakat Programı'
            result['message'] = (
                f"Müşteri {result['customer_id']} düşük riskli. "
                f"Mevcut sadakat programları ile ilişkiyi güçlendir."
            )
            
        return result

# Motoru başlat
action_engine = CustomerRetentionEngine(model, label_encoders)

# Test Senaryoları
print("\n--- Test Senaryoları ---")

# Senaryo 1: Yüksek riskli Premium müşteri (Churn olasılığı > 0.70)
test_customer_1 = {
    'CustomerID': 'CUST_TEST_01',
    'Tenure': 5,
    'MonthlyCharges': 95.0,
    'TotalCharges': 475.0,
    'Contract': 'Month-to-month',
    'TechSupport': 'No',
    'InternetService': 'Fiber optic'
}

# Senaryo 2: Düşük riskli müşteri
test_customer_2 = {
    'CustomerID': 'CUST_TEST_02',
    'Tenure': 48,
    'MonthlyCharges': 45.0,
    'TotalCharges': 2160.0,
    'Contract': 'Two year',
    'TechSupport': 'Yes',
    'InternetService': 'DSL'
}

# Senaryo 3: Orta riskli müşteri
test_customer_3 = {
    'CustomerID': 'CUST_TEST_03',
    'Tenure': 15,
    'MonthlyCharges': 65.0,
    'TotalCharges': 975.0,
    'Contract': 'One year',
    'TechSupport': 'No',
    'InternetService': 'DSL'
}

for i, customer in enumerate([test_customer_1, test_customer_2, test_customer_3], 1):
    action = action_engine.generate_action(customer)
    print(f"\n--- Senaryo {i}: {customer['CustomerID']} ---")
    print(f"Churn Olasılığı: {action['churn_probability']}")
    print(f"Risk Seviyesi: {action['risk_level']}")
    print(f"Premium Müşteri: {'Evet' if action['is_premium'] else 'Hayır'}")
    print(f"Önerilen Aksiyon: {action['recommended_action']}")
    print(f"İndirim Oranı: %{action['discount_rate']}")
    print(f"Mesaj: {action['message']}")

# -----------------------------------------------------------------------------
# 8. TOPLU MÜŞTERİ ANALİZİ (Batch Processing)
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TOPLU MÜŞTERİ ANALİZİ (BATCH)")
print("=" * 60)

# Test setindeki ilk 10 müşteriyi analiz et
batch_results = []
test_df = X_test.copy()
test_df['CustomerID'] = customer_ids.iloc[y_test.index].values
test_df['Churn'] = y_test.values

for idx, row in test_df.head(10).iterrows():
    customer_dict = row[['CustomerID', 'Tenure', 'MonthlyCharges', 'TotalCharges', 
                         'Contract', 'TechSupport', 'InternetService']].to_dict()
    
    # LabelEncoder'ları tersine çevir (orijinal değerler için)
    for col in ['Contract', 'TechSupport', 'InternetService']:
        le = label_encoders[col]
        customer_dict[col] = le.inverse_transform([int(customer_dict[col])])[0]
    
    action = action_engine.generate_action(customer_dict)
    batch_results.append({
        'CustomerID': action['customer_id'],
        'Churn_Prob': action['churn_probability'],
        'Risk': action['risk_level'],
        'Premium': action['is_premium'],
        'Action': action['recommended_action'],
        'Discount_%': action['discount_rate']
    })

batch_df = pd.DataFrame(batch_results)
print("\nİlk 10 Müşteri İçin Aksiyon Özet Tablosu:")
print(batch_df.to_string(index=False))

# -----------------------------------------------------------------------------
# 9. SONUÇ ve ÖZET
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("PROJE ÖZETİ")
print("=" * 60)
print(f"""
Proje: Müşteri Kaybı Tahmini ve Karar Destek Sistemi
Veri Seti: {df.shape[0]} satır, {df.shape[1]} sütun
Model: CatBoostClassifier (AUC: {roc_auc_score(y_test, y_pred_proba):.4f})
Anahtar Insight'lar:
1. Kontratsız (Month-to-month) müşteriler en yüksek churn riskini taşır.
2. Aylık ücreti yüksek ve teknik destek almayan müşteriler kritik gruptur.
3. SHAP analizi ile model kararları şeffaf bir şekilde açıklanabilir.
4. Premium + Yüksek Riskli müşteriler için otomatik %20 indirim teklifi 
   aktive edilmiştir.

Üretilen Dosyalar:
- eda_analysis.png
- correlation_heatmap.png
- confusion_matrix.png
- feature_importance.png
- shap_summary_bar.png
- shap_summary_beeswarm.png
""")

print("\nSistem hazır ve çalışıyor! ✅")