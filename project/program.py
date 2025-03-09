#1.Adım - Tanımlamalar, Kurulum ve Kütüphaneler

import os 
import numpy as np # Numpy, çok boyutlu diziler ve matrisler üzerinde çalışmak için yüksek performanslı bir kütüphanedir.
import pandas as pd # Pandas, veri analizi ve işleme için kullanılan yüksek performanslı, kullanımı kolay bir veri yapıları ve fonksiyonları kütüphanesidir.
import matplotlib.pyplot as plt # matplotlib.pyplot, verileri görselleştirmek için kullanılan bir alt kütüphanedir.
import mplfinance as mpf  # mplfinance, finansal grafikler oluşturmanıza olanak tanır.
from matplotlib.dates import DateFormatter # Matplotlib.dates, tarih ve saatlerle ilgili işlemler yapmanıza olanak tanır.
from pandas.plotting import autocorrelation_plot # autocorrelation_plot, veri serisinin otokorelasyonunu çizmek için kullanılır.
from statsmodels.tsa.seasonal import seasonal_decompose #seasonal_decompose, bir zaman serisinin bileşenlerini çıkarmak için kullanılır.
import plotly.express as px # Plotly, etkileşimli grafikler oluşturmanıza olanak tanır.
import seaborn as sns # Seaborn, veri görselleştirmesi için Python kütüphanesidir.
from sklearn.preprocessing import MinMaxScaler # Scikit-learn MinMaxScaler, verileri belirli bir aralığa(0-1) ölçeklemek için kullanılır. 
from sklearn.model_selection import train_test_split # Scikit-learn train_test_split, veri kümesini eğitim ve test alt kümelerine bölmek için kullanılır.
from sklearn.metrics import  # Scikit-learn mean_absolute_percentage_error, ortalama mutlak yüzde hatasını hesaplamak için kullanılır.
import tensorflow as tf # TensorFlow, makine öğrenimi ve derin öğrenme için açık kaynaklı bir platformdur.
from keras import Model # Keras, derin öğrenme için yüksek seviyeli bir API'dir.
from tensorflow.keras.models import Sequential #Sequential, Keras API'sinde model oluşturmaya en uygun yoldurç
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, LeakyReLU, ELU, ReLU # LSTM, uzun kısa süreli bellek ağıdır.
# Dense, tam bağlantılı katmandır. Dropout, aşırı uyumu azaltmak için kullanılır. Input, modelin girişini tanımlar. 
# #LeakyReLU, ReLU'nun türevi olan bir aktivasyon fonksiyonudur. ELU, ReLU'nun yumuşak bir versiyonudur.
from tensorflow.keras.optimizers import Adam, AdamW, RMSprop, Nadam # Adam, AdamW, RMSprop ve Nadam, optimizasyon algoritmalarıdır. 
#Adam en yaygın kullanılan optimizasyon algoritması olup öğrenme oranını ayarlamaktadır. 
# #Nadam Adam ve Nesterov Momentum'un birleşimidir. AdamW ise ağırlıkların düzenlileştirilmesi için Adam optimizasyon algoritmasının bir varyasyonudur.
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint # EarlyStopping, modelin aşırı uyumu önlemek için kullanılır. ModelCheckpoint, modelin ağırlıklarını kaydetmek için kullanılır.
from sklearn.metrics import r2_score # r2_score, r-kare skoru hesaplamak için kullanılır. r-kare skoru, modelin ne kadar iyi uyum sağladığını gösterir.
import optuna 
# Optuna hiperparametre optimizasyonu sürecini otomatize etmek için geliştirilmiş ve en iyi performans için deneme yanılma yoluyla optimum değerleri otomatik olarak arayıp bulan bir framework.
# Optuna, optimum hiperparametre değerlerini tespit edebilmek için parametre denemelerinin kaydından faydalanır. 
# Geçmiş verileri kullanarak gelecek vaat eden bir alanı tahmin eder ve o alandaki değerleri dener. 
# Ulaştığı yeni sonuca dayanarak daha da umut vaat eden bir bölge seçimi gerçekleştirerek gelişen bir yapıya sahiptir. 
# Optimizasyon süreci devam ederken uygulanan denemelerin (trial) geçmiş verilerini kullanarak işlemleri tekrarlar. 
#Kodumuzdaki hiperparametreler, LSTM katmanlarının sayısı, gizli katmanların sayısı, dropout oranı, optimizasyon algoritması ve aktivasyon fonksiyonları gibi değerlerdir.
# Optuna ile hiperparametre optimizasyonu yaparken, belirli bir aralıkta rastgele değerler seçilir ve bu değerler modelin performansını ölçmek için kullanılır.
# Bu değerlerin performansı ölçüldükten sonra, en iyi performansı veren hiperparametreler seçilir ve model bu hiperparametrelerle eğitilir.
# Bu süreç, belirli bir sayıda deneme yapılıncaya kadar devam eder ve en iyi performansı veren hiperparametreler seçilir.

#TensorFlow log seviyesini ayarladım
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

df = pd.read_csv('usd_try_daily_data.csv') # usd_try_daily_data.csv dosyasını okuyarak veri setini oluşturduk.
df
df.info() 
df.describe()
df['date'] = pd.to_datetime(df['date']) 
df.info() 
df.isnull().sum().sum() 
df 

#2. Adım - Veri Görselleştirme
df_avg=pd.DataFrame(df) #df_avg adında bir DataFrame oluşturduk 

plt.figure(figsize=(12, 6))  # boyutu belirledik

# veriyi çizdik
plt.plot(df['date'], df['close'], color='navy', linewidth=2, label='Dolar/TL Kapanış Fiyatı')
df_avg['Moving_Avg'] = df['close'].rolling(window=10).mean()  # 10-Günlük Hareketli Ortalama
plt.plot(df['date'], df_avg['Moving_Avg'], label='10-Günlük Hareketli Ortalama (MA10)', color='red')
plt.legend()

# Eksen ve başlık
plt.title('Son 5 Yıllık Fiyat Geçmişi (2020 - Günümüz)', fontsize=16, fontweight='bold')
plt.xlabel('Tarih', fontsize=14, labelpad=10)
plt.ylabel('Dolar/TL', fontsize=14, labelpad=10)

# Eksenlerin boyutunu ve döndürme açısını ayarladık
plt.xticks(fontsize=12, rotation=45)
plt.yticks(fontsize=12)

# Izgara ekledik
plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
plt.legend(fontsize=12, loc='upper left')

# Grafikleri sıkılaştırdık
plt.tight_layout()

# Görseli göster
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(df['close'], kde=True, bins=30, color='green') # Histogram çizdik
plt.title('USD/TRY Kapanış Fiyatlarının Dağılımı') # Başlık
plt.xlabel('Dolar/TL') # X eksenini belirledik
plt.ylabel('Frekans')
plt.grid()
plt.show()

plt.figure(figsize=(10, 5))
autocorrelation_plot(df['close'])
plt.title('Dolar/TL fiyatlarının otokorelasyonu') # Otokorelasyon grafiğini çizdik otokorelasyon, bir zaman serisinin kendi kendine benzerliğini ölçer.
plt.show()

decomposition = seasonal_decompose(df['close'], model='additive', period=365) # Veri setinin yıllık bileşenlerini çıkardık
decomposition.plot()
plt.show()

df_timeless=df.drop(columns=['date'],axis=1) # Tarih sütununu çıkardık
df_timeless

#3.Adım - Veri Ön İşleme

df_test=df_timeless.iloc[1185:].astype(np.float32)  # Test verisi için son 4 ay
df_val=df_timeless.iloc[1065:1185].astype(np.float32)  # Doğrulama verisi için ondan önceki son 4 ay
df_train=df_timeless.iloc[:1065].astype(np.float32)  # Eğitim verisi için 2020-den son 8 ay
# Eğitim, doğrulama ve test verisi için tarihler
df_train['date'] = pd.to_datetime(df['date'].iloc[:1065].values)
df_val['date'] = pd.to_datetime(df['date'].iloc[1065:1185].values)
df_test['date'] = pd.to_datetime(df['date'].iloc[1185:].values)
df_train.info() 

plt.figure(figsize=(12, 6)) # boyutu belirledik

# Eğitim verisini çizdik
plt.plot(df_train['date'], df_train['close'], label='Eğitim', color='blue')

# Doğrulama verisini çizdik
plt.plot(df_val['date'], df_val['close'], label='Doğrulama', color='yellow')

# Test verisini çizdik
plt.plot(df_test['date'], df_test['close'], label='Test', color='red')

# Eksen ve başlık
plt.xlabel('Tarih')
plt.ylabel('Dolar/TL')
plt.title('Parite Fiyatı : Eğitim, Doğrulama ve Test')
plt.legend()
plt.grid()
plt.savefig('usd_try_fiyat_grafigi_asamalari.png', dpi=300)
# Görseli gösterdik
plt.show()

#ölçeklenecek özelliklerin listesi (hedef 'close' hariç)
features_to_scale = ['open', 'high', 'low']

# Ölçekleyicileri başlat
scaler = MinMaxScaler(feature_range=(0, 1))  # Özellikler için
scaler_y = MinMaxScaler(feature_range=(0, 1))  # Hedef 'close' için

# Ölçekleyicileri uyumlu hale getirmek için
scaler.fit(df_train[features_to_scale])  # Giriş özelliklerine uyum sağlar
scaler_y.fit(df_train[['close']].values.reshape(-1, 1))  # Sadece hedef 'close' için uyum sağlar

# Özellikleri dönüştürdük
df_train[features_to_scale] = scaler.transform(df_train[features_to_scale]) 
df_val[features_to_scale] = scaler.transform(df_val[features_to_scale])
df_test[features_to_scale] = scaler.transform(df_test[features_to_scale])

# 'close' sütununu dönüştür ve DataFrame'deki 'close' sütununu güncelle
df_train['close'] = scaler_y.transform(df_train[['close']].values.reshape(-1, 1))
df_val['close'] = scaler_y.transform(df_val[['close']].values.reshape(-1, 1))
df_test['close'] = scaler_y.transform(df_test[['close']].values.reshape(-1, 1))

# Eğitim, doğrulama ve test setleri için hedef değişkenleri hazırla
y_train = df_train[['close']].values
y_val = df_val[['close']].values
y_test = df_test[['close']].values

# Dizi oluşturmak için fonksiyon (X için yalnızca özelliklerin kullanılmasını sağlar) dönüşüm işlemi yapar
def create_sequences(df_timeless, target_col, look_back=30, feature_cols=None): # Zaman serileri oluşturmak için fonksiyon
    X, y = [], []
    for i in range(len(df_timeless) - look_back): 
        X.append(df_timeless[feature_cols].iloc[i:i+look_back].values)  # X için yalnızca seçilen özellikleri kullan
        y.append(df_timeless.iloc[i + look_back][target_col])  # y için 'close' kullan
    return np.array(X), np.array(y) 

# X için kullanılacak özelliklerin listesi (hedef 'close' hariç)
features = ['open', 'high', 'low']

# Eğitim, doğrulama ve test setleri için dizileri oluştur
X_train, y_train = create_sequences(df_train, target_col='close', look_back=30, feature_cols=features)
X_val, y_val = create_sequences(df_val, target_col='close', look_back=30, feature_cols=features)
X_test, y_test = create_sequences(df_test, target_col='close', look_back=30, feature_cols=features)

y_train = np.reshape(y_train, (-1, 1))
y_val = np.reshape(y_val, (-1, 1))
y_test = np.reshape(y_test, (-1, 1))

# Oluşan dizilerin şekillerini kontrol ettik
print(f"Eğitim X şekli: {X_train.shape}, Y şekli: {y_train.shape}")
print(f"Doğrulama X şekli: {X_val.shape}, Y şekli: {y_val.shape}")
print(f"Test X şekli: {X_test.shape}, Y şekli: {y_test.shape}")

#4.Adım - LSTM

# LSTM modelini oluştur zaman serisi verilerini tahminlemek için
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),  
    LSTM(64, return_sequences=False),
    Dense(32, activation='relu'),  # Ara öğrenme için yoğun katman
    Dense(1)  # Çıktı katmanı ('Close' fiyatını tahmin etmek)
])

# Modeli derle
learning_rate = 0.001  # Oluşturulan modeli derlerken kullanılacak öğrenme oranı

custom_optimizer = Adam(learning_rate=learning_rate) # Adam optimizasyon algoritması seçildi
model.compile(optimizer=custom_optimizer, loss='mean_squared_error') # Modeli derlerken kullanılacak kayıp fonksiyonu
early_stop = EarlyStopping(monitor='val_loss', patience=1000, restore_best_weights=True,verbose=1) # Aşırı uyumu önlemek için EarlyStopping kullanıldı

# Model özeti
model.summary()

# Modeli eğit
history = model.fit(
    X_train, y_train, # Eğitim verileri
    validation_data=(X_val, y_val), # Doğrulama verileri
    epochs=150, # Epoch sayısı (tekrar sayısı)
    batch_size=64, # Batch boyutu (GPU performansı için ayarlandı)
    verbose=1,callbacks=[early_stop] # EarlyStopping kullanıldı ve eğitim süreci başlatıldı
)

test_loss = model.evaluate(X_test, y_test, verbose=0) # Test verileri üzerinde modelin performansını değerlendir
r2 = r2_score(y_test, model.predict(X_test)) # R² skorunu hesapla
print(f"Test verisi üzerinde R² Skoru: {r2:.4f}") # R² skorunu yazdır
print(f"Test Kaybı (OKA): {test_loss:.4f}") #Mean Squared Error (MSE) Ortalama Kare Hatası değeri



# OPTUNA İLE HİPERPARAMETRE OPTİMİZASYONU

# Birden fazla GPU için MirroredStrategy kullanılabilir benim bilgisayarımda bir GPU olduğu için OneDeviceStrategy kullandım.
strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0") # Bir GPU varsa
print("Number of devices:", strategy.num_replicas_in_sync)  # GPU sayısını yazdırır
#optuna.delete_study(study_name="time_series_optimization_gpu", storage="sqlite:///optuna_study_gpu.db") # SQLite veritabanındaki mevcut çalışmayı siler
with strategy.scope(): # Strategy kapsamındaki her şey 
    # Optuna hiperparametre optimizasyonunu çalıştır
    def objective(trial):
        # Hyperparametre aralıklarını belirle
        lstm_units = trial.suggest_int("lstm_units", 128, 256, step=128) # LSTM birim sayısı
        dense_units = trial.suggest_int("dense_units", 16, 128, step=16) # Yoğun katman birim sayısı
        dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.5) # Dropout oranı aşırı uyumu azaltmak için kullanılır
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "Nadam"]) # Optimizasyon algoritması seçimi
        activation_function = trial.suggest_categorical("activation_function", ["relu", "leaky_relu", "elu"]) # Aktivasyon fonksiyonu seçimi

        # Optimizasyon algoritmasını seçtik
        if optimizer_name == "Adam":
            optimizer = Adam(learning_rate=0.00025)
        elif optimizer_name == "AdamW":
            optimizer = AdamW(learning_rate=0.00025)
        elif optimizer_name == "Nadam":
            optimizer = Nadam(learning_rate=0.00025)
            #return_sequences=True, input_shape=(look_back, len(df_train_x.columns))

        # LSTM modelini oluştur
        model = Sequential([
            Input(shape=(X_train.shape[1], X_train.shape[2])),
            LSTM(lstm_units, return_sequences=True),
            Dropout(dropout_rate),
            LSTM(lstm_units, return_sequences=True),
            Dropout(dropout_rate),
            LSTM(lstm_units, return_sequences=False),
            Dropout(dropout_rate),
            Dense(dense_units, activation=None),  # Aktivasyon fonksiyonu yok
            LeakyReLU(negative_slope=0.1) if activation_function == "leaky_relu" else 
            ELU(alpha=1.0) if activation_function == "elu" else 
            ReLU(),
            Dense(1)  # Çıktı katmanı ('Close' fiyatını tahmin etmek)
        ])
        model.compile(optimizer=optimizer, loss='mean_squared_error') # Modeli derle

        # EarlyStopping kullanarak modeli eğit
        early_stop = EarlyStopping(
            monitor='val_loss', # Doğrulama kaybını izle
            patience=20, # Sabır sayısı
            restore_best_weights=True, # En iyi ağırlıkları geri yükle
            verbose=1 # Eğitim sürecini göster
        )

        # Modeli eğit
        model.fit(
            X_train, y_train, # Eğitim verileri
            validation_data=(X_val, y_val), # Doğrulama verileri
            epochs=100, # Epoch sayısı
            batch_size=128,  # Batch boyutu
            verbose=0, # Eğitim sürecini gösterme
            callbacks=[early_stop] # EarlyStopping kullan
        )

        # Test verileri üzerinde modelin performansını değerlendir
        y_test_pred = model.predict(X_test) #,batch_size=128)  # Test verileri üzerinde tahmin yap
        #y_test_pred = y_test_pred.flatten()  # Diziyi düzleştir
        r2 = r2_score(y_test, y_test_pred) # R² skorunu hesapla

        # Eğitilen modeli kaydet
        model.save('trained_model.h5')  

        return 1 - r2  # Optuna minimize eder bu yüzden 1 - r² döndür

    # Optuna çalışma adı ve depolama adını belirle
    study_name = "time_series_optimization_gpu"  # Optuna çalışma adı Zaman Serisi Optimizasyonu
    storage_name = "sqlite:///optuna_study_gpu.db"  # SQLite veritabanı adı

    # Optuna çalışmasını oluştur veya yükle
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="minimize",
        load_if_exists=True
    )

    # Hiperparametre optimizasyonunu çalıştır optuna'nın optimize fonksiyonunu kullanarak
    study.optimize(objective, n_trials=50)

    # En iyi denemenin sonucunu yazdırdık
    print("En iyi deneme:")
    trial = study.best_trial
    print(f"Test üzerindeki R² skoru: {1 - trial.value:.4f}") 
    print(f"En iyi Hiperparametreler: {trial.params}") 


#5.Adım - Değerlendirme

result = model.evaluate(X_test, y_test) # Test verileri üzerinde modelin performansını değerlendirdik
y_pred = model.predict(X_test)   # Test verileri üzerinde tahmin yaptık

y_test_true = scaler_y.inverse_transform(y_test) # Gerçek değerleri ters dönüştürdük
y_test_pred = scaler_y.inverse_transform(y_pred) # Tahmin edilen değerleri ters dönüştürdük

# Gerçek ve tahmin edilen değerleri DataFrame'e dönüştürdük
df_final = pd.DataFrame(df[:1185])
df_final_test=pd.DataFrame(df[1185:])

# Gerçek değerleri DataFrame'e ekledik
aligned_dates = df_final_test['date'][30:].reset_index(drop=True)

# Tahmin edilen değerleri DataFrame'e ekledik
y_test_pred_DF = pd.DataFrame({
    'date': aligned_dates[:len(y_test_pred)],  # Tahminleri gerçek tarihlerle hizaladık
    'predicted_close': y_test_pred.flatten()  # Diziyi düzleştirdik
})

# Gerçek ve tahmin edilen değerleri birleştirdik

df_test2=pd.DataFrame(df[1215:]).reset_index(drop=True) 

y_test_pred_df=pd.DataFrame(y_test_pred_DF,columns=['predicted_close']).reset_index(drop=True)
y_test_pred_df = pd.concat([df_test2, y_test_pred_df], axis=1) 

# Verileri çizdik
print(y_test_pred_df)

plt.figure(figsize=(12, 6), facecolor='white')

# Eğitim verisini çizdik
plt.plot(df_final['date'], df_final['close'], color='blue')
plt.plot(df_final_test['date'], df_final_test['close'], color='black')
plt.plot(y_test_pred_df['date'], y_test_pred_df['predicted_close'], color='red')

# Eksen ve başlık
plt.xlabel('Tarih')
plt.ylabel('Fiyat')
plt.title('Dolar/TL Fiyatı: Eğitim, Test ve Tahmin')
plt.legend()
plt.grid()

plt.show()


df_final['date'] = pd.to_datetime(df_final['date'])
df_final_test['date'] = pd.to_datetime(df_final_test['date'])
y_test_pred_df['date'] = pd.to_datetime(y_test_pred_df['date'])

# Doğruluk değerini hesapladık
accuracy = 97.48 

# --- Veri Hazırlama ---

# Son 8 ay için eğitim verilerini seçtik
train_last_8_months = df_final[df_final['date'] >= (df_final['date'].max() - pd.DateOffset(months=8))]

# Hazırlanan verileri kontrol ettik
train_candle_data = train_last_8_months[['date', 'open', 'high', 'low', 'close']].copy()
train_candle_data.set_index('date', inplace=True)

# Test verilerini kontrol ettik
test_candle_data = df_final_test[['date', 'open', 'high', 'low', 'close']].copy()
test_candle_data.set_index('date', inplace=True)

# Tahmin edilen verileri kontrol ettik
prediction_candle_data = y_test_pred_df[['date', 'open', 'high', 'low', 'predicted_close']].copy()
prediction_candle_data.rename(
    columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'predicted_close': 'close'
    },
    inplace=True
)
prediction_candle_data.set_index('date', inplace=True)

# --- Çizim ---

# Oluşturulacak figür ve alt grafiklerin düzeni
fig = plt.figure(figsize=(20, 20))
gs = fig.add_gridspec(2, 1)  # 2 satır, 1 sütun 

# --- Mum Grafiği (Eğitim, Test ve Tahminler) ---

# Alt grafiklerden biri için
ax1 = fig.add_subplot(gs[0, 0])

# Eğitim Verilerini Mum Grafiği Olarak Çizdirdik
mpf.plot(
    train_candle_data,
    type='candle',
    style='yahoo', 
    show_nontrading=True,
    ax=ax1,
    ylabel='Price'
)

custom_marketcolors_test = mpf.make_marketcolors(
    up='lime',         # yukarı doğru mumlar için renk
    down='crimson',    # aşağı doğru mumlar için renk
    edge='black',      # kenar rengi
    wick='black',      # fitil rengi
)


custom_style_test = mpf.make_mpf_style(
    base_mpf_style='yahoo', 
    marketcolors=custom_marketcolors_test
)

# Test Verilerini Mum Grafiği Olarak Çizdirdik
mpf.plot(
    test_candle_data,
    type='candle',
    style=custom_style_test,
    show_nontrading=True,
    ax=ax1
)

custom_marketcolors_pred = mpf.make_marketcolors(
    up='green',         # yukarı doğru mumlar için renk
    down='red',     # aşağı doğru mumlar için renk
    edge='black',      # Kenar rengi
    wick='black',      # Fitil rengi
)

custom_style_pred = mpf.make_mpf_style(
    base_mpf_style='yahoo',
    marketcolors=custom_marketcolors_pred
)

# Tahmin Edilen Verileri Mum Grafiği Olarak Çizdirdik
mpf.plot(
    prediction_candle_data,
    type='candle',
    style='yahoo',
    show_nontrading=True, 
    ax=ax1
)

ax1.plot(
    prediction_candle_data.index,
    prediction_candle_data['close'],
    color='red',
    alpha=1,
    linestyle='-',
    label='tahmin edilen kapanış fiyatı'
)

ax1.grid(True, linestyle='--', linewidth=0.5)

ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
fig.autofmt_xdate()  # X eksenindeki tarihleri döndürdük

# Eksen ve başlık
ax1.set_title('Mum grafiği: Eğitim, Test, ve Tahmin verileri', fontsize=18)

# --- Çizgi grafiği (eğitim,test ve tahmin) ---

# Diğer alt grafik için
ax2 = fig.add_subplot(gs[1, 0])

# Eğitim Verilerini Çizgi Grafiği Olarak Çizdirdik
ax2.plot(train_last_8_months['date'], train_last_8_months['close'], color='blue', label='Eğitim verisi (son 8 ay)')
ax2.plot(df_final_test['date'], df_final_test['close'], color='black', label='Test verisi')
ax2.plot(y_test_pred_df['date'], y_test_pred_df['predicted_close'], color='red', label='Tahmin edilen veri')


ax2.set_xlabel('tarih', fontsize=18)
ax2.set_ylabel('Dolar/TL ($)', fontsize=20)
ax2.legend(loc='upper left', fontsize=20)
ax2.grid(True)

# Eksenleri döndürdük
ax2.text(0.01, 0.95, f'Kesinlik: {accuracy:.2f}%', transform=ax2.transAxes, fontsize=12, color='green', ha='left')

ax2.set_title('Eğitim, test ve tahmin verisi', fontsize=18)

ax1.tick_params(axis='x', labelsize=20)  
ax1.tick_params(axis='y', labelsize=20)  
ax2.tick_params(axis='x', labelsize=20)  
ax2.tick_params(axis='y', labelsize=20)  

plt.tight_layout()
plt.savefig('fiyat_grafigi_tahmin.png', dpi=300)
plt.show()


#6.Adım - Gelecek Fiyat Tahmini

# Ölçeklenecek özelliklerin listesi (hedef 'close' hariç)
features_to_scale = ['open', 'high', 'low']

# Ölçekleyicileri başlat
scaler = MinMaxScaler(feature_range=(0, 1))  # Özellikler için
scaler_y = MinMaxScaler(feature_range=(0, 1))  # Hedef 'close' için

# Ölçekleyicileri uyumlu hale getir
scaler.fit(df_timeless[features_to_scale])  # sadece giriş özelliklerine uyum sağlar
scaler_y.fit(df_timeless[['close']].values.reshape(-1, 1))  # Sadece hedef 'close' için uyum sağlar

# Özellikleri dönüştür
df_timeless[features_to_scale] = scaler.transform(df_timeless[features_to_scale])

# 'close' sütununu dönüştür ve DataFrame'deki 'close' sütununu güncelle
df_timeless['close'] = scaler_y.transform(df_timeless[['close']].values.reshape(-1, 1))

# Eğitim verileri için hedef değişkenleri hazırla
y_train_future = df_timeless[['close']].values

# Dizi oluşturmak için fonksiyon (X için yalnızca özelliklerin kullanılmasını sağlar)
def create_sequences(df_timeless, target_col, look_back=30, feature_cols=None):
    X, y = [], []
    for i in range(len(df_timeless) - look_back):
        X.append(df_timeless[feature_cols].iloc[i:i+look_back].values)  # X için yalnızca seçilen özellikleri kullan
        y.append(df_timeless.iloc[i + look_back][target_col])  # y için 'close' kullan
    return np.array(X), np.array(y)

# X için kullanılacak özelliklerin listesi (hedef 'close' hariç)
features = ['open', 'high', 'low']

# Eğitim verileri için dizileri oluştur
X_train_future, y_train_future = create_sequences(df_timeless, target_col='close', look_back=30, feature_cols=features)
y_train_future = np.reshape(y_train_future, (-1, 1))

# Oluşan dizilerin şekillerini kontrol ettik
print(f"Eğitim X_gelecek şekil: {X_train_future.shape}, y_gelecek şekil: {y_train_future.shape}")


# LSTM modelini oluştur
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train_future.shape[1], X_train_future.shape[2])),  
    LSTM(64, return_sequences=False),
    Dense(32, activation='relu'),  #kullanılan aktivasyon fonksiyonu
    Dense(1)  # Çıktı katmanı ('Close' fiyatını tahmin etmek)
])

# Modeli derle
learning_rate = 0.001  # Oluşturulan modeli derlerken kullanılacak öğrenme oranı

custom_optimizer = Adam(learning_rate=learning_rate) # Adam optimizasyon algoritması seçildi
model.compile(optimizer=custom_optimizer, loss='mean_squared_error') # Modeli derlerken kullanılacak kayıp fonksiyonu
early_stop = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True,verbose=1) # Aşırı uyumu önlemek için EarlyStopping kullanıldı

# Model özeti
model.summary()

# Modeli eğit
history = model.fit(
    X_train_future, y_train_future,
    epochs=100,
    batch_size=64,
    verbose=1
)

# Gelecek fiyat tahmini yap df_timeless veri seti üzerinde 
df_timeless_features = df_timeless.drop(columns=['close']) 
look_back=30 # 30 tane zaman serisi oluşturuldu
last_sequence = df_timeless_features.iloc[-look_back:].values.reshape(1, look_back, X_train_future.shape[2]) # Son 30 zaman serisini al

future_prediction_x = model.predict(last_sequence) # Gelecek fiyat tahmini yap
future_prediction = scaler_y.inverse_transform(future_prediction_x) # Tahmin edilen fiyatı ters dönüştür
print(f"Gelecek Tahmini: {future_prediction[0][0]}") # Gelecek fiyat tahminini yazdır

look_back = 30  # 30 tane zaman serisi oluşturuldu
current_sequence = X_train_future[-1:].copy()  # Son 30 zaman serisini al
future_predictions = []

for _ in range(10):  # 10 günlük tahmin yap
    
    next_pred = model.predict(current_sequence)
    #next_pred = scaler_y.inverse_transform(next_pred_X)
    future_predictions.append(next_pred[0][0])  # Tahmin edilen fiyatı listeye ekle

    # Yeni bir özellik vektörü oluştur
    new_feature_vector = current_sequence[:, -1:, :].copy()  # Kopyala ve son zaman serisini al
    new_feature_vector[0, 0, -1] = next_pred[0][0]  # Yeni özellik vektörüne tahmin edilen fiyatı ekle

    # Yeni özellik vektörünü mevcut zaman serisine ekle
    current_sequence = np.append(current_sequence[:, 1:, :], new_feature_vector, axis=1)

print(f"10 Günlük Tahminler: {future_predictions}")

# Gelecek fiyat tahminlerini ters dönüştür
next_pred_y = scaler_y.inverse_transform(np.array(future_predictions).reshape(-1, 1))
print(next_pred_y)


# Gelecek 10 gün için tahmin edilen fiyatları oluştur
prediction_dates = pd.date_range(start="2025-01-03", periods=10)

# Tahmin edilen fiyatları DataFrame'e dönüştür
predicted_df = pd.DataFrame({
    "Date": prediction_dates,
    "Predicted_Y": next_pred_y.flatten()  # Diziyi düzleştir
})
predicted_df
# Son olarak gelecek 10 gün için tahmin edilen fiyatları ekrana yazdırdık.
print(predicted_df)