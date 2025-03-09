import yfinance as yf
import pandas as pd

# USD/TRY paritesinin verilerini çekme
symbol = "USDTRY=X"
data = yf.download(symbol, start="2020-01-01", end="2025-01-03", interval="1d")

# İstediğiniz sütunları seçme
data = data[['Open', 'High', 'Low', 'Close']]

# Veriye tarih sütunu ekleme (index olarak zaten tarih var, ancak 'Date' olarak yeniden düzenliyoruz)
data['Date'] = data.index

# Tarih formatını ay/gün/yıl şeklinde düzenleme ve 0'lı ayları kaldırma
data['Date'] = data['Date'].dt.strftime('%-m/%-d/%Y')

# Verideki tüm sayısal sütunları virgülden sonra 2 basamağa yuvarlama
data['Open'] = data['Open'].round(2)
data['High'] = data['High'].round(2)
data['Low'] = data['Low'].round(2)
data['Close'] = data['Close'].round(2)

# Sadece gerekli sütunları seçme: date, open, high, low, close,
data = data[['Date', 'Open', 'High', 'Low', 'Close']]

# CSV dosyasına kaydetme
data.to_csv("usd_try_daily_data.csv", index=False)

# Veriyi görüntüleme
print("Veri başarıyla CSV dosyasına kaydedildi!")
print(data.head())
