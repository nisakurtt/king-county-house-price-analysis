import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split

df = pd.read_csv("kc_house_data.csv")
print(df.head())

print("Boyutları:", df.shape)
print("Sütunlar:", df.columns) 
df.info() 
#Sütun isimleri, veri tipleri,null sayısı.

"""
Veri seti 21613 gözlem ve 21 değişkenden oluşmaktadır. 
Eksik veri bulunmamaktadır. 
Hedef değişken olan "price" sürekli bir değişkendir ve problem regresyon problemidir. 
Veri setinde çoğunlukla sayısal değişkenler bulunmakta olup, "zipcode" değişkeni kategorik olarak ele alınmalıdır.
"id" değişkeni modelleme açısından anlamlı olmadığı için çıkarılmalıdır. 
"date" değişkeni ise uygun formata dönüştürülmeli veya analiz dışında bırakılmalıdır.
"""

print(df["price"].describe())
#temel istatiksel bilgileri verir.
plt.hist(df["price"], bins=50)
plt.title("House Price Distribution")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

#aykırı değerler 
plt.figure(figsize=(8, 4))
plt.boxplot(df["price"], vert=False)
plt.title("Boxplot of House Prices")
plt.xlabel("Price")
plt.show()

#yüksek fiyatlı evler, dağılımından sapmaktadır ve modeli olumsuz etkileyebilir.

# Q1 ve Q3 Hesaplama
Q1 = df["price"].quantile(0.25)
Q3 = df["price"].quantile(0.75) 

# IQR Hesaplama
IQR = Q3 - Q1

#Alt ve Üst Sınır
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print("Alt Sınır:", lower_bound)
print("Üst Sınır:", upper_bound)

#Filtreleme
df_clean = df[(df["price"] >= lower_bound) & (df["price"] <= upper_bound)]
print("Eski veri boyutu:", df.shape)
print("Temizlenmiş veri boyutu:", df_clean.shape)

#Yani IQR yöntemiyle aykırı değerler temizlenmiştir ve yeni veri seti df_clean olarak kaydedilmiştir.

plt.hist(df_clean["price"], bins=50)
plt.title("Cleaned Price Distribution")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

#gereksiz sütunları çıkarma
df_model = df_clean.drop(columns=["id", "date"], axis=1)

#x ve y değişkenlerini ayırma(x = özellikler, y = hedef)
X = df_model.drop("price", axis=1)
y = df_model["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)
#random_state parametresi, train_test_split fonksiyonunun her çalıştırıldığında aynı rastgele örnekleme sonucunu üretmesini sağlar.

#KNN MODELİ
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
knn = KNeighborsRegressor(n_neighbors=5)
#n_neighbors parametresi, modelin tahmin yaparken kaç komşuyu dikkate alacağını belirler.
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
#performans değerlendirmesi
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)

#Random Forest Modeli
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, random_state=41)
#n_estimators parametresi, modelin oluşturacağı ağaç sayısını belirler
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
print("Random Forest Mean Absolute Error:", mae_rf)


#KONUM ANALİZİ
plt.figure(figsize=(8,6))

plt.scatter(
    df_clean["long"],
    df_clean["lat"],
    c=df_clean["price"],
    cmap="viridis",
    alpha=0.5
)

plt.colorbar(label="Price")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Geographical Distribution of House Prices")

plt.show()

#FEATURE IMPORTANCE - FİYATI EN ÇOK ETKİLEYEN DEĞİŞKENLER
importances = rf.feature_importances_
features = X.columns
feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)
print("Feature Importances:\n", feat_imp.head(10))

#Modeli en çok etkileyen özellik konummuş.
