###############################################################
# RFM ile Müşteri Segmentasyonu 
###############################################################

import os
import datetime as dt
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.float_format", lambda x: "%.3f" % x)

###############################################################
# GÖREV 1: Veriyi Anlama ve Hazırlama
###############################################################

df1 = pd.read_csv(r"C:\Users\ed024981\Desktop\Miuul\week3\FLOCLTVPrediction\flo_data_20k.csv")
df = df1.copy()

# 2. Veri setini inceleme
print("İlk 10 gözlem:")
print(df.head(10))

print("\nDeğişken isimleri:")
print(df.columns)

print("\nBoyut:")
print(df.shape)

print("\nBetimsel istatistik:")
print(df.describe().T)

print("\nBoş değer sayısı:")
print(df.isnull().sum())

print("\nDeğişken tipleri:")
print(df.dtypes)

# 3. Toplam alışveriş sayısı ve toplam harcama değişkenleri
# Omnichannel: müşteri hem online hem offline alışveriş yapabilir.
df["total_order_num"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_customer_value"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]

# 4. Tarih değişkenlerini datetime tipine çevirme
date_cols = [col for col in df.columns if "date" in col]
df[date_cols] = df[date_cols].apply(pd.to_datetime)

print("\nTarih dönüşümü sonrası değişken tipleri:")
print(df.dtypes)

# 5. Alışveriş kanallarındaki müşteri sayısı, toplam ürün/sipariş ve toplam harcama dağılımı
channel_summary = df.groupby("order_channel").agg({
    "master_id": "count",
    "total_order_num": "sum",
    "total_customer_value": "sum"
})
print("\nKanal bazında müşteri sayısı, toplam sipariş ve toplam harcama:")
print(channel_summary)

# 6. En fazla kazancı getiren ilk 10 müşteri
top_10_customers_by_value = df.sort_values("total_customer_value", ascending=False).head(10)
print("\nEn fazla kazanç getiren ilk 10 müşteri:")
print(top_10_customers_by_value[["master_id", "total_customer_value"]])

# 7. En fazla sipariş veren ilk 10 müşteri
top_10_customers_by_order = df.sort_values("total_order_num", ascending=False).head(10)
print("\nEn fazla sipariş veren ilk 10 müşteri:")
print(top_10_customers_by_order[["master_id", "total_order_num"]])

# 8. Veri ön hazırlık function

def flo_data_prep(dataframe):
    """FLO veri seti için toplam sipariş/harcama değişkenlerini oluşturur ve tarihleri dönüştürür."""
    dataframe = dataframe.copy()

    dataframe["total_order_num"] = (
        dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    )
    dataframe["total_customer_value"] = (
        dataframe["customer_value_total_ever_online"] + dataframe["customer_value_total_ever_offline"]
    )
    date_cols = [col for col in dataframe.columns if "date" in col]
    dataframe[date_cols] = dataframe[date_cols].apply(pd.to_datetime)

    return dataframe

###############################################################
# GÖREV 2: RFM Metriklerinin Hesaplanması
###############################################################

df = flo_data_prep(df1)
df.head()

# Veri setindeki en son alışveriş tarihinden 2 gün sonrası analiz tarihi olarak seçilir.
analysis_date = df["last_order_date"].max() + dt.timedelta(days=2)
print("\nAnaliz tarihi:", analysis_date)

# customer_id, recency, frequency, monetary değerlerinin yer aldığı rfm dataframe'i
rfm = df.groupby("master_id").agg({
    "last_order_date": lambda date: (analysis_date - date.max()).days,
    "total_order_num": "sum",
    "total_customer_value": "sum"
})

rfm.columns = ["recency", "frequency", "monetary"]
rfm = rfm[rfm["monetary"] > 0]

print("\nRFM metrikleri:")
print(rfm.head())

###############################################################
# GÖREV 3: RF ve RFM Skorlarının Hesaplanması
###############################################################

# Recency düşükse müşteri daha günceldir, bu yüzden düşük recency yüksek skor alır.
rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1])

# Frequency ve monetary yüksekse daha iyi olduğu için yüksek değerlere yüksek skor verilir.
# Frequency'de tekrar eden değerler fazla olabileceği için rank(method="first") kullanılır.
rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5])

# Segmentleme RF_SCORE üzerinden yapılır.
rfm["RF_SCORE"] = rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str)

print("\nSkorlu RFM:")
print(rfm.head())

###############################################################
# GÖREV 4: RF Skorlarının Segment Olarak Tanımlanması
###############################################################

seg_map = {
    r"[1-2][1-2]": "hibernating",
    r"[1-2][3-4]": "at_risk",
    r"[1-2]5": "cant_loose",
    r"3[1-2]": "about_to_sleep",
    r"33": "need_attention",
    r"[3-4][4-5]": "loyal_customers",
    r"41": "promising",
    r"51": "new_customers",
    r"[4-5][2-3]": "potential_loyalists",
    r"5[4-5]": "champions"
}

rfm["segment"] = rfm["RF_SCORE"].replace(seg_map, regex=True)

print("\nSegmentli RFM:")
print(rfm.head())

###############################################################
# GÖREV 5: Aksiyon Zamanı
###############################################################

# 1. Segmentlerin recency, frequency ve monetary ortalamaları
segment_summary = rfm.groupby("segment").agg({
    "recency": ["mean", "count"],
    "frequency": ["mean", "count"],
    "monetary": ["mean", "count"]
})
print("\nSegment özeti:")
print(segment_summary)

# RFM tablosuna kategori bilgilerini ekle
rfm = rfm.reset_index()
rfm = rfm.merge(df[["master_id", "interested_in_categories_12"]], on="master_id", how="left")

# 2.a Yeni kadın ayakkabı markası hedef kitlesi:
# Segment: champions veya loyal_customers
# Kategori: KADIN
# monetary > 250 TL
new_brand_target_customers = rfm[
    (rfm["segment"].isin(["champions", "loyal_customers"]))
    & (rfm["monetary"] > 250)
    & (rfm["interested_in_categories_12"].str.contains("KADIN", na=False))
][["master_id"]]

new_brand_target_customers.to_csv("yeni_marka_hedef_musteri_id.csv", index=False)
print("\nYeni marka hedef müşteri sayısı:", new_brand_target_customers.shape[0])

# 2.b Erkek ve çocuk ürünleri indirimi hedef kitlesi:
# Segment: cant_loose, hibernating, new_customers
# Kategori: ERKEK or COCUK

discount_target_customers = rfm[
    (rfm["segment"].isin(["cant_loose", "hibernating", "new_customers"]))
    & (rfm["interested_in_categories_12"].str.contains("ERKEK|COCUK", regex=True, na=False))
][["master_id"]]

discount_target_customers.to_csv("indirim_hedef_musteri_ids.csv", index=False)
print("İndirim hedef müşteri sayısı:", discount_target_customers.shape[0])

# Full RFM çıktısı
rfm.to_csv("rfm.csv", index=False)
segment_summary.to_csv("segment_summary.csv")

###############################################################
# GÖREV 6: Tüm Süreci Fonksiyonlaştırma
###############################################################

def create_rfm(dataframe, csv=False):
    """FLO veri setinden RFM tablosu ve segmentleri üretir."""
    dataframe = flo_data_prep(dataframe)

    analysis_date = dataframe["last_order_date"].max() + dt.timedelta(days=2)

    rfm = dataframe.groupby("master_id").agg({
        "last_order_date": lambda date: (analysis_date - date.max()).days,
        "total_order_num": "sum",
        "total_customer_value": "sum"
    })

    rfm.columns = ["recency", "frequency", "monetary"]
    rfm = rfm[rfm["monetary"] > 0]

    rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1])
    rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5])
    rfm["RF_SCORE"] = rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str)

    seg_map = {
        r"[1-2][1-2]": "hibernating",
        r"[1-2][3-4]": "at_risk",
        r"[1-2]5": "cant_loose",
        r"3[1-2]": "about_to_sleep",
        r"33": "need_attention",
        r"[3-4][4-5]": "loyal_customers",
        r"41": "promising",
        r"51": "new_customers",
        r"[4-5][2-3]": "potential_loyalists",
        r"5[4-5]": "champions"
    }

    rfm["segment"] = rfm["RF_SCORE"].replace(seg_map, regex=True)
    rfm = rfm.reset_index()
    rfm = rfm.merge(dataframe[["master_id", "interested_in_categories_12"]], on="master_id", how="left")

    if csv:
        rfm.to_csv("rfm.csv", index=False)

    return rfm

# Fonksiyon kullanımı
rfm_final = create_rfm(df1, csv=True)
