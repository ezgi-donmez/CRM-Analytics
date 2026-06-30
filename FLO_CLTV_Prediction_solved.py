##############################################################
# BG-NBD ve Gamma-Gamma ile FLO CLTV Prediction
##############################################################

# GÖREVLER
# 1. Veriyi hazırlama
# 2. CLTV veri yapısını oluşturma
# 3. BG/NBD ve Gamma-Gamma modelleri ile CLTV hesaplama
# 4. CLTV'ye göre segment oluşturma
# BONUS: Süreci fonksiyonlaştırma

# Gerekli kütüphaneler

import pandas as pd
import datetime as dt
import warnings
from lifetimes import BetaGeoFitter, GammaGammaFitter

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 300)
# pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: "%.3f" % x)

###############################################################
# GÖREV 1: Veriyi Hazırlama
###############################################################

# 1. flo_data_20K.csv verisini okuyunuz. Dataframe'in kopyasını oluşturunuz.

df1 = pd.read_csv(r"C:\Users\ed024981\Desktop\Miuul\week3\FLOCLTVPrediction\flo_data_20k.csv")
df = df1.copy()
df.head()
df.shape
df.isnull().sum()

# İlk kontroller
print("Veri boyutu:", df.shape)
print(df.head())
print(df.info())
print(df.describe().T)
print(df.isnull().sum())


# 2. Aykırı değerleri baskılamak için gerekli fonksiyonları tanımlayınız.
# Not: CLTV hesaplanırken frequency değerleri integer olmalıdır. Bu nedenle limitler round() ile yuvarlandı.

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return round(low_limit), round(up_limit)


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[dataframe[variable] < low_limit, variable] = low_limit
    dataframe.loc[dataframe[variable] > up_limit, variable] = up_limit


# 3. "order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online" değişkenlerinin
# aykırı değerleri varsa baskılayanız.
outlier_cols = [
    "order_num_total_ever_online",
    "order_num_total_ever_offline",
    "customer_value_total_ever_offline",
    "customer_value_total_ever_online",
]

for col in outlier_cols:
    replace_with_thresholds(df, col)

# Frequency kolonlarının integer olması için tip dönüşümü
order_cols = ["order_num_total_ever_online", "order_num_total_ever_offline"]
df[order_cols] = df[order_cols].astype(int)


# 4. Toplam alışveriş sayısı ve toplam harcama değişkenlerini oluşturunuz.
df["total_order_num"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_customer_value"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]


# 5. Tarih ifade eden değişkenleri datetime tipine çeviriniz.
date_cols = [col for col in df.columns if "date" in col]
for col in date_cols:
    df[col] = pd.to_datetime(df[col])

print(df.dtypes)

###############################################################
# GÖREV 2: CLTV Veri Yapısının Oluşturulması
###############################################################

# 1. En son alışveriş tarihinden 2 gün sonrasını analiz tarihi olarak alınız.
analysis_date = df["last_order_date"].max() + dt.timedelta(days=2)
print("Analiz tarihi:", analysis_date)


# 2. customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerini oluşturunuz.
cltv = pd.DataFrame()
cltv["customer_id"] = df["master_id"]
cltv["recency_cltv_weekly"] = (df["last_order_date"] - df["first_order_date"]).dt.days / 7
cltv["T_weekly"] = (analysis_date - df["first_order_date"]).dt.days / 7  # T: müşteri yaşı 
cltv["frequency"] = df["total_order_num"].astype(int)  
cltv["monetary_cltv_avg"] = df["total_customer_value"] / df["total_order_num"]

# BG/NBD ve Gamma-Gamma için frequency > 1 olmalıdır.
cltv = cltv[(cltv["frequency"] > 1) & (cltv["monetary_cltv_avg"] > 0)]
cltv.reset_index(drop=True, inplace=True)

print(cltv.head())
print(cltv.describe().T)

###############################################################
# GÖREV 3: BG/NBD, Gamma-Gamma Modellerinin Kurulması ve CLTV'nin Hesaplanması
###############################################################

# 1. fit BG/NBD modelin
bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(
    cltv["frequency"],
    cltv["recency_cltv_weekly"],
    cltv["T_weekly"],
)

# 3 ay içerisinde beklenen satın alma sayısı
cltv["exp_sales_3_month"] = bgf.predict(
    4 * 3,
    cltv["frequency"],
    cltv["recency_cltv_weekly"],
    cltv["T_weekly"],
)

# 6 ay içerisinde beklenen satın alma sayısı
cltv["exp_sales_6_month"] = bgf.predict(
    4 * 6,
    cltv["frequency"],
    cltv["recency_cltv_weekly"],
    cltv["T_weekly"],
)

# 3. ve 6. ayda en çok satın alım gerçekleştirecek 10 kişi
print("\n3 ayda en çok satın alma beklenen 10 müşteri:")
print(cltv.sort_values("exp_sales_3_month", ascending=False).head(10))

print("\n6 ayda en çok satın alma beklenen 10 müşteri:")
print(cltv.sort_values("exp_sales_6_month", ascending=False).head(10))


# 2. Gamma-Gamma modelini fit ediniz.
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv["frequency"], cltv["monetary_cltv_avg"])

# Müşterilerin beklenen ortalama değerini tahminleyiniz.
cltv["exp_average_value"] = ggf.conditional_expected_average_profit(
    cltv["frequency"],
    cltv["monetary_cltv_avg"],
)

print("\nBeklenen ortalama değeri en yüksek 10 müşteri:")
print(cltv.sort_values("exp_average_value", ascending=False).head(10))


# 3. 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.
cltv["cltv"] = ggf.customer_lifetime_value(
    bgf,
    cltv["frequency"],
    cltv["recency_cltv_weekly"],
    cltv["T_weekly"],
    cltv["monetary_cltv_avg"],
    time=6,          # 6 aylık CLTV
    freq="W",        # recency ve T haftalık olduğu için W
    discount_rate=0.01,
)

# CLTV değeri en yüksek 20 kişi
print("\nCLTV değeri en yüksek 20 müşteri:")
print(cltv.sort_values("cltv", ascending=False).head(20))


###############################################################
# GÖREV 4: CLTV'ye Göre Segmentlerin Oluşturulması
###############################################################

# 1. 6 aylık CLTV'ye göre tüm müşterileri 4 segmente ayırınız.
cltv["cltv_segment"] = pd.qcut(cltv["cltv"], 4, labels=["D", "C", "B", "A"])

print("\nSegmentlere göre ilk gözlemler:")
print(cltv.sort_values("cltv", ascending=False).head(20))


# 2. Segmentlerin recency, frequency ve monetary ortalamalarını inceleyiniz.
segment_summary = cltv.groupby("cltv_segment", observed=True).agg({
    "customer_id": "count",
    "recency_cltv_weekly": "mean",
    "T_weekly": "mean",
    "frequency": "mean",
    "monetary_cltv_avg": "mean",
    "exp_sales_3_month": "mean",
    "exp_sales_6_month": "mean",
    "exp_average_value": "mean",
    "cltv": "mean",
}).sort_values("cltv", ascending=False)

print("\nSegment özeti:")
print(segment_summary)


# Yönetim için 6 aylık aksiyon önerileri:
# A segmenti: CLTV değeri en yüksek müşteri grubu. Bu müşteriler için VIP sadakat programı,
# kişiselleştirilmiş kampanyalar, yeni koleksiyonlara erken erişim ve premium ürün önerileri yapılabilir.
# Amaç, yüksek değerli müşterilerin elde tutulması ve sepet tutarının artırılmasıdır.
#
# D segmenti: CLTV değeri en düşük müşteri grubu. Bu gruba yüksek bütçeli kampanyalar yerine
# düşük maliyetli yeniden aktivasyon kampanyaları, sınırlı süreli indirim kuponları ve kanal bazlı
# hatırlatma iletişimleri uygulanabilir. Amaç, pasif/düşük değerli müşterileri kontrollü maliyetle canlandırmaktır.


# Çıktıları kaydetme
cltv.to_csv("flo_cltv_prediction.csv", index=False)
segment_summary.to_csv("flo_cltv_segment_summary.csv")

###############################################################
# Tüm Süreci Fonksiyonlaştırma
###############################################################

def create_cltv_prediction(dataframe, month=6):
    """
    FLO veri seti için BG/NBD + Gamma-Gamma modelleriyle CLTV hesaplar.
    """
    dataframe = dataframe.copy()

    # Aykırı değerleri baskılama
    for col in outlier_cols:
        replace_with_thresholds(dataframe, col)

    dataframe[order_cols] = dataframe[order_cols].astype(int)

    # Toplam alışveriş ve toplam değer
    dataframe["total_order_num"] = (
        dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    )
    dataframe["total_customer_value"] = (
        dataframe["customer_value_total_ever_online"] + dataframe["customer_value_total_ever_offline"]
    )

    # Tarih dönüşümü
    date_cols_func = [col for col in dataframe.columns if "date" in col]
    for col in date_cols_func:
        dataframe[col] = pd.to_datetime(dataframe[col])

    analysis_date_func = dataframe["last_order_date"].max() + dt.timedelta(days=2)

    # CLTV veri yapısı
    cltv_final = pd.DataFrame()
    cltv_final["customer_id"] = dataframe["master_id"]
    cltv_final["recency_cltv_weekly"] = (
        dataframe["last_order_date"] - dataframe["first_order_date"]
    ).dt.days / 7
    cltv_final["T_weekly"] = (analysis_date_func - dataframe["first_order_date"]).dt.days / 7
    cltv_final["frequency"] = dataframe["total_order_num"].astype(int)
    cltv_final["monetary_cltv_avg"] = (
        dataframe["total_customer_value"] / dataframe["total_order_num"]
    )

    cltv_final = cltv_final[
        (cltv_final["frequency"] > 1) & (cltv_final["monetary_cltv_avg"] > 0)
    ]
    cltv_final.reset_index(drop=True, inplace=True)

    # BG/NBD
    bgf_model = BetaGeoFitter(penalizer_coef=0.001)
    bgf_model.fit(
        cltv_final["frequency"],
        cltv_final["recency_cltv_weekly"],
        cltv_final["T_weekly"],
    )

    cltv_final["exp_sales_3_month"] = bgf_model.predict(
        4 * 3,
        cltv_final["frequency"],
        cltv_final["recency_cltv_weekly"],
        cltv_final["T_weekly"],
    )
    cltv_final["exp_sales_6_month"] = bgf_model.predict(
        4 * 6,
        cltv_final["frequency"],
        cltv_final["recency_cltv_weekly"],
        cltv_final["T_weekly"],
    )

    # Gamma-Gamma
    ggf_model = GammaGammaFitter(penalizer_coef=0.01)
    ggf_model.fit(cltv_final["frequency"], cltv_final["monetary_cltv_avg"])

    cltv_final["exp_average_value"] = ggf_model.conditional_expected_average_profit(
        cltv_final["frequency"],
        cltv_final["monetary_cltv_avg"],
    )

    # CLTV
    cltv_final["cltv"] = ggf_model.customer_lifetime_value(
        bgf_model,
        cltv_final["frequency"],
        cltv_final["recency_cltv_weekly"],
        cltv_final["T_weekly"],
        cltv_final["monetary_cltv_avg"],
        time=month,
        freq="W",
        discount_rate=0.01,
    )

    # Segment
    cltv_final["cltv_segment"] = pd.qcut(cltv_final["cltv"], 4, labels=["D", "C", "B", "A"])

    segment_summary_final = cltv_final.groupby("cltv_segment", observed=True).agg({
        "customer_id": "count",
        "recency_cltv_weekly": "mean",
        "T_weekly": "mean",
        "frequency": "mean",
        "monetary_cltv_avg": "mean",
        "exp_sales_3_month": "mean",
        "exp_sales_6_month": "mean",
        "exp_average_value": "mean",
        "cltv": "mean",
    }).sort_values("cltv", ascending=False)

    return cltv_final, segment_summary_final


# Fonksiyonun çalıştırılması
cltv_final_func, segment_summary_func = create_cltv_prediction(df1, month=6)
print("\nFonksiyon çıktısı - ilk 5 gözlem:")
print(cltv_final_func.head())
print("\nFonksiyon çıktısı - segment özeti:")
print(segment_summary_func)
