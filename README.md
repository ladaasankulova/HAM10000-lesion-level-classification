[README.md](https://github.com/user-attachments/files/29994301/README.md)
# HAM10000-lesion-level-classification
# HAM10000 Üzerinde Cilt Lezyonu Sınıflandırması: Lezyon Düzeyinde Bölme ve Veri Sızıntısının Etkisi

Bu depo, **"Kanser Verilerinin Sınıflandırılmasında Makine Öğrenmesi Algoritmalarının Performans Karşılaştırması"** başlıklı yüksek lisans tezinin kaynak kodunu ve deneysel sonuçlarını içermektedir.

**Yazar:** Lada Asankulova
**Danışman:** Prof. Dr. Ali Konuralp
**Kurum:** Manisa Celal Bayar Üniversitesi, Lisansüstü Eğitim Enstitüsü, Matematik Anabilim Dalı
**Yıl:** 2026

---

## Özet

HAM10000 dermatoskopi veri seti üzerinde dört derin öğrenme mimarisi (DenseNet121, ConvNeXt-Tiny, ResNet50, EfficientNet-B3) karşılaştırılmıştır. Çalışmanın özgün katkısı, **veri sızıntısının (data leakage) sınıflandırma başarımı üzerindeki etkisini kontrollü bir deneyle ölçmesidir**: aynı lezyona ait farklı görüntülerin eğitim ve test kümelerine dağılması durumunda ortaya çıkan yapay performans artışı nicel olarak gösterilmiştir.

Üç deney gerçekleştirilmiştir:

1. **Çok sınıflı sınıflandırma** — 7 lezyon tipi (akiec, bcc, bkl, df, mel, nv, vasc)
2. **İkili sınıflandırma** — iyi huylu (benign) / kötü huylu (malignant)
3. **Veri sızıntısı deneyi** — sızıntılı protokol ile lezyon düzeyinde bölmenin karşılaştırılması

---

## Metodolojik İlkeler

- **Lezyon düzeyinde bölme (lesion-level split):** Aynı `lesion_id` değerine sahip görüntüler yalnızca tek bir kümede (eğitim / doğrulama / test) yer alır. HAM10000'de aynı lezyonun birden fazla görüntüsü bulunduğundan, görüntü düzeyinde rastgele bölme veri sızıntısına yol açmaktadır.
- **Çoklu tohum protokolü:** Tüm deneyler üç farklı rastgele tohumla (**13, 21, 42**) bağımsız olarak tekrarlanmış; sonuçlar **ortalama ± standart sapma** olarak raporlanmıştır.
- **Sabit bölme:** Veri bölmesi tüm koşularda sabit tutulmuştur (`seed = 42`). Bu nedenle raporlanan standart sapma **eğitim kaynaklı varyansı** yansıtır; bölme kaynaklı varyans ölçülmemiştir.
- **Sınıf dengeleme:** Alt örnekleme (undersampling) + `WeightedRandomSampler`
- **Veri artırma:** Albumentations (renk bilgisini koruyan, medikal görüntülemeye uygun dönüşümler)
- **Optimizasyon:** Nesterov momentumlu SGD + OneCycleLR (süper yakınsama)

---

## Klasör Yapısı

```
├── notebooks/
│   ├── 01_data_analysis.ipynb            # Keşifsel veri analizi
│   ├── 02_data_preparation.ipynb         # Lezyon düzeyinde bölme + sınıf dengeleme
│   ├── 03_binary_training_42.ipynb          # İkili sınıflandırma (4 model × 3 tohum)
│   ├── 04_multiclass_training_42.ipynb      # Çok sınıflı sınıflandırma (4 model × 3 tohum)
│   └── 06_data_leakage_experiment_42.ipynb  # Veri sızıntısı deneyi (4 model × 3 tohum)
├── src/
│   ├── dataset.py                        # HAMDataset, Albumentations dönüşümleri, sampler
│   ├── train.py                          # Eğitim döngüsü (OneCycleLR)
│   └── models/
│       └── pretrained_models.py          # Model fabrikası (transfer öğrenmesi)
├── results/
│   ├── multi-summary/               # Çok sınıflı: ort±std tablo + hata çubuklu grafikler
│   ├── binary-summary/                   # İkili: ort±std tablo + hata çubuklu grafikler
│                       
├── requirements.txt
└── README.md
```

**Not:** `data/` (HAM10000 görüntüleri) ve `models/` (eğitilmiş `.pth` ağırlıkları) dosya boyutu nedeniyle bu depoya dahil edilmemiştir.

---

**Veri seti:** HAM10000, Harvard Dataverse üzerinden indirilebilir:
https://doi.org/10.7910/DVN/DBW86T

İndirilen görüntüler ve `HAM10000_metadata.csv` dosyası `data/` klasörüne yerleştirilmelidir.

**Önemli:** Notebook'lardaki `BASE` değişkeni yerel dosya yolunu (`F:/CancerDataClassification`) göstermektedir. Kendi sisteminizde çalıştırmadan önce bu yolu güncelleyiniz.

---

## Çalıştırma Sırası

1. `01_data_analysis.ipynb` — veri setinin incelenmesi
2. `02_data_preparation.ipynb` — bölme ve dengeleme (**bir kez** çalıştırılır, `seed = 42` sabit)
3. Aşağıdaki üç notebook'tan herhangi biri:
   - `03_binary_training_42.ipynb`
   - `04_multiclass_training_42.ipynb`
   - `06_data_leakage_experiment_42.ipynb`

**Çoklu tohum koşusu:** Her eğitim notebook'unun ilk hücresinde `SEED` değişkeni bulunmaktadır. Notebook, `SEED = 13`, `SEED = 21` ve `SEED = 42` değerleriyle üç kez baştan sona çalıştırılır. Her koşu, sonuçlarını tohuma özgü bir klasöre kaydeder. Üç koşu tamamlandıktan sonra, notebook'un sonundaki toplama hücresi ortalama ± standart sapma tablosunu ve hata çubuklu grafikleri üretir.

---

## Atıf

```
L. Asankulova, "Kanser Verilerinin Sınıflandırılmasında Makine Öğrenmesi
Algoritmalarının Performans Karşılaştırması", Yüksek Lisans Tezi,
Manisa Celal Bayar Üniversitesi, Lisansüstü Eğitim Enstitüsü, 2026.
```

## Veri Seti Atfı

```
P. Tschandl, C. Rosendahl, H. Kittler, "The HAM10000 dataset, a large collection
of multi-source dermatoscopic images of common pigmented skin lesions",
Scientific Data, 5, 180161, 2018. doi: 10.1038/sdata.2018.161
```
