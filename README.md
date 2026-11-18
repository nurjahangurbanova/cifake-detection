# CIFAKE: Yapay Zekâ ile Üretilmiş Görüntülerin Tespiti

Kocaeli Üniversitesi, Yapay Sinir Ağları dersi proje çalışması.

Bir görüntünün gerçek fotoğraf mı yoksa yapay zekâ (Stable Diffusion) üretimi mi olduğunu evrişimsel sinir ağları (CNN) ve transfer öğrenme ile tespit ediyoruz.

**Referans makale:** Bird, J. J., & Lotfi, A. (2024). CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images. *IEEE Access*, 12, 15642–15650. https://doi.org/10.1109/ACCESS.2024.3356122

## Ekip
- Nurjahan Gurbanova
- *(grup arkadaşı)*

## Kurulum

```bash
git clone https://github.com/nurjahangurbanova/cifake-detection.git
cd cifake-detection
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Veri seti

Veri seti boyutu nedeniyle repoda yoktur. Kaggle'dan **"CIFAKE: Real and AI-Generated Synthetic Images"** (birdy654) veri setini indirip `data/` klasörüne çıkarın:

```
data/train/REAL   data/train/FAKE
data/test/REAL    data/test/FAKE
```

## Çalıştırma

```bash
python src/01_veri_kontrol.py
```

## Yol haritası
- [x] Veri kontrolü
- [ ] Makaledeki CNN'in yeniden uygulanması
- [ ] Transfer öğrenme (ResNet-50, EfficientNet-B0)
- [ ] Sonuç karşılaştırma tablosu
- [ ] Grad-CAM ile açıklanabilirlik
- [ ] Midjourney / DALL-E görselleriyle genelleme testi
- [ ] Demo
- [ ] IEEE raporhead -5 README.md