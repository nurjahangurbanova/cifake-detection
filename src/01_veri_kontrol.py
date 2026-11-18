"""
01 - Veri kontrolü
- Veri seti doğru yerde mi?
- Her klasörde kaç görüntü var?
- Görüntü boyutu ne?
- Örnek gerçek / sahte görüntüleri gösterir ve kaydeder.
- Bilgisayarda GPU (Mac'te MPS, Windows'ta CUDA) var mı kontrol eder.

Çalıştırma (proje ana klasöründeyken):
    python src/01_veri_kontrol.py
"""

from pathlib import Path
import random

import matplotlib.pyplot as plt
from PIL import Image
import torch

# Yollar script'in yerine göre ayarlanıyor, os.chdir derdi yok
PROJE = Path(__file__).resolve().parent.parent
VERI = PROJE / "data"
CIKTI = PROJE / "outputs"
CIKTI.mkdir(exist_ok=True)

SINIFLAR = ["REAL", "FAKE"]


def klasor_say():
    print("=== Görüntü sayıları ===")
    for bolum in ["train", "test"]:
        for sinif in SINIFLAR:
            klasor = VERI / bolum / sinif
            if not klasor.exists():
                print(f"[HATA] Bulunamadı: {klasor}")
                print("Veri setini data/ klasörüne doğru çıkardığından emin ol.")
                raise SystemExit(1)
            sayi = len(list(klasor.glob("*.jpg"))) + len(list(klasor.glob("*.png")))
            print(f"{bolum:5s} / {sinif:4s}: {sayi:6d} görüntü")


def boyut_kontrol():
    ornek = next((VERI / "train" / "REAL").iterdir())
    img = Image.open(ornek)
    print(f"\n=== Örnek görüntü ===\nBoyut: {img.size}, Mod: {img.mode}")


def ornek_goster(adet=8):
    fig, axes = plt.subplots(2, adet, figsize=(adet * 1.6, 3.8))
    for satir, sinif in enumerate(SINIFLAR):
        dosyalar = list((VERI / "train" / sinif).iterdir())
        for i, dosya in enumerate(random.sample(dosyalar, adet)):
            ax = axes[satir, i]
            ax.imshow(Image.open(dosya))
            ax.axis("off")
            if i == 0:
                ax.set_title(sinif, loc="left", fontsize=11, fontweight="bold")
    plt.suptitle("CIFAKE: Üst satır gerçek, alt satır AI üretimi")
    plt.tight_layout()
    yol = CIKTI / "ornek_goruntuler.png"
    plt.savefig(yol, dpi=150)
    print(f"\nÖrnek görüntüler kaydedildi: {yol}")
    plt.show()


def cihaz_kontrol():
    print("\n=== Donanım ===")
    if torch.cuda.is_available():
        print("CUDA GPU var:", torch.cuda.get_device_name(0))
    elif torch.backends.mps.is_available():
        print("Apple GPU (MPS) var, eğitim hızlı olur.")
    else:
        print("GPU yok, CPU kullanılacak. Büyük modeller için Colab kullanırız.")


if __name__ == "__main__":
    random.seed(42)
    klasor_say()
    boyut_kontrol()
    cihaz_kontrol()
    ornek_goster()
    