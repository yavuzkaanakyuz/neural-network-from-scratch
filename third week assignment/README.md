# Third Week Assignment — Makemore

Bu klasördeki `makemore_bigram_trigram.ipynb` defteri haftanın altı görevini tek bir
akışta tamamlar:

1. Bigram sayımı: Python dictionary ve 27×27 PyTorch tensor
2. Satır normalizasyonu, broadcasting kontrolü ve isim örnekleme
3. NLL ve sahte sayım (smoothing)
4. One-hot girişli, 27×27 ağırlık matrisli tek katmanlı sinir ağı
5. Türkçe karakterlerle genişletilmiş alfabe ve iki modelin Türkçe veride eğitimi
6. Trigram, %80/%10/%10 bölme, dev loss ile smoothing seçimi ve test karşılaştırması

Defter çalıştırılmış ve çıktıları kaydedilmiştir. Yeniden çalıştırmak için PyTorch,
Matplotlib ve pandas bulunan bir Python ortamı yeterlidir. Çalışma dizini bu klasör
veya repo kökü olabilir.

## Veri kaynakları

- İngilizce isimler: [karpathy/makemore `names.txt`](https://github.com/karpathy/makemore/blob/master/names.txt)
- Türkçe isimler: [niyazikemer/turkce_isimler](https://github.com/niyazikemer/turkce_isimler)
- Broadcasting referansı: [PyTorch broadcasting semantics](https://pytorch.org/docs/stable/notes/broadcasting.html)

Türkçe CSV deftere gömülmez; defter çalışırken kaynak depodaki güncel CSV'yi okur.
Üretilen diziler model örneğidir ve gerçek kişi adı olmak zorunda değildir.
