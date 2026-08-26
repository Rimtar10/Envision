This folder needs two files before the OCR feature will work — they're binary
model files too large to generate/download from this environment, so grab
them yourself (takes under a minute):

1. eng.traineddata
   https://github.com/tesseract-ocr/tessdata_fast/blob/main/eng.traineddata
   (click "Download raw file" / the download icon on that page)

2. ara.traineddata
   https://github.com/tesseract-ocr/tessdata_fast/blob/main/ara.traineddata
   (same — download raw file)

Drop both files directly into this folder (assets/tessdata/), so you end up
with:
  assets/tessdata/eng.traineddata
  assets/tessdata/ara.traineddata

Then delete this README (or leave it — it won't break anything, just extra
weight in the bundle) and run:
  flutter pub get
  flutter clean && flutter run

Using tessdata_fast (not tessdata_best) on purpose — smaller files, faster
inference, good enough accuracy for real-time-ish mobile OCR. If accuracy on
Arabic text turns out to be too low in testing, swap to tessdata_best's
ara.traineddata (larger, slower, more accurate) — same filename, same drop-in
location.
