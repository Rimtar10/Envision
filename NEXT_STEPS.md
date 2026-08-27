# Envision — what changed and what to do next

Working session, August 2026. Everything below is already applied to the repo
unless marked **TODO**.

---

## 0. Do this first (2 minutes)

Git could not be driven safely from the review tooling, so these are yours:

```bash
# 1. Rotate the Roboflow key — it was hard-coded in two scripts on a public repo.
#    Roboflow dashboard -> Settings -> API keys -> regenerate.
#    Then, once per machine:
#      PowerShell:  setx ROBOFLOW_API_KEY "your-new-key"
#      bash:        export ROBOFLOW_API_KEY="your-new-key"

# 2. Stop tracking build output that is already in history
git rm -r --cached android/build .dart_tool build 2>$null

# 3. Commit. .gitignore is fixed, so `git add .` is now safe —
#    it will NOT pull in .venv, runs/, the datasets or the 110 MB of weights.
git add -A
git commit -m "Fix wake-word starving hazard announcements; per-class NMS; model geometry from tensors; honest metrics scripts"
git push
```

There is a `_to_delete/` folder holding stale `.git/*.lock` files. Delete it.

---

## 1. Safety bug — fixed

`VoiceService.announceDetections` opened with
`if (_stt.isListening || _inConversation) return;`. The wake-word poller holds
the mic ~83% of the time (4 s listen, 800 ms gap) **and was auto-enabled at
startup**, so obstacle warnings were dropped on roughly 5 of every 6 frames.

Now: a user command session still wins, ordinary objects still yield, but a
**hazard class takes the mic back and speaks**. Wake word is opt-in via the
existing `Wake: OFF` button.

Also: `_currentDetections` is now assigned before the early return, so
`describeScene()` and `countObjects()` no longer read a stale scene.

**TODO (real fix):** replace the STT polling loop with an on-device wake-word
engine (Porcupine / openWakeWord). Polling a cloud recogniser continuously is a
battery, privacy and cost problem, and it belongs in the report's ethics section.

---

## 2. Speed — what was done and what it buys

| Change | Effect |
|---|---|
| COCO model `yolov8n` → `yolo11n` | smaller (2.62 M vs 3.15 M params) **and** more accurate. Already exported. |
| Accessibility model → **float16** | 36 MB → 18 MB, ~2x CPU throughput, no measurable accuracy loss |
| `_accEveryNFrames` 6 → 2 | doors/stairs refresh 3x faster (paid for by float16) |
| Android **GPU delegate** | previously iOS-only; Android ran 4 CPU threads and nothing else. Falls back to CPU automatically if a kernel is missing. |
| Frame ceiling 80 ms → 50 ms | throughput was hard-capped at 12.5 FPS |
| Bundle trimmed | ~110 MB of assets → **29 MB** (only the two models actually loaded are packaged) |

**Not yet enabled — needs one command from you:**

`TensorflowHelper.fillInputBufferFused` folds the 90° rotation into the
letterbox sampling loop, removing a full-resolution allocation + copy on every
frame. It ships **disabled** because the rotation index maths must match the
`image` package's `copyRotate` convention exactly, and a silent mismatch would
rotate every bounding box in an app blind people rely on. To turn it on:

```bash
flutter test test/letterbox_parity_test.dart
```

If it passes, set `useFusedRotationLetterbox = true` in `tensorflow_helper.dart`.
If it fails, the failure message tells you which mapping your version wants
(swap the 90 and 270 cases).

**Biggest remaining lever: input size.** `ModelGeometry` now reads input size,
class count and anchor count from the model's own tensors, so re-exporting at
`imgsz=448` is a **drop-in change** — no Dart edits at all. 448² is about half
the pixels of 640², so expect close to a 2x inference win for a small mAP cost
on large objects like doors and stairs. Try it.

---

## 3. Accuracy — and why "lower the loss" is the wrong lever

`runs/detect/runs/accessibility_v2/results.csv`, the model you ship:

| epoch | train box | val box | mAP50 | mAP50-95 |
|---|---|---|---|---|
| 60 | 1.053 | 1.195 | 0.670 | 0.458 |
| 90 | 0.969 | 1.172 | 0.695 | 0.482 |
| 120 | 0.852 | 1.165 | 0.706 | **0.488** |

Epochs 90 → 120 bought **+0.006 mAP50-95** while the train/val box-loss gap
widened from 0.14 to 0.31. Training loss kept falling; validation loss went
flat at ~1.165 from epoch 90. That is convergence with mild overfitting. **More
epochs, lower LR, longer schedules will not help.** The model is data-limited.

What will actually move accuracy:

1. **Per-class NMS** (fixed). It was class-agnostic, so a person in a doorway,
   a person on a bicycle or a chair at a table lost one of the two boxes.
   Recovers real detections at zero cost.
2. **Hazard-exempt stability filter** (fixed). Every class had to persist two
   frames, so a car appearing suddenly was discarded on its first frame — extra
   latency on exactly the events that matter. Hazards now pass immediately.
3. **Distance correction** (fixed). Box heights are divided by `yStretch`
   before the pinhole estimate. Undoing letterbox padding stretches boxes on
   whichever axis was padded; when that is y, everything read *closer* than it
   was — silent, device-dependent, always in the dangerous direction.
4. **Drop the Window class.** It is the weakest (mAP50-95 ~0.31–0.50) and the
   least actionable — a blind pedestrian rarely needs to know a window exists.
   It is in the set because the Roboflow dataset happened to label it.
5. **Add curb / kerb.** Almost certainly worth more than Door and Window
   combined, and there is no good public dataset — 200–300 self-collected Beirut
   street images would be a genuine contribution.
6. **Separate ascending from descending stairs.** A descending staircase is the
   single most dangerous thing for a blind pedestrian. Labelling problem, not a
   modelling one.

**TODO — the retrain you chose:** `metrics/train_accessibility.py` now trains
**YOLO11n** (was 11s), batch 8, 150 epochs, run name `accessibility_v3`, and
exports **float16**. ~4–5 h on the 3050.

```bash
python metrics/train_accessibility.py
```

Then compare `runs/detect/runs/accessibility_v3/results.csv` against
`accessibility_v2/results.csv` before swapping. The script prints the exact
3-step handoff (pubspec → app_constants → `_accEveryNFrames = 1`).

Rationale: 11n loses maybe 2–4 mAP points but is cheap enough to run **every
frame**. For a safety app, a staircase reported 500 ms late is a fall; refresh
rate on Stair is worth more than the mAP.

---

## 4. Metrics — what was wrong, and what you can no longer quote

Three things in `metrics/output/` cannot go in a report:

**`accuracy_report.py` was inventing numbers.** It pointed at
`runs/detect/accessibility_detector/weights/best.pt`, which does not exist, so
every run fell into a hard-coded `EMBEDDED` fallback and wrote those constants
to `metrics_summary.txt` as if freshly measured. They described the **retired
5-class v1 model**. The fallback is deleted; the script now points at the
shipping model and **fails loudly** if it cannot validate.

**The COCO comparison evaluated on training data.** `coco128.yaml` is the first
128 images of COCO *train2017* — images both sets of weights were trained on.
mAP50 0.607 / 0.671 are train-set scores. Now `coco.yaml` (val2017, held out).

**The RT-DETR comparison was invalid three ways:** different datasets (YOLO rows
were coco128, RT-DETR was doors/stairs), different budgets (10 epochs at 512px
on 50% of data vs full COCO pretraining), and a speed column mixing GPU and CPU
timings — which made the transformer look 6x *faster* than the CNNs, the
opposite of the truth on a phone. `train_rtdetr.py` now **trains a matched CNN
baseline in the same run**, on the same yaml, same epochs, same imgsz, same
fraction, timed in the same process.

**Runtime numbers are stale.** `runtime_summary.txt` (89 ms, 10.2 FPS) was
captured in June, before the 38 MB float32 accessibility model landed. It is
also only 120 frames ≈ 12 seconds.

**TODO — regenerate all three:**
```bash
python metrics/accuracy_report.py       # fails loudly if paths are wrong
python metrics/compare_coco_models.py   # downloads val2017 once (~1 GB)
flutter run | Tee-Object perf_log.txt   # walk 3 min, indoor AND outdoor
python metrics/runtime_report.py perf_log.txt
```
Report device model, Android version, and **p95/p99** — for a safety app the
tail matters more than the mean.

---

## 5. Tests

New, and runnable with no device:

- `test/letterbox_parity_test.dart` — gates the fused fast path (see §2)
- `test/detection_logic_test.dart` — pins the pinhole distance maths, including
  the `yStretch` correction and resolution-independence at 448 vs 640

```bash
flutter test
```

---

## 6. Still open

- **`com.example.tensorflow_demo` cannot be published to Google Play.** Package
  rename touches every Dart import (`package:tensorflow_demo/...`), so it was
  left alone deliberately — do it in one focused commit.
- **README is 10 bytes.** Highest-leverage hour on the whole project.
- **The APK scanner should go.** `MainActivity.scanForApks()` walks all external
  storage and offers voice-triggered sideloading. Play policy problem, security
  anti-pattern, no user value.
- **No blind or low-vision testers.** Every timing and phrasing choice —
  3-second interval, "Person, 1.5 meters, to your left", speech rate 0.48 — is a
  guess. Five sessions with three users beats another training run.
- **VLM cognition layer.** Agreed: after the above. Wire it into the existing
  double-tap and "what do you see" command, keep the current label-list
  description as the offline fallback.
