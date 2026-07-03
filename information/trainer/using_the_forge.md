# Using The FORGE

Step-by-step guide to the FORGE page in the desktop GUI.

---

## Layout

The FORGE page has two resizable columns (drag the sash to resize):
1. **Controls** (left) — Training settings and buttons
2. **Output Log** (right) — Real-time training progress

---

## Training a Model

1. **Select data** — Pick a file from the data source dropdown.
   The file content loads into the Data Editor.
2. **Set model size** — Type your target in the size field (e.g. `8b`, `500m`, `small`).
3. **Set parameters** — Epochs, batch size, learning rate.
4. **Click START TRAINING** — Training runs in a background thread.
5. **Monitor the log** — Watch epoch loss decrease.
6. **Models save to** `models/` when training completes.

---

## Training a Tokenizer

1. Select your training data from the dropdown.
2. Set the vocabulary size (default: 8000).
3. Click TRAIN TOKENIZER.
4. The tokenizer saves to `models/tokenizer.json`.

---

## Editing Training Data

Training data files can be edited on the **DOCS page**:
- Open the DOCS page from the nav rail
- Select a data file from the file browser
- Edit and save directly in the built-in editor

---

## Creating Models

Use the MODELS page (not the FORGE) to create empty model files:
1. Enter a name.
2. Enter the size (e.g. `1.5b`).
3. Click CREATE.

Then train it on the FORGE page.

---

## Stopping Training

Click STOP to halt training after the current epoch completes.
The model is NOT saved when stopped early — only completed training saves.
