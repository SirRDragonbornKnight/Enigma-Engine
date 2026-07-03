# Data Preparation

How to prepare training data for the FORGE.

---

## Supported Formats

- **Plain text (.txt)** — The model learns to predict the next token.
  Just raw text, one continuous stream.
- **JSONL** — Structured prompt/completion pairs for instruction tuning.

---

## Plain Text Tips

1. **Quality over quantity** — 10 KB of clean, focused text beats 1 MB of noise.
2. **Keep it consistent** — If training on conversations, keep the format uniform.
3. **Remove junk** — Strip HTML tags, URLs, repeated whitespace, encoding errors.
4. **Use real language** — The model learns what you feed it. No filler.

---

## Editing Data Files

Training data can be edited on the **DOCS page**:
- Open the DOCS page from the nav rail
- Select a data file from the file browser
- Edit and save directly in the built-in editor
- Create new files with the + NEW button

---

## File Location

Training data files live in the `data/` directory.
The FORGE scans this folder automatically.

Supported extensions: `.txt`

---

## Example Data

A simple training file (`data/training.txt`):

```
The quick brown fox jumps over the lazy dog.
She sells seashells by the seashore.
To be or not to be, that is the question.
```

For conversation training:

```
User: What is the capital of France?
Assistant: The capital of France is Paris.

User: How far is the moon?
Assistant: The moon is about 384,400 km from Earth.
```
