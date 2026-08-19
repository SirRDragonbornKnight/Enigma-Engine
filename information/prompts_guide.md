# Prompts Guide

Prompts shape how the AI behaves. A good system prompt makes the
difference between useful and useless responses.

---

## The System Prompt

Enigma is served over the OpenAI API (`python serve_enigma.py`), so the
system prompt is simply the `system` message of your request:

```python
client.chat.completions.create(
    model="enigma",
    messages=[
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Fix this bug..."},
    ],
)
```

The server composes the final system context from your system message
plus whatever it needs to inject: available tool specs (built-ins are
intent-gated per request) and, with `--memory-dir`, memories relevant
to the user's message. Your prompt sits alongside those -- you do not
need to describe her tools yourself.

### Tips for Good Prompts

1. **Be specific** -- tell the AI exactly what role to play
2. **Set boundaries** -- explain what it should and should not do
3. **Give examples** -- show the format you expect
4. **Keep it focused** -- shorter prompts often work better, especially
   at the 238M-class scale; every prompt token competes with the conversation
   for the context window

---

## Prompts in Training Data

The same rules apply to the `system` turns of SFT records
(`data/sft/*.jsonl`): the model generalizes tool use and persona from
the system prompts it was trained with, so varied, honest system
prompts in training data beat a single fixed one. See
[training_guide.md](training_guide.md).

---

## Prompt Templates

### Chat
```
You are a helpful assistant. Answer questions clearly and concisely.
```

### Code Review
```
You are a senior developer reviewing code. Point out bugs, suggest
improvements, and explain your reasoning. Be constructive.
```

### Creative Writing
```
You are a creative writer. Write vivid, engaging prose. Use sensory
details and strong verbs. Vary sentence length for rhythm.
```
