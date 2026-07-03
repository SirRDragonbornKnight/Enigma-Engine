# Prompts Guide

Prompts shape how the AI behaves. A good system prompt makes the
difference between useful and useless responses.

---

## Route Prompts

Each AI route has its own editable prompt file in `data/prompts/`.
Edit these from the **DOCS page** under the **Prompts** category.

| File | Route | What It Controls |
|------|-------|-----------------|
| chat.md | CHAT | Default system prompt for chat conversations |
| trainer.md | TRAINER | Instructions prepended to the FORGE trainer AI |

### How Route Prompts Work

- **Chat prompt** — loaded as the default system prompt when a model
  is assigned to the CHAT route for the first time. You can still
  override it per-model using the CORE page sidebar prompt editor.
- **Trainer prompt** — prepended to the built-in trainer system prompt
  in the FORGE. This lets you add custom instructions, personality,
  or domain knowledge that the trainer AI should know before training.

### Creating New Route Prompts

Add a `.md` or `.txt` file to `data/prompts/` with any name.
It will appear in the DOCS page under Prompts for easy editing.

---

## System Prompt (CORE Page)

The CORE page sidebar has a **SYSTEM PROMPT** panel. This is the
per-model prompt — it overrides the default chat.md prompt for
whichever model is currently loaded.

- Each model remembers its own system prompt (saved to its context)
- Editing and clicking APPLY updates the current model's prompt
- When you load a different model, its own prompt is restored
- Per-model prompts are stored in `data/model_contexts/<model>/context.json`

**Example:**
```
You are a helpful coding assistant. You write clean, well-documented
Python code. When asked to fix bugs, explain what went wrong first.
```

### Tips for Good Prompts

1. **Be specific** — tell the AI exactly what role to play
2. **Set boundaries** — explain what it should and should not do
3. **Give examples** — show the format you expect
4. **Keep it focused** — shorter prompts often work better

---

## FORGE Training Prompts

The FORGE uses prompts in several ways to guide training:

### Focus Field

The **Focus field** on the FORGE page lets you specify a domain
(e.g. "medical", "coding", "cooking"). When set, this focus is
injected into all training prompts — the trainer AI generates
data and instructions relevant to that domain.

### Training Stages

When using **Train with AI**, the training stage buttons control
what the trainer focuses on:

| Stage | What the Trainer Teaches |
|-------|-------------------------|
| BASICS | Greetings, short answers, basic facts, vocabulary |
| CONVERSATION | Multi-sentence responses, turn-taking, follow-ups |
| COMMANDS | [CMD]command[/CMD] syntax, when to use commands |
| WEB | search.web, web.fetch, summarize web results |

The selected stage, focus field, and training brief all combine
into the trainer's system prompt via `_build_trainer_system_prompt()`.

### Training Brief

The **Training Brief** panel (shown with Train with AI) lets you
describe the kind of AI you want to create. It includes quick
profile fields and a custom text area. This description is passed
to the trainer system prompt so it knows what personality and
capabilities to develop.

---

## Profiles

Profiles are saved configurations that include:

- **System prompt** — the AI's personality
- **Generation parameters** — temperature, top-p, etc.
- **Memory config** — conversation and memory settings

### Built-in Profiles

| Profile | Description |
|---------|-------------|
| Assistant | General-purpose helpful assistant |
| Coding Helper | Focused on writing and debugging code |
| Creative Writer | Creative writing with higher temperature |
| Researcher | Analytical, thorough research assistant |

Profiles live in the `profiles/` directory and are used by the
AI Profile system (`ai_profile.py`) and the web API (`server.py`).

### Creating a Profile

Create a JSON file in the `profiles/` directory:

```json
{
  "name": "My Custom Profile",
  "id": "my_custom",
  "description": "A custom AI personality",
  "system_prompt": "You are a friendly AI that loves puns.",
  "generation": {
    "temperature": 0.8,
    "top_p": 0.9,
    "max_tokens": 2048
  }
}
```

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

### Trainer — Domain Expert
```
You are a domain expert in [TOPIC]. When generating training data,
focus on practical real-world knowledge. Include specific examples,
common mistakes to avoid, and hands-on exercises.
```
