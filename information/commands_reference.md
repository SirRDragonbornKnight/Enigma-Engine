# Commands Reference

The AI can execute commands using `[CMD]command[/CMD]` blocks in its
responses. You can also run commands from the CMD page in ENGINE mode.

There are **49 registered engine commands** plus **10 info commands**
available in the CMD page.

---

## Engine Commands

### Config
| Command | Description | Usage |
|---------|-------------|-------|
| config.get | Get a config value | `config.get <key>` |
| config.set | Set a config value | `config.set <key> <value>` |
| config.list | List all config values | `config.list` |

### Model
| Command | Description | Usage |
|---------|-------------|-------|
| model.info | Show loaded model info | `model.info` |
| model.list | List available models | `model.list` |
| model.load | Load model by path | `model.load <path>` |
| model.switch | Switch to model by name | `model.switch <name>` |
| model.download | Download from HuggingFace | `model.download <repo_id> [filename]` |

### Training
| Command | Description | Usage |
|---------|-------------|-------|
| train.status | Show training status | `train.status` |
| train.start | Start training | `train.start <data> [--epochs N]` |
| train.stop | Stop training | `train.stop` |
| train.data.add | Add training example | `train.data.add <text>` |
| train.data.list | List training data files | `train.data.list` |

### Files
| Command | Description | Usage |
|---------|-------------|-------|
| file.list | List directory contents | `file.list [path]` |
| file.read | Read a file | `file.read <path>` |
| file.write | Write to a file | `file.write <path> <content>` |
| file.append | Append to a file | `file.append <path> <content>` |

### Search & Web
| Command | Description | Usage |
|---------|-------------|-------|
| search.files | Find files by pattern | `search.files <pattern>` |
| search.content | Search text in files | `search.content <dir> <query>` |
| search.web | Search the web via DuckDuckGo | `search.web <query>` |
| search.images | Search for images via DuckDuckGo | `search.images <query>` |
| web.fetch | Fetch content from a URL | `web.fetch <url>` |

### System
| Command | Description | Usage |
|---------|-------------|-------|
| system.info | Show hardware information | `system.info` |
| system.clear | Clear terminal | `system.clear` |
| clipboard.copy | Copy text to clipboard | `clipboard.copy <text>` |
| shell | Run a terminal command | `shell <command>` |
| stop | Stop current generation | `stop` |
| history | Show command history | `history [count]` |
| help | Show all commands or help for one | `help [command]` |

### Code
| Command | Description | Usage |
|---------|-------------|-------|
| code.run | Execute Python code in sandboxed subprocess | `code.run <python_code>` |

### Image Generation
| Command | Description | Usage |
|---------|-------------|-------|
| imagegen.generate | Generate image from text prompt | `imagegen.generate <prompt> [--width N] [--height N] [--steps N] [--seed N] [--negative <text>]` |
| imagegen.status | Check available image generation backends | `imagegen.status` |

### Mods
| Command | Description | Usage |
|---------|-------------|-------|
| mod.list | List installed mods | `mod.list` |
| mod.status | Show router status | `mod.status` |
| mod.start | Start mod router | `mod.start` |
| mod.stop | Stop mod router | `mod.stop` |
| mod.send | Send command to a mod | `mod.send <mod_id> <cmd> [args]` |

### Conversation Memory
| Command | Description | Usage |
|---------|-------------|-------|
| memory.save | Save conversation to memory | `memory.save <name>` |
| memory.load | Load conversation from memory | `memory.load <name>` |
| memory.list | List saved memory keys | `memory.list` |

### Persistent Memory

These commands manage the AI's long-term memory stored in
`data/notes/memory.md`. See how_the_ai_works.md for details.

| Command | Description | Usage |
|---------|-------------|-------|
| memory.remember | Remember a fact about the user | `memory.remember <fact>` |
| memory.forget | Forget a fact matching keyword | `memory.forget <keyword>` |
| memory.notes | Show all remembered facts | `memory.notes` |
| memory.clear_notes | Clear all persistent memories | `memory.clear_notes` |

### Notes
| Command | Description | Usage |
|---------|-------------|-------|
| note.add | Add a quick note | `note.add <text>` |
| note.list | List recent notes | `note.list` |

---

## CMD Page

### Modes

**SYSTEM mode** — Real PowerShell terminal. Run any system command.
Type `cancel`, `ctrl+c`, or `kill` to terminate a running command.

**ENGINE mode** — Run engine commands from the registry. Also supports
the `ask` command and 10 info commands (see below).

### AI ACCESS Toggle

When **ON**, the AI can execute real system commands through PowerShell.
When **OFF** (default), the AI is restricted to engine commands only.

### `ask` Command

In ENGINE mode, type `ask <question>` to send a question to the
currently loaded AI model and get a response displayed in the terminal.

### Info Commands

These are CMD-page-only shortcuts (not available as engine commands).
They display system information in the terminal:

| Command | Description |
|---------|-------------|
| status | Show overall engine status (model, routes, mods) |
| sysinfo | Show hardware info (CPU, RAM, GPU, VRAM) |
| gpu | Show GPU details (name, VRAM, CUDA version) |
| memory | Show RAM usage |
| models | List available models |
| routes | Show current route assignments |
| sessions | Show saved chat sessions |
| mods | List installed mods and their status |
| data | List training data files |
| uptime | Show how long the engine has been running |

### Status Strip

A live status bar at the bottom of the CMD page showing GPU name,
VRAM, and uptime. Updates automatically.

### Keyboard Shortcuts

- **Up/Down arrows** — navigate command history
- **Enter** — execute command
