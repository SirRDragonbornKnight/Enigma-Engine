#!/usr/bin/env python3
"""
Code Generation - Standalone AI Code Generator

Generates code from natural language prompts using:
- Local Enigma model (default) - the project's own trainable LLM; code is
  Enigma's primary skill per the project goals, not an external bridge.
- Template provider (explicit opt-in) - deterministic keyword-template
  generation for offline / no-model environments. Never silently substituted
  for the local engine; users pick it explicitly via --provider template.

2.1-codegen slice (May 25 2026): killed `LocalCode._fallback = TemplateCode()`
silent fallback (same anti-pattern as imagegen/threed/videogen/audiogen);
flipped default template -> local. Engine choice: Enigma itself rather than
an external model (Qwen2.5-Coder, DeepSeek-Coder, StarCoder2, CodeLlama) -
because the project goal is to TRAIN Enigma for code generation, not to
delegate to a pre-finished competitor.

Usage:
    python codegen.py                              # Start service
    python codegen.py --port 9902                 # Custom port
    python codegen.py --generate "sort a list"   # Generate code
    python codegen.py --language python          # Language setting
    python codegen.py --provider template          # Offline template mode
"""

import argparse
import json
import logging
import re
import socket
import struct
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("outputs/code")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Protocol
# =============================================================================


class MessageType(Enum):
    REGISTER = "register"
    COMMAND = "command"
    RESPONSE = "response"
    EVENT = "event"
    HEARTBEAT = "heartbeat"
    SHUTDOWN = "shutdown"


@dataclass
class Message:
    type: MessageType
    payload: Dict[str, Any] = field(default_factory=dict)
    id: str = ""

    def to_bytes(self) -> bytes:
        data = json.dumps({"type": self.type.value, "payload": self.payload, "id": self.id}).encode("utf-8")
        return struct.pack(">I", len(data)) + data

    @classmethod
    def from_bytes(cls, data: bytes) -> "Message":
        obj = json.loads(data.decode("utf-8"))
        return cls(type=MessageType(obj["type"]), payload=obj.get("payload", {}), id=obj.get("id", ""))


# =============================================================================
# Code Generation Providers
# =============================================================================


class TemplateCode:
    """Template-based code generation for common patterns."""

    TEMPLATES = {
        "python": {
            "sort": '''def sort_list(items, reverse=False):
    """Sort a list of items."""
    return sorted(items, reverse=reverse)

# Example
numbers = [3, 1, 4, 1, 5, 9, 2, 6]
print(sort_list(numbers))  # [1, 1, 2, 3, 4, 5, 6, 9]
''',
            "class": '''class {name}:
    """A {name} class."""
    
    def __init__(self):
        pass
    
    def __repr__(self):
        return f"{name}()"

# Example
obj = {name}()
print(obj)
''',
            "function": '''def {name}(*args, **kwargs):
    """Function to {description}."""
    result = None

    # Process input
    for arg in args:
        pass  # Handle each argument

    return result


# Example
# result = {name}()
# print(result)
''',
            "api": '''import requests

def fetch_data(url: str) -> dict:
    """Fetch data from an API endpoint."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()

# Example
# data = fetch_data("https://api.example.com/data")
''',
            "file": '''def read_file(filepath: str) -> str:
    """Read contents of a file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def write_file(filepath: str, content: str):
    """Write content to a file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
''',
            "loop": """# Iterate over items
items = [1, 2, 3, 4, 5]

for item in items:
    print(item)

# With index
for i, item in enumerate(items):
    print(f"{i}: {item}")

# List comprehension
doubled = [x * 2 for x in items]
""",
        },
        "javascript": {
            "sort": """function sortArray(items, reverse = false) {
    return [...items].sort((a, b) => reverse ? b - a : a - b);
}

// Example
const numbers = [3, 1, 4, 1, 5, 9, 2, 6];
console.log(sortArray(numbers));
""",
            "class": """class {name} {{
    constructor() {{
        // Initialize
    }}
    
    toString() {{
        return `{name} instance`;
    }}
}}

// Example
const obj = new {name}();
console.log(obj.toString());
""",
            "function": """function {name}(...args) {{
    // {description}
    let result = null;

    // Process input
    for (const arg of args) {{
        // Handle argument
    }}

    return result;
}}

// Example
// const result = {name}();
// console.log(result);
""",
            "api": """async function fetchData(url) {
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`HTTP error: ${response.status}`);
    }
    return response.json();
}

// Example
// const data = await fetchData("https://api.example.com/data");
""",
        },
    }

    def __init__(self):
        self.is_loaded = False

    def load(self) -> bool:
        self.is_loaded = True
        return True

    def unload(self):
        self.is_loaded = False

    def _find_template(self, prompt: str, language: str) -> tuple:
        """Find matching template based on keywords."""
        prompt_lower = prompt.lower()
        templates = self.TEMPLATES.get(language, self.TEMPLATES.get("python", {}))

        keywords = {
            "sort": ["sort", "order", "arrange"],
            "class": ["class", "object", "create a class"],
            "function": ["function", "def", "method", "create a function"],
            "api": ["api", "fetch", "request", "http", "rest"],
            "file": ["file", "read", "write", "open"],
            "loop": ["loop", "iterate", "for each", "foreach"],
        }

        for key, words in keywords.items():
            if any(w in prompt_lower for w in words):
                return key, templates.get(key, "")

        return "function", templates.get("function", "# Code generation")

    def generate(self, prompt: str, language: str = "python", **kwargs) -> Dict[str, Any]:
        if not self.is_loaded:
            return {"success": False, "error": "Not loaded"}

        try:
            start = time.time()
            key, template = self._find_template(prompt, language)

            # Extract name from prompt
            name_match = re.search(r'(?:called|named|for)\s+["\']?(\w+)["\']?', prompt, re.I)
            name = name_match.group(1) if name_match else "MyCode"

            code = template.format(name=name, description=prompt[:50])

            return {
                "success": True,
                "code": code,
                "language": language,
                "template": key,
                "duration": time.time() - start,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


class LocalCode:
    """Local code generation using the in-tree Enigma model.

    No silent fallback to TemplateCode - load failures are surfaced loud
    (logger.error + return False) so the user knows the local engine isn't
    serving requests. Users who want template-based output must pass
    --provider template explicitly. This matches the loud-on-real-issue
    pattern shipped across imagegen/threed/videogen/audiogen in slice 2.1.
    """

    def __init__(self, model_name: str = "small_enigma_engine"):
        self.model_name = model_name
        self.engine = None
        self.is_loaded = False

    def load(self) -> bool:
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from enigma_engine.core.model_registry import ModelRegistry
            from enigma_engine.core.inference import EnigmaEngine
            from enigma_engine.core.tokenizer import get_tokenizer
            import torch
        except ImportError as e:
            logger.error(
                f"Enigma core not importable for LocalCode: {e}. "
                "Run from the Enigma Engine repo root, or use --provider template."
            )
            return False

        try:
            registry = ModelRegistry()
            model, config = registry.load_model(self.model_name)

            self.engine = EnigmaEngine.__new__(EnigmaEngine)
            self.engine.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.engine.model = model
            self.engine.model.to(self.engine.device)
            self.engine.model.eval()
            self.engine.tokenizer = get_tokenizer()
            self.engine.use_half = False
            self.engine.enable_tools = False
            self.engine.module_manager = None
            self.engine._tool_executor = None

            self.is_loaded = True
            logger.info(f"Loaded Enigma model: {self.model_name}")
            return True
        except Exception as e:
            logger.error(
                f"Enigma model '{self.model_name}' failed to load: {e}. "
                "Train or register the model, or use --provider template."
            )
            self.engine = None
            self.is_loaded = False
            return False

    def unload(self):
        self.engine = None
        self.is_loaded = False

    def generate(self, prompt: str, language: str = "python", **kwargs) -> Dict[str, Any]:
        if not self.is_loaded or self.engine is None:
            return {
                "success": False,
                "error": "Local Enigma provider not loaded (use --provider template for offline mode)",
            }

        try:
            start = time.time()
            code_prompt = f"Write {language} code:\n{prompt}\n\n```{language}\n"

            result = self.engine.generate(
                code_prompt,
                max_tokens=kwargs.get("max_tokens", 500),
                temperature=kwargs.get("temperature", 0.3),
            )

            return {"success": True, "code": result, "language": language, "duration": time.time() - start}
        except Exception as e:
            return {"success": False, "error": str(e)}


# =============================================================================
# CodeGen Service
# =============================================================================


class CodeGen:
    """Code Generation Service."""

    PROVIDERS = {
        "template": TemplateCode,
        "local": LocalCode,
    }

    def __init__(self, default_provider: str = "local"):
        # 2.1-codegen slice (May 25 2026): flipped default template -> local.
        # Enigma is the project's trainable code model; template stays as
        # explicit opt-in (--provider template) for offline / no-model usage.
        self.providers: Dict[str, Any] = {}
        self.default_provider = default_provider
        self._running = False
        self._socket: Optional[socket.socket] = None

        self.commands = {
            "generate": self._cmd_generate,
            "load_provider": self._cmd_load_provider,
            "unload_provider": self._cmd_unload_provider,
            "list_providers": self._cmd_list_providers,
            "set_default": self._cmd_set_default,
            "status": self._cmd_status,
            "save_code": self._cmd_save_code,
        }

    def get_provider(self, name: str):
        if name not in self.providers:
            if name not in self.PROVIDERS:
                return None
            self.providers[name] = self.PROVIDERS[name]()
        return self.providers[name]

    def _cmd_generate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        prompt = params.get("prompt", "")
        language = params.get("language", "python")
        provider_name = params.get("provider", self.default_provider)

        provider = self.get_provider(provider_name)
        if not provider:
            return {"success": False, "error": f"Unknown provider: {provider_name}"}

        if not provider.is_loaded:
            if not provider.load():
                return {"success": False, "error": f"Failed to load {provider_name}"}

        # prompt/language/provider are passed explicitly; only forward the rest
        # (e.g. max_tokens, temperature) so we don't double-pass keywords.
        extra = {k: v for k, v in params.items() if k not in ("prompt", "language", "provider")}
        return provider.generate(prompt, language=language, **extra)

    def _cmd_load_provider(self, params: Dict[str, Any]) -> Dict[str, Any]:
        name = params.get("provider", self.default_provider)
        provider = self.get_provider(name)
        if not provider:
            return {"success": False, "error": f"Unknown provider: {name}"}

        if provider.is_loaded:
            return {"success": True, "message": f"{name} already loaded"}

        success = provider.load()
        return {"success": success, "provider": name}

    def _cmd_unload_provider(self, params: Dict[str, Any]) -> Dict[str, Any]:
        name = params.get("provider", self.default_provider)
        provider = self.providers.get(name)
        if provider:
            provider.unload()
        return {"success": True, "provider": name}

    def _cmd_list_providers(self, params: Dict[str, Any]) -> Dict[str, Any]:
        result = {}
        for name, cls in self.PROVIDERS.items():
            provider = self.providers.get(name)
            result[name] = {"loaded": provider.is_loaded if provider else False, "class": cls.__name__}
        return {"success": True, "providers": result, "default": self.default_provider}

    def _cmd_set_default(self, params: Dict[str, Any]) -> Dict[str, Any]:
        name = params.get("provider")
        if name not in self.PROVIDERS:
            return {"success": False, "error": f"Unknown provider: {name}"}
        self.default_provider = name
        return {"success": True, "default": name}

    def _cmd_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        loaded = [n for n, p in self.providers.items() if p.is_loaded]
        return {"success": True, "service": "codegen", "loaded_providers": loaded, "default": self.default_provider}

    def _cmd_save_code(self, params: Dict[str, Any]) -> Dict[str, Any]:
        code = params.get("code", "")
        filename = params.get("filename")
        language = params.get("language", "python")

        if not filename:
            ext = {
                "python": ".py",
                "javascript": ".js",
                "typescript": ".ts",
                "java": ".java",
                "cpp": ".cpp",
                "c": ".c",
            }.get(language, ".txt")
            filename = f"code_{int(time.time())}{ext}"

        filepath = OUTPUT_DIR / filename
        filepath.write_text(code)
        return {"success": True, "path": str(filepath)}

    def handle_command(self, cmd: str, params: Dict[str, Any]) -> Dict[str, Any]:
        handler = self.commands.get(cmd)
        if not handler:
            return {"success": False, "error": f"Unknown command: {cmd}"}
        try:
            return handler(params)
        except Exception as e:
            return {"success": False, "error": str(e)}

    def connect_to_router(self, host: str = "localhost", port: int = 9900):
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.connect((host, port))

            reg_msg = Message(
                type=MessageType.REGISTER,
                payload={
                    "name": "codegen",
                    "capabilities": ["generate_code", "code"],
                    "commands": list(self.commands.keys()),
                },
            )
            self._socket.sendall(reg_msg.to_bytes())

            self._running = True
            logger.info(f"Connected to router at {host}:{port}")

            while self._running:
                try:
                    len_data = self._socket.recv(4)
                    if not len_data:
                        break
                    msg_len = struct.unpack(">I", len_data)[0]
                    msg_data = b""
                    while len(msg_data) < msg_len:
                        chunk = self._socket.recv(min(4096, msg_len - len(msg_data)))
                        if not chunk:
                            break
                        msg_data += chunk

                    msg = Message.from_bytes(msg_data)

                    if msg.type == MessageType.COMMAND:
                        cmd = msg.payload.get("command", "")
                        params = msg.payload.get("params", {})
                        result = self.handle_command(cmd, params)

                        resp = Message(type=MessageType.RESPONSE, payload=result, id=msg.id)
                        self._socket.sendall(resp.to_bytes())

                    elif msg.type == MessageType.SHUTDOWN:
                        self._running = False

                except socket.timeout:
                    continue
                except Exception as e:
                    logger.error(f"Error: {e}")
                    break

        except ConnectionRefusedError:
            logger.warning("Could not connect to router")
        except Exception as e:
            logger.error(f"Connection error: {e}")
        finally:
            if self._socket:
                self._socket.close()

    def run_standalone(self, port: int = 9902):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("localhost", port))
        server.listen(5)

        logger.info(f"CodeGen server listening on port {port}")
        self._running = True

        while self._running:
            try:
                server.settimeout(1.0)
                try:
                    client, addr = server.accept()
                except socket.timeout:
                    continue

                threading.Thread(target=self._handle_client, args=(client,), daemon=True).start()
            except Exception as e:
                logger.error(f"Server error: {e}")

        server.close()

    def _handle_client(self, client: socket.socket):
        try:
            while self._running:
                len_data = client.recv(4)
                if not len_data:
                    break
                msg_len = struct.unpack(">I", len_data)[0]
                msg_data = b""
                while len(msg_data) < msg_len:
                    chunk = client.recv(min(4096, msg_len - len(msg_data)))
                    if not chunk:
                        break
                    msg_data += chunk

                msg = Message.from_bytes(msg_data)

                if msg.type == MessageType.COMMAND:
                    cmd = msg.payload.get("command", "")
                    params = msg.payload.get("params", {})
                    result = self.handle_command(cmd, params)

                    resp = Message(type=MessageType.RESPONSE, payload=result, id=msg.id)
                    client.sendall(resp.to_bytes())
        except Exception as e:
            logger.error(f"Client error: {e}")
        finally:
            client.close()

    def shutdown(self):
        self._running = False
        for provider in self.providers.values():
            if provider.is_loaded:
                provider.unload()


def main():
    parser = argparse.ArgumentParser(description="Code Generation Service")
    parser.add_argument("--port", type=int, default=9902, help="Server port")
    parser.add_argument("--router", type=str, help="Router address (host:port)")
    parser.add_argument(
        "--provider",
        type=str,
        default="local",
        choices=["template", "local"],
        help="Default provider (local=Enigma model, template=offline keyword templates)",
    )
    parser.add_argument("--generate", type=str, help="Generate code")
    parser.add_argument("--language", type=str, default="python")

    args = parser.parse_args()

    service = CodeGen(default_provider=args.provider)

    if args.generate:
        result = service.handle_command(
            "generate", {"prompt": args.generate, "language": args.language, "provider": args.provider}
        )
        if result.get("success"):
            print(result.get("code", ""))
        else:
            print(f"Error: {result.get('error')}", file=sys.stderr)
        return

    try:
        if args.router:
            host, port = args.router.split(":")
            service.connect_to_router(host, int(port))
        else:
            service.run_standalone(args.port)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        service.shutdown()


if __name__ == "__main__":
    main()
