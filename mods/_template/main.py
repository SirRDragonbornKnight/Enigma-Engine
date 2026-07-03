"""
YOUR MOD - Edit This File!

HOW TO CREATE YOUR MOD:
1. Copy this folder (mods/_template/) to mods/yourmod/
2. Edit mod.json - set your mod's id, name, commands, and ui widgets
3. Edit THIS FILE - implement your command handlers below

WHAT YOU EDIT:
    cmd_generate()     - Your main command logic
    cmd_*()            - Any other commands (match mod.json "commands")

WHAT YOU DON'T EDIT:
    mod_base.py        - Connection protocol (handled for you)
    connect/register   - Handled by base class

UI WIDGETS:
    Your mod's interface is defined in mod.json under "ui".widgets.
    The desktop GUI renders these automatically. Supported widget types:
        text_input  - Single-line text entry
        text_area   - Multi-line text box (set "rows" for height)
        number      - Numeric entry with default value
        button      - Sends a command (set "command" to match a cmd_* method)
        dropdown    - Selection from options list (set "options" and "default")
        checkbox    - Boolean toggle (set "default" to true/false)

    When a button is clicked, the GUI gathers all widget values by their
    "id" field and passes them as args to your cmd_* handler.

THREADING MODEL:
    All cmd_* handlers are plain synchronous methods. Each mod runs
    in its own subprocess, so blocking I/O (HTTP requests, file ops,
    heavy computation) is perfectly fine — it won't block the GUI.
"""

import logging
import time
from typing import Any, Dict

# Import the base class (DO NOT EDIT mod_base.py)
from mod_base import ModClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# YOUR MOD CLASS - EDIT THIS!
# =============================================================================


class MyMod(ModClient):
    """
    Your mod implementation. Rename this class to match your mod.

    Example: ImageGenMod, AudioMod, CodeMod, etc.
    """

    # =========================================================================
    # REQUIRED: Your main command
    # =========================================================================

    def cmd_generate(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Your main command. Called when AI sends:
        [CMD]mod.send yourmod generate prompt=...[/CMD]

        EDIT THIS to do what your mod actually does!

        Args:
            args: {"prompt": "...", ...} from AI or GUI

        Returns:
            Result dict sent back to AI
        """
        prompt = args.get("prompt", "")

        # =====================================================================
        # YOUR LOGIC HERE - replace this placeholder!
        # =====================================================================

        # Example: Send progress updates
        self.send_update("progress", {"percent": 0, "message": "Starting..."})

        # Do your work here...
        time.sleep(0.5)  # Simulated work

        self.send_update("progress", {"percent": 50, "message": "Processing..."})

        # More work...
        time.sleep(0.5)  # Simulated work

        self.send_update("progress", {"percent": 100, "message": "Done!"})

        # Return result to AI
        return {
            "result": f"Generated from: {prompt}",
            "type": "text",  # or "image", "audio", etc.
        }

    # =========================================================================
    # OPTIONAL: Additional commands (match mod.json "commands")
    # =========================================================================

    def cmd_status(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Return mod status. Called by AI: [CMD]mod.status yourmod[/CMD]"""
        return {
            "name": self.name,
            "mod_id": self.mod_id,
            "running": self.running,
            # Add your custom status info here
        }

    def cmd_stop(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Stop the mod. Called by AI: [CMD]mod.stop yourmod[/CMD]"""
        self.running = False
        return {"stopped": True}


# =============================================================================
# ENTRY POINT - runs your mod
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("  My Mod — Enigma Engine Plugin")
    print("=" * 50)
    print()
    print("Connecting to router on port 9900...")
    print("Make sure the router is running first!")
    print()

    mod = MyMod()
    try:
        mod.run()
    except KeyboardInterrupt:
        logger.info("Mod stopped by user")
