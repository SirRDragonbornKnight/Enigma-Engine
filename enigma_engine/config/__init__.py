"""
Forge Configuration Management

This package provides:
- Default configuration
- User configuration loading
- Environment variable support
- Config persistence
"""

from .defaults import CONFIG, get_config, save_config, update_config

__all__ = ["CONFIG", "get_config", "save_config", "update_config"]
