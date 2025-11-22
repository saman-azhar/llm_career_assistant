import yaml
from pathlib import Path

class Config:
    def __init__(self, env: str = None, config_path: str = "config/configs.yml"):
        self.env = env or "dev"
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        # Apply environment-specific overrides
        env_overrides = self.cfg.get("environments", {}).get(self.env, {})
        self._merge_dicts(self.cfg, env_overrides)

    def _merge_dicts(self, base: dict, overrides: dict):
        """Recursively merge overrides into base dict"""
        for k, v in overrides.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                self._merge_dicts(base[k], v)
            else:
                base[k] = v

    def get(self, key: str, default=None):
        """Get config using dot notation, e.g., 'semantic_matching.batch_size'"""
        keys = key.split(".")
        val = self.cfg
        for k in keys:
            val = val.get(k, default)
            if val is default:
                break
        return val

    def all(self):
        return self.cfg
