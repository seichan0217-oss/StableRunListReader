import json
import os
import re
from pathlib import Path


def replace_environment_variables(input_str, secret_file):
    env_config = {}
    secret_path = Path(secret_file)
    if secret_path.exists():
        with secret_path.open("r", encoding="utf-8") as input_file:
            env_config = json.load(input_file)

    def replace(match):
        var_name = match.group(1)
        if var_name in env_config:
            return str(env_config[var_name])
        env_value = os.getenv(var_name)
        if env_value is not None:
            return env_value
        return match.group(0)

    return re.sub(r"\$([A-Za-z0-9_]+)", replace, input_str)
