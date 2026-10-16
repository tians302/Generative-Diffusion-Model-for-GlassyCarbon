import os
import yaml

# GC_CONFIG lets you point every script at another config (e.g. a short smoke-test run) without editing this one
def load_config(config_file=None):
    config_file = config_file or os.environ.get("GC_CONFIG", "config.yaml")
    with open(config_file, "r") as file:
        return yaml.safe_load(file)
