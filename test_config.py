import json
import os

def get_projector_config():
    if os.path.exists("mapping_config.json"):
        try:
            with open("mapping_config.json", "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return {"points": data}
                if isinstance(data, dict):
                    return data
        except: pass
    return {}

# Test 1: Legacy List
with open("mapping_config.json", "w") as f:
    json.dump([1, 2, 3], f)
config = get_projector_config()
print(f"List test: {type(config)} - {config}")
assert isinstance(config, dict)
assert "points" in config

# Test 2: Dict
with open("mapping_config.json", "w") as f:
    json.dump({"points": [1, 2, 3], "monitor_index": 1}, f)
config = get_projector_config()
print(f"Dict test: {type(config)} - {config}")
assert isinstance(config, dict)
assert config["monitor_index"] == 1

print("Tests passed!")
