"""
Parses models.json (created by refresh_models_json.py) to extract chat and embedding models.
Outputs:
- The chat and embedding model tuples which can be copied to llm_github_models.py.
- A Markdown fragment file with model details for the base README.
"""

import json
from pprint import pprint

chat_models = []
embedding_models = []


def supports_streaming(name):
    if name in ["o1", "o1-mini", "o1-preview", "o3-mini", "o3", "o4-mini"]:
        return False
    return True


def api_version(name: str) -> str:
    if name in ["o3-mini", "o3", "o4-mini"]:
        return "2024-12-01-preview"


def embedding_extra_dimensions(name: str) -> list[int]:
    if name == "text-embedding-3-large":
        return [256, 1024]
    elif name == "text-embedding-3-small":
        return [512]
    else:
        return []


with open("models.json", "r", encoding="utf-8") as f:
    models = json.load(f)

models_by_task = {
    "chat-completion": [],
    "embeddings": [],
    "unknown": [],
}
for model in models:
    if model["task"] not in models_by_task:
        models_by_task["unknown"].append(model)
    else:
        models_by_task[model["task"]].append(model)

for model in models_by_task["chat-completion"]:
    chat_models.append(
        (
            model["original_name"],
            supports_streaming(model["name"]),
            model["supported_input_modalities"],
            model["supported_output_modalities"],
            api_version(model["name"]),
        )
    )

for model in models_by_task["embeddings"]:
    embedding_models.append(
        (
            model["original_name"],
            embedding_extra_dimensions(model["name"])
        )
    )

if models_by_task["unknown"]:
    print("Not sure what to do with these models: ", [
        model["name"] for model in models_by_task["unknown"]])

print("Chat models:")
# sort by name
chat_models = sorted(chat_models, key=lambda x: x[0])
pprint(chat_models)
print("Embedding models:")
# sort by name
embedding_models = sorted(embedding_models)
pprint(embedding_models)

# Make a Markdown series for the models

with open("models.fragment.md", "w", encoding="utf-8") as f:
    f.write("## Supported Chat Models\n\n")
    for model in models_by_task["chat-completion"]:
        f.write(f"### {model['friendly_name']}\n\n")
        f.write(f"![Model Image](https://github.com/{model['logo_url']})\n\n")
        f.write(f"Usage: `llm -m github/{model['name']}`\n\n")
        f.write(f"**Publisher:** {model["publisher"]} \n\n")
        f.write(
            f"**Description:** {model["description"].replace("\n## ", "\n#### ")} \n\n")

    f.write("## Supported Embedding Models\n\n")
    for model in models_by_task["embeddings"]:
        f.write(f"### {model['friendly_name']}\n\n")
        f.write(f"![Model Image](https://github.com/{model['logo_url']})\n\n")
        f.write(f"Usage: `llm embed -m github/{model['name']}`\n\n")
        f.write(f"**Publisher:** {model["publisher"]} \n\n")
        f.write(
            f"**Description:** {model["description"].replace("\n## ", "\n#### ")} \n\n")
