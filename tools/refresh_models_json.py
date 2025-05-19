"""
Refreshes the models.json file by fetching the latest model information from the GitHub API.
First the GitHub Models Catalog is fetched, then each model's details are scraped from the GitHub Marketplace page.

This script requires the GITHUB_TOKEN environment variable to be set with a valid GitHub token.
"""

import json
import os
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


def fetch_catalog():
    github_token = os.environ.get("GITHUB_TOKEN")
    if not github_token:
        raise RuntimeError("GITHUB_TOKEN environment variable not set")

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {github_token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    url = "https://models.github.ai/catalog/models"
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()


def build_model_url(model_id: str) -> str:
    """
    Build the URL for a model's details page.
    """
    publisher, name = model_id.split("/", 1)

    azureml_part = (
        "azureml" if publisher == "microsoft"
        else "azure-openai" if publisher == "openai"
        else f"azureml-{publisher.split('-')[0]}"
    )

    name_part = name.replace(".", "-")

    return f"https://github.com/marketplace/models/{azureml_part}/{name_part}"


def scrape_models_json(model_id: str) -> dict:
    """
    Scrape the models.json file for a specific model.
    """
    url = build_model_url(model_id)
    response = requests.get(url)
    response.raise_for_status()
    html = response.text
    soup = BeautifulSoup(html, "html.parser")
    script_tag = soup.find(
        "script", {"type": "application/json", "data-target": "react-app.embeddedData"})
    if not script_tag:
        raise ValueError("Could not find the embedded JSON data script tag")
    data = json.loads(script_tag.string)
    return data["payload"]["model"]


catalog = fetch_catalog()
models = []
for entry in tqdm(catalog, "Fetching models from GitHub catalog"):
    models.append(scrape_models_json(entry["id"]))

models = sorted(models, key=lambda x: x["id"])

with open("models.json", "w", encoding="utf-8") as cache_file:
    json.dump(models, cache_file, indent=2, ensure_ascii=False)
