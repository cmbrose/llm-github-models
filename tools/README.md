# Model Refreshing Tools

This section documents the tools responsible for refreshing and parsing model metadata from GitHub.

## Refreshing Models

The refresh script retrieves the latest model information by performing the following operations:

1. **Fetching the Catalog**  
Using the GitHub Models Catalog API, the script fetches a list of available models. A valid GitHub token is required via the GITHUB_TOKEN environment variable. The API request is authenticated using this token along with a specific Accept header.

2. **Scraping Detailed Model Information**  
For each model retrieved from the catalog, the tool builds a URL pointing to the GitHub Marketplace details page. The script then scrapes the embedded JSON data contained in a `<script>` tag. This JSON payload provides detailed metadata for the model (e.g., friendly name, description, publisher, logos).

3. **Sorting and Saving Data**  
The collected model details are sorted (by model id) and saved into a file named `models.json` in a human-readable, indented format. This file serves as a cache/storage of current model metadata.

## Parsing Models

After refreshing, another tool parses the `models.json` file to extract specific information for chat and embedding models:

1. **Classification by Task**  
The parser groups models into three categories:
- `chat-completion`
- `embeddings`
- `unknown` (if the task is not recognized)

2. **Extracting Chat Models Data**  
For models marked with the `chat-completion` task, a tuple is created containing:
- The original model name.
- A flag indicating if streaming is supported (with special cases for certain models).
- Supported input modalities (e.g., text, image, audio).
- Supported output modalities.

3. **Extracting Embedding Models Data**  
For models targeted at embeddings, a tuple is created with:
- The original model name.
- Any extra dimensions to consider for specific embedding models.

4. **Generating Markdown Fragments**  
A Markdown fragment (`models.fragment.md`) is generated with detailed model information. This fragment includes:
- A header for chat and embedding models.
- Friendly model names.
- Model logos (as GitHub image links).
- Usage instructions (including the model slug).
- Publisher and enhanced description details.

The fragment can then be added to the repo's README.md

5. **Plugin model definitions**
Tuples for the chat and embedding models are written to stdout:
- Chat models have:
   - `name: str`
   - `supports_streaming: bool`
   - `supported_input_modalities: List[str]`
   - `supported_output_modalities: List[str]`
   - `api_version: str | None`
- Embedding models have:
   - `name: str`
   - `supported_dimensions: List[int]`

These snippets should be copy-pasted into `CHAT_MODELS` and `EMBEDDING_MODELS` of `llm_github_models.py`.

