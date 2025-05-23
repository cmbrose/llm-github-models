import llm
from llm.models import Attachment, Conversation, Prompt, Response
from typing import Optional, Iterator, List

from azure.ai.inference import ChatCompletionsClient, EmbeddingsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import (
    ChatRequestMessage,
    AssistantMessage,
    AudioContentItem,
    TextContentItem,
    ImageContentItem,
    ContentItem,
    InputAudio,
    AudioContentFormat,
    ImageDetailLevel,
    ImageUrl,
    SystemMessage,
    UserMessage,
)

INFERENCE_ENDPOINT = "https://models.inference.ai.azure.com"

CHAT_MODELS = [
    ('AI21-Jamba-1.5-Large', True, ['text'], ['text'], None),
    ('AI21-Jamba-1.5-Mini', True, ['text'], ['text'], None),
    ('Codestral-2501', True, ['text'], ['text'], None),
    ('Cohere-command-r', True, ['text'], ['text'], None),
    ('Cohere-command-r-08-2024', True, ['text'], ['text'], None),
    ('Cohere-command-r-plus', True, ['text'], ['text'], None),
    ('Cohere-command-r-plus-08-2024', True, ['text'], ['text'], None),
    ('DeepSeek-R1', True, ['text'], ['text'], None),
    ('DeepSeek-V3-0324', True, ['text'], ['text'], None),
    ('Llama-3.2-11B-Vision-Instruct',
        True,
        ['text', 'image', 'audio'],
        ['text'],
        None),
    ('Llama-3.2-90B-Vision-Instruct',
        True,
        ['text', 'image', 'audio'],
        ['text'],
        None),
    ('Llama-3.3-70B-Instruct', True, ['text'], ['text'], None),
    ('Llama-4-Maverick-17B-128E-Instruct-FP8',
        True,
        ['text', 'image'],
        ['text'],
        None),
    ('Llama-4-Scout-17B-16E-Instruct', True,
     ['text', 'image'], ['text'], None),
    ('MAI-DS-R1', True, ['text'], ['text'], None),
    ('Meta-Llama-3-70B-Instruct', True, ['text'], ['text'], None),
    ('Meta-Llama-3-8B-Instruct', True, ['text'], ['text'], None),
    ('Meta-Llama-3.1-405B-Instruct', True, ['text'], ['text'], None),
    ('Meta-Llama-3.1-70B-Instruct', True, ['text'], ['text'], None),
    ('Meta-Llama-3.1-8B-Instruct', True, ['text'], ['text'], None),
    ('Ministral-3B', True, ['text'], ['text'], None),
    ('Mistral-Large-2411', True, ['text'], ['text'], None),
    ('Mistral-Nemo', True, ['text'], ['text'], None),
    ('Phi-3-medium-128k-instruct', True, ['text'], ['text'], None),
    ('Phi-3-medium-4k-instruct', True, ['text'], ['text'], None),
    ('Phi-3-mini-128k-instruct', True, ['text'], ['text'], None),
    ('Phi-3-mini-4k-instruct', True, ['text'], ['text'], None),
    ('Phi-3-small-128k-instruct', True, ['text'], ['text'], None),
    ('Phi-3-small-8k-instruct', True, ['text'], ['text'], None),
    ('Phi-3.5-MoE-instruct', True, ['text'], ['text'], None),
    ('Phi-3.5-mini-instruct', True, ['text'], ['text'], None),
    ('Phi-3.5-vision-instruct', True, ['text', 'image'], [], None),
    ('Phi-4', True, ['text'], ['text'], None),
    ('Phi-4-mini-instruct', True, ['text'], ['text'], None),
    ('Phi-4-mini-reasoning', True, ['text'], ['text'], None),
    ('Phi-4-multimodal-instruct',
        True,
        ['audio', 'image', 'text'],
        ['text'],
        None),
    ('Phi-4-reasoning', True, ['text'], ['text'], None),
    ('cohere-command-a', True, ['text'], ['text'], None),
    ('gpt-4.1', True, ['text', 'image'], ['text'], None),
    ('gpt-4.1-mini', True, ['text', 'image'], ['text'], None),
    ('gpt-4.1-nano', True, ['text', 'image'], ['text'], None),
    ('gpt-4o', True, ['text', 'image', 'audio'], ['text'], None),
    ('gpt-4o-mini', True, ['text', 'image', 'audio'], ['text'], None),
    ('jais-30b-chat', True, ['text'], ['text'], None),
    ('mistral-medium-2505', True, ['text', 'image'], ['text'], None),
    ('mistral-small-2503', True, ['text', 'image'], ['text'], None),
    ('o1', False, ['text', 'image'], ['text'], None),
    ('o1-mini', False, ['text'], ['text'], None),
    ('o1-preview', False, ['text'], ['text'], None),
    ('o3', False, ['text', 'image'], ['text'], '2024-12-01-preview'),
    ('o3-mini', False, ['text'], ['text'], '2024-12-01-preview'),
    ('o4-mini', False, ['text', 'image'], ['text'], '2024-12-01-preview')
]

EMBEDDING_MODELS = [
    ('Cohere-embed-v3-english', []),
    ('Cohere-embed-v3-multilingual', []),
    ('text-embedding-3-large', [256, 1024]),
    ('text-embedding-3-small', [512])
]


@llm.hookimpl
def register_models(register):
    # Register both sync and async versions of each model
    # TODO: Dynamically fetch this list
    for model_id, can_stream, input_modalities, output_modalities, api_version in CHAT_MODELS:
        register(
            GitHubModels(
                model_id,
                can_stream=can_stream,
                input_modalities=input_modalities,
                output_modalities=output_modalities,
                api_version=api_version
            )
        )


@llm.hookimpl
def register_embedding_models(register):
    # Register embedding models
    for model_id, supported_dimensions in EMBEDDING_MODELS:
        register(
            GitHubEmbeddingModel(model_id)
        )
        for dimensions in supported_dimensions:
            register(
                GitHubEmbeddingModel(
                    model_id,
                    dimensions=dimensions
                )
            )


IMAGE_ATTACHMENTS = {
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
}

AUDIO_ATTACHMENTS = {
    "audio/wav",
    "audio/mpeg",
}


def attachment_as_content_item(attachment: Attachment) -> ContentItem:
    if attachment.resolve_type().startswith("audio/"):
        audio_format = (
            AudioContentFormat.WAV
            if attachment.resolve_type() == "audio/wav"
            else AudioContentFormat.MP3
        )
        return AudioContentItem(
            input_audio=InputAudio.load(
                audio_file=attachment.path, audio_format=audio_format
            )
        )
    if attachment.resolve_type().startswith("image/"):
        if attachment.url:
            return ImageContentItem(
                image_url=ImageUrl(
                    url=attachment.url,
                    detail=ImageDetailLevel.AUTO,
                ),
            )
        if attachment.path:
            return ImageContentItem(
                image_url=ImageUrl.load(
                    image_file=attachment.path,
                    image_format=attachment.resolve_type().split("/")[1],
                    detail=ImageDetailLevel.AUTO,
                ),
            )

    raise ValueError(
        f"Unsupported attachment type: {attachment.resolve_type()}")


def build_messages(
    prompt: Prompt, conversation: Optional[Conversation]
) -> List[ChatRequestMessage]:
    messages: List[ChatRequestMessage] = []
    current_system = None
    if conversation is not None:
        for prev_response in conversation.responses:
            if (
                prev_response.prompt.system
                and prev_response.prompt.system != current_system
            ):
                messages.append(SystemMessage(prev_response.prompt.system))
                current_system = prev_response.prompt.system
            if prev_response.attachments:
                attachment_message: list[ContentItem] = []
                if prev_response.prompt.prompt:
                    attachment_message.append(
                        TextContentItem(text=prev_response.prompt.prompt)
                    )
                for attachment in prev_response.attachments:
                    attachment_message.append(
                        attachment_as_content_item(attachment))
                messages.append(UserMessage(attachment_message))
            else:
                messages.append(UserMessage(prev_response.prompt.prompt))
            messages.append(AssistantMessage(prev_response.text_or_raise()))
    if prompt.system and prompt.system != current_system:
        messages.append(SystemMessage(prompt.system))
    if not prompt.attachments:
        messages.append(UserMessage(content=prompt.prompt))
    else:
        attachment_message = []
        if prompt.prompt:
            attachment_message.append(TextContentItem(text=prompt.prompt))
        for attachment in prompt.attachments:
            attachment_message.append(attachment_as_content_item(attachment))
        messages.append(UserMessage(attachment_message))
    return messages


class GitHubModels(llm.Model):
    needs_key = "github"
    key_env_var = "GITHUB_MODELS_KEY"

    def __init__(
        self,
        model_id: str,
        can_stream: bool,
        input_modalities: Optional[List[str]] = None,
        output_modalities: Optional[List[str]] = None,
        api_version: Optional[str] = None,
    ):
        self.model_id = f"github/{model_id}"
        self.model_name = model_id
        self.can_stream = can_stream
        self.attachment_types = set()
        if input_modalities and "image" in input_modalities:
            self.attachment_types.update(IMAGE_ATTACHMENTS)
        if input_modalities and "audio" in input_modalities:
            self.attachment_types.update(AUDIO_ATTACHMENTS)

        self.input_modalities = input_modalities
        self.output_modalities = output_modalities
        self.api_version = api_version

    def execute(
        self,
        prompt: Prompt,
        stream: bool,
        response: Response,
        conversation: Optional[Conversation],
    ) -> Iterator[str]:
        key = self.get_key()

        extra = {}
        if self.api_version:
            extra["api_version"] = self.api_version

        client = ChatCompletionsClient(
            endpoint=INFERENCE_ENDPOINT,
            credential=AzureKeyCredential(key),
            model=self.model_name,
            **extra,
        )
        messages = build_messages(prompt, conversation)
        if stream:
            completion = client.complete(
                messages=messages,
                stream=True,
            )
            chunks = []
            for chunk in completion:
                chunks.append(chunk)
                try:
                    content = chunk.choices[0].delta.content
                except IndexError:
                    content = None
                if content is not None:
                    yield content
            response.response_json = None  # TODO
        else:
            completion = client.complete(
                messages=messages,
                stream=False,
            )
            response.response_json = None  # TODO
            yield completion.choices[0].message.content


class GitHubEmbeddingModel(llm.EmbeddingModel):
    needs_key = "github"
    key_env_var = "GITHUB_MODELS_KEY"
    batch_size = 100

    def __init__(self, model_id: str, dimensions: Optional[int] = None):
        self.model_id = f"github/{model_id}"
        if dimensions is not None:
            self.model_id += f"-{dimensions}"

        self.model_name = model_id
        self.dimensions = dimensions

    def embed_batch(self, texts: List[str]) -> Iterator[List[float]]:
        if not texts:
            return []

        key = self.get_key()
        client = EmbeddingsClient(
            endpoint=INFERENCE_ENDPOINT,
            credential=AzureKeyCredential(key),
        )

        kwargs = {
            "input": texts,
            "model": self.model_name,
        }
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions

        response = client.embed(**kwargs)
        return ([float(x) for x in item.embedding] for item in response.data)
