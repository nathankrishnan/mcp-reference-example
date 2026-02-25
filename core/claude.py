from typing import Optional
from anthropic import Anthropic
from anthropic.types import Message, MessageParam, TextBlockParam, ToolUnionParam


class Claude:
    """
    A wrapper class for the Anthropic API client to facilitate chat interactions.
    """

    def __init__(self, model: str):
        """
        Initialize the Claude client wrapper.

        Args:
            model (str): The name of the Anthropic model to use (e.g. 'claude-sonnet-4-6').
        """
        self.client = Anthropic()
        self.model = model

    def add_user_message(
        self, messages: list[MessageParam], message: str | Message
    ) -> None:
        """
        Append a user message to the conversation history.

        Args:
            messages (list[MessageParam]): The current list of conversation messages.
            message (str | Message): The text content or a Message object to add as the user.
        """
        user_message: MessageParam = {
            "role": "user",
            "content": message.content if isinstance(message, Message) else message,
        }

        messages.append(user_message)

    def add_assistant_message(
        self, messages: list[MessageParam], message: str | Message
    ) -> None:
        """
        Append an assistant message to the conversation history.

        Args:
            messages (list[MessageParam]): The current list of conversation messages.
            message (str | Message): The text content or a Message object to add as the assistant.
        """
        assistant_message: MessageParam = {
            "role": "assistant",
            "content": message.content if isinstance(message, Message) else message,
        }

        messages.append(assistant_message)

    def text_from_message(self, message: Message) -> str:
        """
        Extract and concatenate all text blocks from an Anthropic Message object.

        Args:
            message (Message): The message object returned by the Anthropic API.

        Returns:
            str: The concatenated text content from the message.
        """
        return "\n".join(
            [block.text for block in message.content if block.type == "text"]
        )

    def chat(
        self,
        messages: list[MessageParam],
        system: Optional[str | list[TextBlockParam]] = None,
        temperature: float = 1.0,
        stop_sequences: Optional[list[str]] = None,
        tools: Optional[list[ToolUnionParam]] = None,
        thinking: bool = False,
        thinking_budget: int = 1024,
    ) -> Message:
        """
        Send a chat completion request to the Anthropic API.

        Args:
            messages (list[MessageParam]): The conversation history.
            system (Optional[str | list[TextBlockParam]], optional): System prompt to guide the model's behavior. Defaults to None.
            temperature (float, optional): Randomness of the output (0.0 to 1.0). Defaults to 1.0.
            stop_sequences (Optional[list[str]], optional): Custom sequences that stop generation. Defaults to None.
            tools (Optional[list[ToolUnionParam]], optional): JSON schemas defining available tools. Defaults to None.
            thinking (bool, optional): Whether to enable the model's extended thinking mode. Defaults to False.
            thinking_budget (int, optional): Minimum tokens reserved for thinking (if enabled). Defaults to 1024.

        Returns:
            Message: The message object containing the model's response.
        """
        if stop_sequences is None:
            stop_sequences = []

        params = {
            "model": self.model,
            "max_tokens": 8000,
            "messages": messages,
            "temperature": temperature,
            "stop_sequences": stop_sequences,
        }

        if thinking:
            params["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}

        if tools:
            params["tools"] = tools

        if system:
            params["system"] = system

        message = self.client.messages.create(**params)
        return message
