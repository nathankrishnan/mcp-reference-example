from core.chat import Chat
from core.claude import Claude
from client.mcp_client import MCPClient

from mcp.types import Prompt, PromptMessage
from anthropic.types import MessageParam


class CliChat(Chat):
    """
    A CLI-specific Chat subclass that adds slash command and @mention support via MCP prompts and resources.
    """

    def __init__(
        self,
        doc_client: MCPClient,
        clients: dict[str, MCPClient],
        claude_service: Claude,
    ):
        """
        Initialize the CliChat instance.

        Args:
            doc_client (MCPClient): A dedicated MCP client for the document server that provides prompts and resources.
            clients (dict[str, MCPClient]): A dictionary mapping client names to MCPClient instances.
            claude_service (Claude): The Claude API wrapper used to send messages.
        """
        super().__init__(clients=clients, claude_service=claude_service)
        self.doc_client = doc_client

    async def list_prompts(self) -> list[Prompt]:
        """Retrieve available prompt templates from the document server."""
        return await self.doc_client.list_prompts()

    async def list_docs_ids(self) -> list[str]:
        """Retrieve available document IDs from the document server."""
        return await self.doc_client.read_resource("docs://documents")

    async def get_doc_content(self, doc_id: str) -> str:
        """
        Retrieve the content of a specific document.

        Args:
            doc_id (str): The document identifier.

        Returns:
            str: The document content.
        """
        return await self.doc_client.read_resource(f"docs://documents/{doc_id}")

    async def get_prompt(self, command: str, doc_id: str) -> list[PromptMessage]:
        """
        Fetch a prompt template rendered with the given document.

        Args:
            command (str): The prompt/command name (e.g. "summarize").
            doc_id (str): The document identifier to pass as an argument.

        Returns:
            list[PromptMessage]: The rendered prompt messages from the MCP server.
        """
        return await self.doc_client.get_prompt(command, {"doc_id": doc_id})

    async def _extract_resources(self, query: str) -> str:
        """
        Find @mentions in the query and fetch the corresponding document content.

        Args:
            query (str): The user's input which may contain @doc_id references.

        Returns:
            str: XML-wrapped document content for all mentioned documents, or empty string if none.
        """
        mentions = [word[1:] for word in query.split() if word.startswith("@")]

        doc_ids = await self.list_docs_ids()
        mentioned_docs: list[tuple[str, str]] = []

        for doc_id in doc_ids:
            if doc_id in mentions:
                content = await self.get_doc_content(doc_id)
                mentioned_docs.append((doc_id, content))

        return "".join(
            f'\n<document id="{doc_id}">\n{content}\n</document>\n'
            for doc_id, content in mentioned_docs
        )

    async def _process_command(self, query: str) -> bool:
        """
        Handle slash commands (e.g. "/summarize report.docx") by fetching
        a prompt template from the MCP server and adding it to the conversation.

        Args:
            query (str): The user's input.

        Returns:
            bool: True if the query was a slash command and was handled, False otherwise.
        """
        if not query.startswith("/"):
            return False

        words = query.split()
        command = words[0].replace("/", "")

        messages = await self.doc_client.get_prompt(command, {"doc_id": words[1]})

        self.messages += convert_prompt_messages_to_message_params(messages)
        return True

    async def _process_query(self, query: str):
        """
        Process a user query by handling slash commands, extracting @mentioned
        document content, and adding the enriched prompt to the conversation.

        Args:
            query (str): The user's input message.
        """
        if await self._process_command(query):
            return

        added_resources = await self._extract_resources(query)

        prompt = f"""
        The user has a question:
        <query>
        {query}
        </query>

        The following context may be useful in answering their question:
        <context>
        {added_resources}
        </context>

        Note the user's query might contain references to documents like "@report.docx". The "@" is only
        included as a way of mentioning the doc. The actual name of the document would be "report.docx".
        If the document content is included in this prompt, you don't need to use an additional tool to read the document.
        Answer the user's question directly and concisely. Start with the exact information they need. 
        Don't refer to or mention the provided context in any way - just use it to inform your answer.
        """

        self.claude_service.add_user_message(self.messages, prompt)


def _extract_text_from_content(content) -> str | None:
    """
    Extract text from a single content item (dict or object).

    Args:
        content: A dict or object that may have "type" and "text" fields.

    Returns:
        str | None: The text string if content is a text-type item, None otherwise.
    """
    if not (isinstance(content, dict) or hasattr(content, "__dict__")):
        return None

    content_type = (
        content.get("type", None)
        if isinstance(content, dict)
        else getattr(content, "type", None)
    )
    if content_type != "text":
        return None

    return (
        content.get("text", "")
        if isinstance(content, dict)
        else getattr(content, "text", "")
    )


def _extract_text_blocks_from_list(content) -> list[dict] | None:
    """
    Extract text blocks from a list of content items.

    Args:
        content: A list of dicts or objects that may have "type" and "text" fields.

    Returns:
        list[dict] | None: A list of text block dicts, or None if no text blocks were found.
    """
    if not isinstance(content, list):
        return None

    text_blocks = []
    for item in content:
        text = _extract_text_from_content(item)
        if text is not None:
            text_blocks.append({"type": "text", "text": text})

    return text_blocks or None


def convert_prompt_message_to_message_param(
    prompt_message: PromptMessage,
) -> MessageParam:
    """
    Convert an MCP PromptMessage to an Anthropic MessageParam.

    Args:
        prompt_message (PromptMessage): The MCP prompt message to convert.

    Returns:
        MessageParam: The equivalent Anthropic message dict.
    """
    role = "user" if prompt_message.role == "user" else "assistant"
    content = prompt_message.content

    text = _extract_text_from_content(content)
    if text is not None:
        return {"role": role, "content": text}

    text_blocks = _extract_text_blocks_from_list(content)
    if text_blocks:
        return {"role": role, "content": text_blocks}

    return {"role": role, "content": ""}


def convert_prompt_messages_to_message_params(
    prompt_messages: list[PromptMessage],
) -> list[MessageParam]:
    """
    Convert a list of MCP PromptMessages to Anthropic MessageParams.

    Args:
        prompt_messages (list[PromptMessage]): The MCP prompt messages to convert.

    Returns:
        list[MessageParam]: The equivalent list of Anthropic message dicts.
    """
    return [convert_prompt_message_to_message_param(msg) for msg in prompt_messages]
