from typing import Iterable, Optional
from core.cli_chat import CliChat

from mcp.types import Prompt
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.auto_suggest import AutoSuggest, Suggestion
from prompt_toolkit.document import Document
from prompt_toolkit.buffer import Buffer


class CommandAutoSuggest(AutoSuggest):
    """Provides ghost-text suggestions for slash command arguments."""

    def __init__(self, prompts: list[Prompt]):
        """
        Initialize the auto-suggester with available prompts.

        Args:
            prompts (list[Prompt]): The list of MCP prompts available for slash commands.
        """
        self.prompts = prompts
        self.prompt_dict = {prompt.name: prompt for prompt in prompts}

    def get_suggestion(
        self, buffer: Buffer, document: Document
    ) -> Optional[Suggestion]:
        """
        Return a ghost-text suggestion for the current input.

        If the user has typed a complete slash command name (e.g. "/summarize"),
        suggests the first expected argument name as gray text after the cursor.

        Args:
            buffer (Buffer): The current input buffer.
            document (Document): The current document state of the input.

        Returns:
            Optional[Suggestion]: A suggestion containing the argument name, or None.
        """
        text = document.text

        if not text.startswith("/"):
            return None

        parts = text[1:].split()

        if len(parts) == 1:
            cmd = parts[0]

            if cmd in self.prompt_dict:
                prompt = self.prompt_dict[cmd]
                return Suggestion(f" {prompt.arguments[0].name}")

        return None


class UnifiedCompleter(Completer):
    """Provides dropdown completions for slash commands, their arguments, and @mentions."""

    def __init__(self):
        self.prompts: list[Prompt] = []
        self.prompt_dict: dict[str, Prompt] = {}
        self.resources: list[str] = []

    def update_prompts(self, prompts: list[Prompt]) -> None:
        """
        Replace the cached prompts used for slash command completion.

        Args:
            prompts (list[Prompt]): The updated list of MCP prompts.
        """
        self.prompts = prompts
        self.prompt_dict = {prompt.name: prompt for prompt in prompts}

    def update_resources(self, resources: list[str]) -> None:
        """
        Replace the cached resource IDs used for @mention and argument completion.

        Args:
            resources (list[str]): The updated list of resource identifiers.
        """
        self.resources = resources

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        """
        Yield completions based on the current input context.

        Handles three cases:
        - @mentions: suggests matching resource IDs.
        - Slash command names: suggests matching prompt names.
        - Slash command arguments: suggests resource IDs as arguments.

        Args:
            document (Document): The current document state of the input.
            complete_event (CompleteEvent): The completion event from prompt_toolkit.

        Returns:
            Iterable[Completion]: The matching completions for the current input.
        """
        text = document.text
        text_before_cursor = document.text_before_cursor

        if "@" in text_before_cursor:
            last_at_pos = text_before_cursor.rfind("@")
            prefix = text_before_cursor[last_at_pos + 1 :]

            for resource_id in self.resources:
                if resource_id.lower().startswith(prefix.lower()):
                    yield Completion(
                        resource_id,
                        start_position=-len(prefix),
                        display=resource_id,
                        display_meta="Resource",
                    )
            return

        if text.startswith("/"):
            parts = text[1:].split()

            if len(parts) <= 1 and not text.endswith(" "):
                cmd_prefix = parts[0] if parts else ""

                for prompt in self.prompts:
                    if prompt.name.startswith(cmd_prefix):
                        yield Completion(
                            prompt.name,
                            start_position=-len(cmd_prefix),
                            display=f"/{prompt.name}",
                            display_meta=prompt.description or "",
                        )
                return

            if len(parts) == 1 and text.endswith(" "):
                cmd = parts[0]

                if cmd in self.prompt_dict:
                    for id in self.resources:
                        yield Completion(
                            id,
                            start_position=0,
                            display=id,
                        )
                return

            if len(parts) >= 2:
                doc_prefix = parts[-1]

                for resource in self.resources:
                    if "id" in resource and resource["id"].lower().startswith(
                        doc_prefix.lower()
                    ):
                        yield Completion(
                            resource["id"],
                            start_position=-len(doc_prefix),
                            display=resource["id"],
                        )
                return


class CliApp:
    """Interactive terminal application that provides a rich REPL for chatting with Claude via MCP."""

    def __init__(self, agent: CliChat):
        """
        Initialize the CLI application.

        Sets up the prompt_toolkit session with autocompletion, auto-suggestion,
        key bindings, and styling.

        Args:
            agent (CliChat): The CLI chat agent used to process queries and produce responses.
        """
        self.agent = agent
        self.resources = []
        self.prompts = []

        self.completer = UnifiedCompleter()
        self.command_autosuggester = CommandAutoSuggest([])
        self.kb = KeyBindings()

        @self.kb.add("/")
        def _(event):
            buffer = event.app.current_buffer
            if buffer.document.is_cursor_at_the_end and not buffer.text:
                buffer.insert_text("/")
                buffer.start_completion(select_first=False)
            else:
                buffer.insert_text("/")

        @self.kb.add("@")
        def _(event):
            buffer = event.app.current_buffer
            buffer.insert_text("@")
            if buffer.document.is_cursor_at_the_end:
                buffer.start_completion(select_first=False)

        @self.kb.add(" ")
        def _(event):
            buffer = event.app.current_buffer
            text = buffer.text

            buffer.insert_text(" ")

            if text.startswith("/"):
                parts = text[1:].split()

                if len(parts) == 1:
                    buffer.start_completion(select_first=False)
                elif len(parts) == 2:
                    arg = parts[1]
                    if (
                        "doc" in arg.lower()
                        or "file" in arg.lower()
                        or "id" in arg.lower()
                    ):
                        buffer.start_completion(select_first=False)

        self.history = InMemoryHistory()
        self.session = PromptSession(
            completer=self.completer,
            history=self.history,
            key_bindings=self.kb,
            style=Style.from_dict(
                {
                    "prompt": "#aaaaaa",
                    "completion-menu.completion": "bg:#222222 #ffffff",
                    "completion-menu.completion.current": "bg:#444444 #ffffff",
                }
            ),
            complete_while_typing=True,
            complete_in_thread=True,
            auto_suggest=self.command_autosuggester,
        )

    async def initialize(self) -> None:
        """Fetch available resources and prompts from the MCP server and populate the completers."""
        await self.refresh_resources()
        await self.refresh_prompts()

    async def refresh_resources(self) -> None:
        """Fetch resource IDs from the MCP server and update the completer."""
        try:
            self.resources = await self.agent.list_docs_ids()
            self.completer.update_resources(self.resources)
        except Exception as e:
            print(f"Error refreshing resources: {e}")

    async def refresh_prompts(self) -> None:
        """Fetch prompts from the MCP server and update the completer and auto-suggester."""
        try:
            self.prompts = await self.agent.list_prompts()
            self.completer.update_prompts(self.prompts)
            self.command_autosuggester = CommandAutoSuggest(self.prompts)
            self.session.auto_suggest = self.command_autosuggester
        except Exception as e:
            print(f"Error refreshing prompts: {e}")

    async def run(self) -> None:
        """Run the interactive REPL loop, reading user input and printing Claude's responses."""
        while True:
            try:
                user_input = await self.session.prompt_async("> ")
                if not user_input.strip():
                    continue

                if user_input.strip() in (":q", ":quit"):
                    break

                response = await self.agent.run(user_input)
                print(f"\nResponse:\n{response}")

            except KeyboardInterrupt:
                break
