# MCP Chat Reference Example

A Claude-powered chat CLI app that integrates with [Model Context Protocol (MCP)](https://modelcontextprotocol.io) servers.

## Description

**Model Context Protocol (MCP)** is an open standard that lets AI models interact with external tools, data sources, and services in a composable way. Instead of hardcoding integrations, you define MCP servers that expose:

- **Tools**: functions Claude can call
- **Resources**: data Claude can read
- **Prompts**: reusable prompt templates

Any MCP-compatible client can then use them.

This project demonstrates a complete end-to-end integration:

- An **MCP server** built with [FastMCP](https://github.com/PrefectHQ/fastmcp) that manages a set of mock documents, exposing them as tools, resources, and prompt templates.
- An **MCP client** that connects to one or more servers, discovers their capabilities, and exposes them as tools to Claude.
- A **Claude-powered chat CLI** with an agentic tool-use loop, document @mentions, and slash commands, all driven through a terminal UI with tab completion.

## Architecture

```
+---------+     +-----------+     +------------+
|   CLI   | --> | Chat Loop | <-> | Claude API |
+---------+     +-----------+     +------------+
                      |
                      v
               +------------+     +------------+
               | MCP Client | --> | MCP Server |
               +------------+     +------------+
                                  tools, resources, prompts
```

You type a message in the **CLI**, which parses @mentions and slash commands and enriches the prompt with document context. The **Chat Loop** sends the prompt to **Claude**, which may request tool calls. The **MCP Client** forwards those calls to the **MCP Server** and returns the results. This loop repeats until Claude produces a final response.

## Features

- **Agentic tool-use loop**: Claude autonomously calls MCP tools (read/edit documents) until it has enough information to respond
- **@mentions**: reference any document inline (e.g. `compare @report.pdf with @financials.docx`) and its content is automatically fetched and injected into the prompt
- **Slash commands**: invoke server-side prompt templates directly from the CLI (`/format`, `/summarize`)
- **Tab completion**: autocomplete for slash commands, document IDs, and @mentions
- **Multi-server support**: connect to multiple MCP servers simultaneously; Claude sees all their tools unified
- **MCP primitives**: the server demonstrates all three MCP primitives: tools, resources, and prompts

## Prerequisites

- Python 3.13+
- [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`
- An [Anthropic API key](https://console.anthropic.com/)

## Setup

1. **Install dependencies**

   Using `uv` (recommended):
   ```bash
   uv sync
   ```

   Or with `pip`:
   ```bash
   pip install -e .
   ```

2. **Configure environment variables**

   Create a `.env` file in the project root:

   ```bash
   ANTHROPIC_API_KEY=sk-ant-...       # Your Anthropic API key
   CLAUDE_MODEL=claude-sonnet-4-6     # Claude model to use
   USE_UV=1                           # 1 if using uv, 0 if using python
   ```

3. **Run the application**

   ```bash
   uv run main.py
   # or
   python main.py
   ```

## Usage

### Regular chat

Just type your message and press Enter. Claude has access to tools for reading and editing the example mock documents bundled with the server.

```
> What documents are available?
> Read the deposition and give me a brief summary.
> Edit report.pdf to fix the typo in the second paragraph.
```

### @mentions

Prefix any document ID with `@` to automatically fetch its content and include it in your prompt, without requiring a tool call.

```
> What are the key differences between @financials.docx and @outlook.pdf?
> Does @deposition.md contradict anything in @report.pdf?
```

Tab completion is available after typing `@`.

### Slash commands

Slash commands invoke server-side prompt templates. Tab completion is available for both the command and the document ID.

```
> /summarize report.pdf
> /format spec.txt
```

| Command | Description |
|---|---|
| `/summarize <doc_id>` | Summarize a document |
| `/format <doc_id>` | Reformat a document as Markdown |

### Connecting additional MCP servers

Pass additional server scripts as arguments to `main.py` to connect to them alongside the built-in document server:

```bash
uv run main.py path/to/other_server.py
# or
python main.py path/to/other_server.py
```

All servers' tools are aggregated and made available to Claude automatically.

### Exiting

```
> :q
> :quit
```

## Useful Commands

```bash
# Run the chat app
uv run main.py
# or
python main.py

# Inspect the MCP server interactively with the MCP developer tools
uv run mcp dev server/mcp_server.py

# Run all tests
pytest

# Run tests with verbose output
pytest -v
```

## Project Structure

```
+---------------------+     +-----------------------------+     +----------------+
| CLI                 | --> | Chat Loop                   | <-> | Claude API     |
| core/cli.py         |     | core/chat.py                |     | core/claude.py |
| core/cli_chat.py    |     | core/tools.py               |     +----------------+
+---------------------+     +-----------------------------+
                                        |
                                        v
                             +------------------+     +------------------------+
                             | MCP Client       | --> | MCP Server             |
                             | client/          |     | server/mcp_server.py   |
                             | mcp_client.py    |     +------------------------+
                             +------------------+

                             main.py wires all of the above together
```

| Category | File | Description |
|---|---|---|
| Entry Point | `main.py` | Wires all services together and starts the CLI |
| MCP Server | `server/mcp_server.py` | Mock document tools, resources, and prompt templates |
| MCP Client | `client/mcp_client.py` | Async MCP client wrapper around the MCP SDK |
| Chat | `core/chat.py` | Agentic tool-use loop |
| Chat | `core/tools.py` | Tool aggregation and execution across multiple MCP clients |
| Chat | `core/claude.py` | Anthropic SDK wrapper with message history |
| CLI | `core/cli.py` | Interactive REPL with tab completion |
| CLI | `core/cli_chat.py` | @mention and slash command handling |
| Tests | `tests/` | Unit tests for all core modules and the MCP client |
