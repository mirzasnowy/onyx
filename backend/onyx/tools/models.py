from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from onyx.configs.constants import MessageType
from onyx.context.models import UserMemoryContext
from onyx.context.search.models import PersonaSearchInfo
from onyx.file_store.models import (
    install_lazy_content_loader,
    maybe_materialize_lazy_content,
)
from onyx.tools.constants import FILE_READER_MAX_CHARS
from onyx.utils.headers import HeaderItemDict


class FileReaderArguments(BaseModel):
    file_id: str
    start_char: int = 0
    num_chars: int = FILE_READER_MAX_CHARS


class MemoryArguments(BaseModel):
    memory: str


class ToolConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)

    id: int
    name: str
    description: str | None
    display_name: str | None
    in_code_tool_id: str | None
    enabled: bool
    openapi_schema: dict[str, JsonValue] | None
    mcp_input_schema: dict[str, JsonValue] | None
    custom_headers: list[HeaderItemDict] | None
    passthrough_auth: bool
    mcp_server_id: int | None
    oauth_config_id: int | None


class PersonaToolConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True)

    persona_id: int
    persona_name: str
    tools: list[ToolConfiguration]
    search: PersonaSearchInfo


class ToolCallException(Exception):
    """Exception raised for errors during tool calls."""

    def __init__(self, message: str, llm_facing_message: str):
        # This is the full error message which is used for tracing
        super().__init__(message)
        # LLM made tool calls are acceptable and not flow terminating, this is the message
        # which will populate the tool response.
        self.llm_facing_message = llm_facing_message


class ToolExecutionException(Exception):
    """Tool execution failed."""

    def __init__(self, message: str, emit_error_packet: bool = False) -> None:
        super().__init__(message)
        self.emit_error_packet = emit_error_packet


class SearchToolUsage(str, Enum):
    DISABLED = "disabled"
    ENABLED = "enabled"
    AUTO = "auto"


class ChatMinimalTextMessage(BaseModel):
    message: str
    message_type: MessageType


class DynamicSchemaInfo(BaseModel):
    chat_session_id: UUID | None
    message_id: int | None
    user_id: UUID | None = None
    user_email: str | None = None


class ChatFile(BaseModel):
    """File from a chat session that can be passed to tools."""

    filename: str
    content: bytes
    file_id: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def lazy_from_filename(
        cls,
        *,
        filename: str,
        loader: Callable[[], bytes],
        file_id: str | None = None,
    ) -> "ChatFile":
        """Construct a ChatFile whose ``content`` is loaded on first access.

        Existing eager construction (``ChatFile(filename=..., content=...)``)
        is unchanged. PythonTool's ``.content`` access transparently triggers
        the loader and memoizes the result.
        """
        inst = cls(filename=filename, content=b"", file_id=file_id)
        install_lazy_content_loader(inst, loader)
        return inst

    def __getattribute__(self, name: str):
        if name == "content":
            maybe_materialize_lazy_content(self)
        return object.__getattribute__(self, name)


class ToolContext(BaseModel):
    """Application data available to each tool in an agent step."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    user_memory_context: UserMemoryContext | None = None
    user_info: str | None = None
    citation_mapping: dict[int, str] = Field(default_factory=dict)
    next_citation_num: int = 1
    # On repeat searches, defer to the model's new queries instead of repeating
    # the expansion flow that may have produced poor results on the first pass.
    skip_search_query_expansion: bool = False
    chat_files: list[ChatFile] = Field(default_factory=list)
    url_snippet_map: dict[str, str] = Field(default_factory=dict)
    # When False, don't pass memory context to search tools for query expansion
    # (but still pass it to the memory tool for persistence)
    inject_memories_in_prompt: bool = True


CHAT_SESSION_ID_PLACEHOLDER = "CHAT_SESSION_ID"
MESSAGE_ID_PLACEHOLDER = "MESSAGE_ID"
USER_ID_PLACEHOLDER = "USER_ID"
USER_EMAIL_PLACEHOLDER = "USER_EMAIL"


class BaseCiteableToolResult(BaseModel):
    """Base class for tool results that can be cited."""

    document_citation_number: int
    unique_identifier_to_strip_away: str | None = None
    type: str


class LlmInternalSearchResult(BaseCiteableToolResult):
    """Result from an internal search query"""

    type: Literal["internal_search"] = "internal_search"
    title: str
    excerpt: str
    metadata: dict[str, Any]


class LlmWebSearchResult(BaseCiteableToolResult):
    """Result from a web search query"""

    type: Literal["web_search"] = "web_search"
    url: str
    title: str
    snippet: str


class LlmOpenUrlResult(BaseCiteableToolResult):
    """Result from opening/fetching a URL"""

    type: Literal["open_url"] = "open_url"
    content: str


class PythonArguments(BaseModel):
    code: str


class BashArguments(BaseModel):
    cmd: str
