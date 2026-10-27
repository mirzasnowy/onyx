from collections.abc import Iterator
from enum import Enum
from typing import Any, Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
)

from onyx.agents.models import (
    CompactionCheckpoint,
    CompactionMetadata,
)
from onyx.cache.interface import CacheBackend
from onyx.chat.chat_processing_checker import ChatTurnAdmission
from onyx.chat.response_models import ChatResponseSnapshot
from onyx.configs.constants import MessageType
from onyx.context.models import UserMemoryContext
from onyx.context.search.models import SearchDoc
from onyx.db.enums import IncognitoRecordMode
from onyx.file_store.models import (
    ChatLoadedFile,
    ExtractedContextFiles,
    FileDescriptor,
    FileToolMetadata,
)
from onyx.llm.interfaces import LLM, LLMUserIdentity
from onyx.llm.models import (
    Message,
    ReasoningEffort,
)
from onyx.onyxbot.slack.models import SlackContext
from onyx.server.query_and_chat.models import (
    MessageResponseIDInfo,
    MultiModelMessageResponseIDInfo,
    SendMessageRequest,
)
from onyx.server.query_and_chat.streaming_models import (
    CitationInfo,
    Packet,
)
from onyx.tools.file_snapshot import SavedChatFile, SavedContextFiles
from onyx.tools.models import ChatFile, PersonaToolConfiguration, SearchToolUsage
from onyx.tools.results import GeneratedImage
from onyx.tools.tool_implementations.custom.base_tool_types import ToolResultType
from onyx.tools.tool_implementations.search.models import SearchToolState

MAX_DISCOVERED_AGENTS = 128


class PromptMetadata(CompactionMetadata):
    token_count: int | None = None
    image_files: list[ChatLoadedFile] | None = None
    image_token_count: int = 0
    should_cache: bool = False
    file_id: str | None = None
    is_reminder: bool = False
    omit_tool_result_content: bool = False


class FileContextResult(BaseModel):
    """Result of building a file's LLM context representation."""

    message: Message
    tool_metadata: FileToolMetadata


class ChatHistoryMessage(BaseModel):
    id: int
    message_type: MessageType
    message: str
    token_count: int
    files: list[FileDescriptor]
    is_clarification: bool
    response_messages: list[Message]
    checkpoint: CompactionCheckpoint | None = None


class StreamingError(BaseModel):
    error: str
    stack_trace: str | None = None
    error_code: str | None = (
        None  # e.g., "RATE_LIMIT", "AUTH_ERROR", "TOOL_CALL_FAILED"
    )
    is_retryable: bool = True  # Hint to frontend if retry might help
    details: dict | None = None  # Additional context (tool name, model name, etc.)


class CustomToolResponse(BaseModel):
    response: ToolResultType
    tool_name: str


class CreateChatSessionID(BaseModel):
    chat_session_id: UUID
    # Echoes the pinned mode so the client can verify the server honored an
    # incognito request. A server that omits it did not.
    incognito: bool = False


AnswerStreamPart = (
    Packet
    | MessageResponseIDInfo
    | MultiModelMessageResponseIDInfo
    | StreamingError
    | CreateChatSessionID
)

AnswerStream = Iterator[AnswerStreamPart]


class ToolCallResponse(BaseModel):
    """Tool call with full details for non-streaming response."""

    tool_name: str
    tool_arguments: dict[str, Any]
    tool_result: str
    search_docs: list[SearchDoc] | None = None
    generated_images: list[GeneratedImage] | None = None
    # Reasoning that led to the tool call
    pre_reasoning: str | None = None


class ChatBasicResponse(BaseModel):
    # This is built piece by piece, any of these can be None as the flow could break
    answer: str
    answer_citationless: str

    top_documents: list[SearchDoc]

    error_msg: str | None
    message_id: int
    citation_info: list[CitationInfo]


class ChatFullResponse(BaseModel):
    """Complete non-streaming response with all available data.
    NOTE: This model is used for the core flow of the Onyx application, any changes to it should be reviewed and approved by an
    experienced team member. It is very important to 1. avoid bloat and 2. that this remains backwards compatible across versions.
    """

    # Core response fields
    answer: str
    answer_citationless: str
    pre_answer_reasoning: str | None = None
    tool_calls: list[ToolCallResponse] = []

    # Documents & citations
    top_documents: list[SearchDoc]
    citation_info: list[CitationInfo]

    # Metadata
    message_id: int
    chat_session_id: UUID | None = None
    # Echoes the pinned mode for newly-created sessions, like the streaming
    # packet does. A server that omits it did not honor an incognito request.
    incognito: bool = False
    error_msg: str | None = None


class ChatHistoryResult(BaseModel):
    """Result of converting chat history to simple format.

    Bundles the simple messages with metadata for every text file that was
    injected into the history. After context-window truncation drops older
    messages, callers compare surviving ``file_id`` tags against this map
    to discover "forgotten" files whose metadata should be provided to the
    FileReaderTool.
    """

    messages: list[Message]
    all_injected_file_metadata: dict[str, FileToolMetadata]


class PreparedHistory(BaseModel):
    history: ChatHistoryResult
    files: list[ChatFile]


class SearchParams(BaseModel):
    """Resolved search filter IDs and search-tool usage for a chat turn."""

    project_id_filter: int | None
    persona_id_filter: int | None
    search_usage: SearchToolUsage


class PersistenceStatus(str, Enum):
    SAVED = "saved"
    FAILED = "failed"
    UNCONFIRMED = "unconfirmed"


PERSISTENCE_ERROR_MESSAGES = {
    PersistenceStatus.FAILED: "The response could not be saved. Please try again.",
    PersistenceStatus.UNCONFIRMED: "The response save has not completed. Reload this conversation.",
}


class PendingChatResponseSave(BaseModel):
    response: ChatResponseSnapshot
    deadline: float


class ChatResponseOutcome(BaseModel):
    """Frozen execution output and the application's persistence outcome."""

    model_config = ConfigDict(frozen=True)

    response: ChatResponseSnapshot
    persistence_status: PersistenceStatus

    @property
    def error(self) -> str | None:
        errors = [
            self.response.error,
            PERSISTENCE_ERROR_MESSAGES.get(self.persistence_status),
        ]
        return "\n".join(error for error in errors if error) or None


class AvailableFiles(BaseModel):
    """Separated file IDs for the FileReaderTool so it knows which loader to use."""

    # IDs from the ``user_file`` table (project / persona-attached files).
    user_file_ids: list[UUID] = []
    # IDs from the ``file_record`` table (chat-attached files).
    chat_file_ids: list[UUID] = []


class ChatReminderContext(BaseModel):
    ran_image_gen: bool
    has_open_url_tool: bool
    out_of_cycles: bool
    persona_task_prompt: str | None
    has_context_documents: bool


class ContextDocument(BaseModel):
    document: int
    title: str | None = None
    contents: str


class ContextDocuments(BaseModel):
    documents: list[ContextDocument] = Field(default_factory=list)


class SearchPassage(BaseModel):
    model_config = ConfigDict(extra="allow")
    __pydantic_extra__: dict[str, JsonValue] = Field(init=False)
    document: int
    content: str


class SearchPassages(BaseModel):
    model_config = ConfigDict(extra="allow")
    __pydantic_extra__: dict[str, JsonValue] = Field(init=False)
    results: list[SearchPassage]


class ChatPrompt(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    system_prompt: Message | None
    user_info_prompt: Message | None = None
    custom_prompt: Message | None
    reminder: Message | None


class PersonaPromptConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    system_prompt: str | None
    task_prompt: str | None
    datetime_aware: bool
    replace_base_system_prompt: bool


class ChatFeatureState(BaseModel):
    """Chat settings and accumulated state needed to resume suspended execution."""

    kind: Literal["chat.state.v1"] = "chat.state.v1"
    persona: PersonaPromptConfig | None
    context_files: SavedContextFiles
    file_metadata: dict[str, FileToolMetadata] | None
    memory: UserMemoryContext | None
    reasoning_effort: ReasoningEffort
    include_citations: bool
    inject_memories: bool
    forced_tool_id: int | None
    base_prompt: str
    custom_prompt: str | None
    reminders_enabled: bool

    elapsed_seconds: float
    citation_sources: dict[int, SearchDoc]
    citation_mapping: dict[int, str]
    gathered_documents: list[SearchDoc]
    chat_files: list[SavedChatFile]
    has_called_search_tool: bool
    ran_image_gen: bool
    search_tools: dict[str, SearchToolState]


class ReservedChatResponse(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    llm: LLM
    message_id: int
    display_name: str


class ChatTurnSetup(BaseModel):
    """Request values and service references shared by model executions."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    new_msg_req: SendMessageRequest
    chat_session_id: UUID
    chat_session_project_id: int | None
    # The session's pinned recording policy. None is an ordinary chat.
    incognito_record_mode: IncognitoRecordMode | None
    persona_id: int
    persona: PersonaPromptConfig
    base_system_prompt: str
    tool_configuration: PersonaToolConfiguration
    research_tool_id: int | None
    checkpoint: CompactionCheckpoint | None
    user_message_id: int
    user_identity: LLMUserIdentity
    responses: list[ReservedChatResponse]
    messages: list[Message]
    input_messages: list[Message]
    previous_run_id: str | None = None
    extracted_context_files: ExtractedContextFiles
    # Fences processing status and identifies the buffered stream.
    stream_id: int
    reasoning_effort: ReasoningEffort
    search_params: SearchParams
    all_injected_file_metadata: dict[str, FileToolMetadata]
    available_files: AvailableFiles
    forced_tool_id: int | None
    chat_files_for_tools: list[ChatFile]
    custom_agent_prompt: str | None
    user_memory_context: UserMemoryContext
    # For deep research: was the last assistant message a clarification request?
    skip_clarification: bool
    cache: CacheBackend
    admission: ChatTurnAdmission
    # Execution params forwarded to per-model tool construction
    slack_context: SlackContext | None
    litellm_additional_headers: dict[str, str] | None = None
    custom_tool_additional_headers: dict[str, str] | None
    mcp_headers: dict[str, str] | None
