from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, Field

ToolName = Literal[
    "list_files",
    "read_file",
    "search_code",
    "edit_file",
    "run_command",
    "run_tests",
    "run_lint",
    "run_build",
    "security_scan",
]


class ToolCall(BaseModel):
    tool: ToolName
    args: dict[str, Any] = Field(default_factory=dict)
    step: int = 0
    run_id: str = ""


class ToolObservation(BaseModel):
    tool: ToolName
    result: Any
    error: str | None = None
    step: int = 0
    run_id: str = ""


class ListFilesArgs(BaseModel):
    dir: str = "."
    recursive: bool = False


class ReadFileArgs(BaseModel):
    path: str
    start_line: int | None = None
    limit: int | None = None


class SearchCodeArgs(BaseModel):
    query: str
    glob: str | None = None
    semantic: bool = False
    timeout: int | None = 30


class EditFileArgs(BaseModel):
    path: str
    old_string: str
    new_string: str


class RunCommandArgs(BaseModel):
    cmd: str
    cwd: str | None = None
    timeout: int | None = 120


class ToolProtocol(ABC):
    @abstractmethod
    def list_files(self, args: ListFilesArgs) -> Any: ...

    @abstractmethod
    def read_file(self, args: ReadFileArgs) -> Any: ...

    @abstractmethod
    def search_code(self, args: SearchCodeArgs) -> Any: ...

    @abstractmethod
    def edit_file(self, args: EditFileArgs) -> Any: ...

    @abstractmethod
    def run_command(self, args: RunCommandArgs) -> Any: ...

    @abstractmethod
    def run_tests(self) -> Any: ...

    @abstractmethod
    def run_lint(self) -> Any: ...

    @abstractmethod
    def run_build(self) -> Any: ...

    @abstractmethod
    def security_scan(self, timeout_s: float | None = None) -> Any: ...

    def execute(self, call: ToolCall) -> ToolObservation:
        try:
            if call.tool == "list_files":
                res = self.list_files(ListFilesArgs(**call.args))
            elif call.tool == "read_file":
                res = self.read_file(ReadFileArgs(**call.args))
            elif call.tool == "search_code":
                res = self.search_code(SearchCodeArgs(**call.args))
            elif call.tool == "edit_file":
                res = self.edit_file(EditFileArgs(**call.args))
            elif call.tool == "run_command":
                res = self.run_command(RunCommandArgs(**call.args))
            elif call.tool == "run_tests":
                res = self.run_tests()
            elif call.tool == "run_lint":
                res = self.run_lint()
            elif call.tool == "run_build":
                res = self.run_build()
            elif call.tool == "security_scan":
                res = self.security_scan()
            else:
                raise ValueError(f"unknown tool {call.tool}")
            return ToolObservation(tool=call.tool, result=res, step=call.step, run_id=call.run_id)
        except Exception as e:
            return ToolObservation(
                tool=call.tool, result=None, error=str(e), step=call.step, run_id=call.run_id
            )


def get_openai_tool_schemas() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List files in a directory",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dir": {"type": "string", "default": "."},
                        "recursive": {"type": "boolean", "default": False},
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read file contents, optionally with line range",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "start_line": {"type": "integer"},
                        "limit": {"type": "integer"},
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_code",
                "description": "Search code with ripgrep or semantic (future)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "glob": {"type": "string"},
                        "semantic": {"type": "boolean", "default": False},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": "Edit file with exact string match, returns unified diff",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "old_string": {"type": "string"},
                        "new_string": {"type": "string"},
                    },
                    "required": ["path", "old_string", "new_string"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_command",
                "description": "Run shell command, capture stdout/stderr/exit",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cmd": {"type": "string"},
                        "cwd": {"type": "string"},
                        "timeout": {"type": "integer", "default": 120},
                    },
                    "required": ["cmd"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_tests",
                "description": "Run task-specific tests via verifier or script",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]
