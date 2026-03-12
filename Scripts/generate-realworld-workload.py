#!/usr/bin/env python3
"""Generate realistic multi-turn tool call workload based on actual OpenCode patterns.

Creates request JSON files that mirror real OpenCode→afm conversations:
- OpenCode's exact 11 tool schemas (bash requires 'description', etc.)
- Growing multi-turn conversations (12K→20K+ tokens, 52KB→77KB body)
- Large file content in tool results (up to 10KB per result)
- Code with < operators, triple quotes, escaped strings
- Edit operations with multiline oldString/newString
- The actual "File Organizer PRD" task from OpenCode test runs

Data source: ~/.local/share/opencode/opencode.db + test-reports/opencode-tooling-*
"""

import json
import os
import sys


def generate_workload(output_dir, num_requests=8):
    os.makedirs(output_dir, exist_ok=True)

    # ── OpenCode's system prompt (abbreviated but structurally accurate) ────────
    system_prompt = (
        "You are opencode, an interactive CLI tool that helps users with "
        "software engineering tasks. You have access to tools for reading, "
        "writing, and editing files, running shell commands, searching code, "
        "and managing tasks.\n\n"
        "## Core Principles\n"
        "- Always read files before editing them\n"
        "- Use the edit tool for targeted changes, write for new files or "
        "complete rewrites\n"
        "- Provide a description for every bash command explaining what it does\n"
        "- Search with grep/glob before making assumptions about code structure\n"
        "- Think step by step and be thorough\n"
        "- Handle errors gracefully and suggest fixes\n\n"
        "## Tool Usage Guidelines\n"
        "- bash: ALWAYS include a description parameter explaining what the "
        "command does\n"
        "- edit: The oldString must match EXACTLY including whitespace and "
        "indentation\n"
        "- write: Use for new files or complete file rewrites only\n"
        "- read: Read files before modifying them to understand context\n"
        "- glob: Use to find files by pattern before reading/editing\n"
        "- grep: Use to search for specific patterns in code\n\n"
        "## Project Conventions\n"
        "- Follow PEP 8 for Python code\n"
        "- Use type hints for function signatures\n"
        "- Include docstrings for all public functions and classes\n"
        "- Write tests alongside implementation code\n"
        "- Use pathlib for file path operations"
    )

    # ── OpenCode's exact 11 tool schemas (from SQLite DB extraction) ───────────
    tools = [
        {"type": "function", "function": {
            "name": "bash",
            "description": "Execute a shell command and return stdout/stderr. "
                           "IMPORTANT: You MUST include a 'description' parameter.",
            "parameters": {"type": "object", "properties": {
                "command": {"type": "string",
                            "description": "The shell command to execute"},
                "description": {"type": "string",
                                "description": "A brief description of what "
                                               "this command does"},
                "timeout": {"type": "number",
                            "description": "Max execution time in seconds "
                                           "(default: 30)"},
                "workdir": {"type": "string",
                            "description": "Working directory for the command"}
            }, "required": ["command", "description"]}}},
        {"type": "function", "function": {
            "name": "read",
            "description": "Read the contents of a file at the given path.",
            "parameters": {"type": "object", "properties": {
                "filePath": {"type": "string",
                             "description": "Absolute or relative path to the "
                                            "file"},
                "limit": {"type": "number", "description": "Max lines to read"},
                "offset": {"type": "number",
                            "description": "Starting line number (0-based)"}
            }, "required": ["filePath"]}}},
        {"type": "function", "function": {
            "name": "write",
            "description": "Write content to a file, creating directories as "
                           "needed.",
            "parameters": {"type": "object", "properties": {
                "filePath": {"type": "string",
                             "description": "Path to write the file"},
                "content": {"type": "string",
                            "description": "The full content to write"}
            }, "required": ["filePath", "content"]}}},
        {"type": "function", "function": {
            "name": "edit",
            "description": "Apply a targeted edit by replacing an exact string "
                           "match.",
            "parameters": {"type": "object", "properties": {
                "filePath": {"type": "string",
                             "description": "Path to the file to edit"},
                "oldString": {"type": "string",
                              "description": "Exact string to find (must match "
                                             "precisely including whitespace)"},
                "newString": {"type": "string",
                              "description": "Replacement string"},
                "replaceAll": {"type": "boolean",
                               "description": "Replace all occurrences "
                                              "(default: false)"}
            }, "required": ["filePath", "oldString", "newString"]}}},
        {"type": "function", "function": {
            "name": "glob",
            "description": "Find files matching a glob pattern.",
            "parameters": {"type": "object", "properties": {
                "pattern": {"type": "string",
                            "description": "Glob pattern (e.g. '**/*.py')"},
                "path": {"type": "string",
                         "description": "Base directory to search from"}
            }, "required": ["pattern"]}}},
        {"type": "function", "function": {
            "name": "grep",
            "description": "Search for a regex pattern in files.",
            "parameters": {"type": "object", "properties": {
                "pattern": {"type": "string",
                            "description": "Regular expression pattern"},
                "path": {"type": "string",
                         "description": "File or directory to search"},
                "include": {"type": "string",
                            "description": "Glob to filter files (e.g. "
                                           "'*.py')"}
            }, "required": ["pattern"]}}},
        {"type": "function", "function": {
            "name": "todowrite",
            "description": "Create or update a todo list for tracking work.",
            "parameters": {"type": "object", "properties": {
                "todos": {"type": "array", "items": {"type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "status": {"type": "string",
                                   "enum": ["pending", "in_progress",
                                            "completed"]},
                        "priority": {"type": "string",
                                     "enum": ["high", "medium", "low"]}
                    }, "required": ["content", "status", "priority"]}}
            }, "required": ["todos"]}}},
        {"type": "function", "function": {
            "name": "task",
            "description": "Launch a background task or agent.",
            "parameters": {"type": "object", "properties": {
                "description": {"type": "string",
                                "description": "Short task description"},
                "prompt": {"type": "string",
                           "description": "Detailed prompt for the agent"},
                "subagent_type": {"type": "string",
                                  "description": "Type of agent to use"},
                "command": {"type": "string"},
                "task_id": {"type": "string"}
            }, "required": ["description", "prompt", "subagent_type"]}}},
        {"type": "function", "function": {
            "name": "webfetch",
            "description": "Fetch content from a URL.",
            "parameters": {"type": "object", "properties": {
                "url": {"type": "string", "description": "The URL to fetch"},
                "format": {"type": "string",
                           "enum": ["text", "markdown", "html"],
                           "description": "Output format"},
                "timeout": {"type": "number"}
            }, "required": ["url", "format"]}}},
        {"type": "function", "function": {
            "name": "question",
            "description": "Ask the user a question with options.",
            "parameters": {"type": "object", "properties": {
                "questions": {"type": "array", "items": {"type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "header": {"type": "string"},
                        "options": {"type": "array", "items": {
                            "type": "object", "properties": {
                                "label": {"type": "string"},
                                "description": {"type": "string"}
                            }, "required": ["label", "description"]}},
                        "multiple": {"type": "boolean"}
                    }, "required": ["question", "header", "options"]}}
            }, "required": ["questions"]}}},
        {"type": "function", "function": {
            "name": "skill",
            "description": "Invoke a named skill.",
            "parameters": {"type": "object", "properties": {
                "name": {"type": "string",
                         "description": "Name of the skill to invoke"}
            }, "required": ["name"]}}}
    ]

    # ── Realistic file content (exercises real failure patterns) ────────────────
    # These are based on actual files from OpenCode test sessions.

    main_py = (
        '"""Main entry point for the file organizer CLI."""\n'
        '\n'
        'import argparse\n'
        'import sys\n'
        'from pathlib import Path\n'
        'from typing import Optional\n'
        '\n'
        'from .organizer import FileOrganizer\n'
        'from .file_types import FILE_CATEGORIES, get_category\n'
        '\n'
        '\n'
        'def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:\n'
        '    """Parse command-line arguments.\n'
        '\n'
        '    Args:\n'
        '        argv: Optional argument list (defaults to sys.argv[1:])\n'
        '\n'
        '    Returns:\n'
        '        Parsed namespace with organize/undo subcommands\n'
        '    """\n'
        '    parser = argparse.ArgumentParser(\n'
        '        prog="file-organizer",\n'
        '        description="Organize files into categorized directories"\n'
        '    )\n'
        '    subparsers = parser.add_subparsers(dest="command", required=True)\n'
        '\n'
        '    # organize subcommand\n'
        '    org = subparsers.add_parser("organize", help="Organize files")\n'
        '    org.add_argument("directory", type=Path, help="Directory to organize")\n'
        '    org.add_argument("--dry-run", action="store_true",\n'
        '                     help="Preview without moving")\n'
        '    org.add_argument("--recursive", "-r", action="store_true",\n'
        '                     help="Include subdirectories")\n'
        '    org.add_argument("--max-depth", type=int, default=3,\n'
        '                     help="Maximum recursion depth")\n'
        '    org.add_argument("--exclude", nargs="*", default=[],\n'
        '                     help="Patterns to exclude")\n'
        '\n'
        '    # undo subcommand\n'
        '    undo = subparsers.add_parser("undo", help="Undo last organization")\n'
        '    undo.add_argument("--log-file", type=Path, help="Path to undo log")\n'
        '\n'
        '    return parser.parse_args(argv)\n'
        '\n'
        '\n'
        'def validate_directory(path: Path) -> bool:\n'
        '    """Validate that the target directory exists and is accessible.\n'
        '\n'
        '    Checks:\n'
        '    1. Path exists\n'
        '    2. Path is a directory (not a file or symlink to file)\n'
        '    3. Path is readable and writable\n'
        '    4. Path is not a system directory (/, /usr, /etc, etc.)\n'
        '    """\n'
        '    if not path.exists():\n'
        '        print(f"Error: Directory does not exist: {path}", file=sys.stderr)\n'
        '        return False\n'
        '    if not path.is_dir():\n'
        '        print(f"Error: Not a directory: {path}", file=sys.stderr)\n'
        '        return False\n'
        '\n'
        '    # Safety check: prevent organizing system directories\n'
        '    system_dirs = [Path("/"), Path("/usr"), Path("/etc"), Path("/var")]\n'
        '    resolved = path.resolve()\n'
        '    if resolved in system_dirs:\n'
        '        print(f"Error: Cannot organize system directory: {resolved}",\n'
        '              file=sys.stderr)\n'
        '        return False\n'
        '\n'
        '    # Check read/write permissions\n'
        '    if not os.access(path, os.R_OK | os.W_OK):\n'
        '        print(f"Error: Insufficient permissions for: {path}",\n'
        '              file=sys.stderr)\n'
        '        return False\n'
        '\n'
        '    return True\n'
        '\n'
        '\n'
        'def count_files(directory: Path, recursive: bool = False,\n'
        '                max_depth: int = 3) -> dict[str, int]:\n'
        '    """Count files by category in the target directory.\n'
        '\n'
        '    Returns a dict mapping category names to file counts.\n'
        '    Files with unknown extensions are counted under "Other".\n'
        '    """\n'
        '    counts: dict[str, int] = {}\n'
        '\n'
        '    if recursive:\n'
        '        files = _walk_with_depth(directory, max_depth)\n'
        '    else:\n'
        '        files = [f for f in directory.iterdir() if f.is_file()]\n'
        '\n'
        '    for file_path in files:\n'
        '        category = get_category(file_path.suffix)\n'
        '        counts[category] = counts.get(category, 0) + 1\n'
        '\n'
        '    return counts\n'
        '\n'
        '\n'
        'def _walk_with_depth(directory: Path, max_depth: int,\n'
        '                     current_depth: int = 0) -> list[Path]:\n'
        '    """Walk directory tree with depth limit."""\n'
        '    if current_depth > max_depth:\n'
        '        return []\n'
        '\n'
        '    files = []\n'
        '    try:\n'
        '        for entry in directory.iterdir():\n'
        '            if entry.is_file():\n'
        '                files.append(entry)\n'
        '            elif entry.is_dir() and not entry.name.startswith("."):\n'
        '                files.extend(\n'
        '                    _walk_with_depth(entry, max_depth, current_depth + 1)\n'
        '                )\n'
        '    except PermissionError:\n'
        '        print(f"Warning: Permission denied: {directory}",\n'
        '              file=sys.stderr)\n'
        '\n'
        '    return files\n'
        '\n'
        '\n'
        'def print_report(actions: list[tuple[Path, Path]]) -> None:\n'
        '    """Print a formatted report of file organization actions."""\n'
        '    if not actions:\n'
        '        print("No files to organize.")\n'
        '        return\n'
        '\n'
        '    by_category: dict[str, list[tuple[Path, Path]]] = {}\n'
        '    for src, dst in actions:\n'
        '        cat = dst.parent.name\n'
        '        by_category.setdefault(cat, []).append((src, dst))\n'
        '\n'
        '    total = len(actions)\n'
        '    print(f"\\nOrganization plan: {total} file(s)")\n'
        '    print("=" * 60)\n'
        '\n'
        '    for category in sorted(by_category.keys()):\n'
        '        items = by_category[category]\n'
        '        print(f"\\n  {category}/ ({len(items)} files)")\n'
        '        for src, dst in sorted(items, key=lambda x: x[0].name):\n'
        '            size = src.stat().st_size\n'
        '            if size < 1024:\n'
        '                size_str = f"{size}B"\n'
        '            elif size < 1024 * 1024:\n'
        '                size_str = f"{size / 1024:.1f}KB"\n'
        '            else:\n'
        '                size_str = f"{size / (1024*1024):.1f}MB"\n'
        '            print(f"    {src.name} ({size_str}) -> {dst}")\n'
        '\n'
        '\n'
        'def main(argv: Optional[list[str]] = None) -> int:\n'
        '    """Main entry point."""\n'
        '    args = parse_args(argv)\n'
        '\n'
        '    if args.command == "organize":\n'
        '        if not validate_directory(args.directory):\n'
        '            return 1\n'
        '\n'
        '        organizer = FileOrganizer(\n'
        '            target_dir=args.directory,\n'
        '            recursive=args.recursive,\n'
        '            max_depth=args.max_depth,\n'
        '            exclude_patterns=args.exclude\n'
        '        )\n'
        '\n'
        '        actions = organizer.plan()\n'
        '\n'
        '        if args.dry_run:\n'
        '            print("Preview (dry run):")\n'
        '            print_report(actions)\n'
        '            return 0\n'
        '\n'
        '        moved = organizer.execute(actions)\n'
        '        print(f"Organized {moved} file(s)")\n'
        '        return 0\n'
        '\n'
        '    elif args.command == "undo":\n'
        '        print("Undo not yet implemented", file=sys.stderr)\n'
        '        return 1\n'
        '\n'
        '    return 0\n'
        '\n'
        '\n'
        'if __name__ == "__main__":\n'
        '    sys.exit(main())\n'
    )

    file_types_py = (
        '"""File type mappings for the file organizer."""\n'
        '\n'
        '# Mapping of file extensions to category names.\n'
        'FILE_CATEGORIES: dict[str, str] = {\n'
        '    # Images\n'
        '    ".jpg": "Images",\n'
        '    ".jpeg": "Images",\n'
        '    ".png": "Images",\n'
        '    ".gif": "Images",\n'
        '    ".bmp": "Images",\n'
        '    ".svg": "Images",\n'
        '    ".ico": "Images",\n'
        '    ".webp": "Images",\n'
        '    ".tiff": "Images",\n'
        '    ".raw": "Images",\n'
        '    ".heic": "Images",\n'
        '    # Documents\n'
        '    ".pdf": "Documents",\n'
        '    ".doc": "Documents",\n'
        '    ".docx": "Documents",\n'
        '    ".xls": "Documents",\n'
        '    ".xlsx": "Documents",\n'
        '    ".ppt": "Documents",\n'
        '    ".pptx": "Documents",\n'
        '    ".odt": "Documents",\n'
        '    ".ods": "Documents",\n'
        '    ".txt": "Documents",\n'
        '    ".rtf": "Documents",\n'
        '    ".csv": "Documents",\n'
        '    # Code\n'
        '    ".py": "Code",\n'
        '    ".js": "Code",\n'
        '    ".ts": "Code",\n'
        '    ".jsx": "Code",\n'
        '    ".tsx": "Code",\n'
        '    ".swift": "Code",\n'
        '    ".rs": "Code",\n'
        '    ".go": "Code",\n'
        '    ".java": "Code",\n'
        '    ".c": "Code",\n'
        '    ".cpp": "Code",\n'
        '    ".h": "Code",\n'
        '    ".rb": "Code",\n'
        '    ".php": "Code",\n'
        '    ".sh": "Code",\n'
        '    # Data\n'
        '    ".json": "Data",\n'
        '    ".yaml": "Data",\n'
        '    ".yml": "Data",\n'
        '    ".xml": "Data",\n'
        '    ".toml": "Data",\n'
        '    ".ini": "Data",\n'
        '    ".env": "Data",\n'
        '    # Media\n'
        '    ".mp3": "Media",\n'
        '    ".wav": "Media",\n'
        '    ".flac": "Media",\n'
        '    ".mp4": "Media",\n'
        '    ".avi": "Media",\n'
        '    ".mkv": "Media",\n'
        '    ".mov": "Media",\n'
        '    ".webm": "Media",\n'
        '    # Archives\n'
        '    ".zip": "Archives",\n'
        '    ".tar": "Archives",\n'
        '    ".gz": "Archives",\n'
        '    ".7z": "Archives",\n'
        '    ".rar": "Archives",\n'
        '    ".dmg": "Archives",\n'
        '}\n'
        '\n'
        '\n'
        'def get_category(extension: str) -> str:\n'
        '    """Get the category for a file extension."""\n'
        '    return FILE_CATEGORIES.get(extension.lower(), "Other")\n'
        '\n'
        '\n'
        'def get_all_categories() -> list[str]:\n'
        '    """Get sorted list of all known category names."""\n'
        '    return sorted(set(FILE_CATEGORIES.values()))\n'
        '\n'
        '\n'
        'def get_extensions_for_category(category: str) -> list[str]:\n'
        '    """Get all extensions that map to a given category."""\n'
        '    return sorted(\n'
        '        ext for ext, cat in FILE_CATEGORIES.items() if cat == category\n'
        '    )\n'
    )

    organizer_py = (
        '"""Core organizer logic for the file organizer."""\n'
        '\n'
        'import json\n'
        'import shutil\n'
        'from datetime import datetime\n'
        'from pathlib import Path\n'
        'from typing import Optional\n'
        '\n'
        'from .file_types import get_category\n'
        '\n'
        '\n'
        'class FileOrganizer:\n'
        '    """Organizes files by moving them into categorized subdirectories.\n'
        '\n'
        '    The organizer scans a target directory, categorizes each file by its\n'
        '    extension, and moves files into subdirectories named after their\n'
        '    category.\n'
        '    """\n'
        '\n'
        '    def __init__(\n'
        '        self,\n'
        '        target_dir: Path,\n'
        '        recursive: bool = False,\n'
        '        max_depth: int = 3,\n'
        '        exclude_patterns: Optional[list[str]] = None\n'
        '    ):\n'
        '        self.target_dir = target_dir.resolve()\n'
        '        self.recursive = recursive\n'
        '        self.max_depth = max_depth\n'
        '        self.exclude_patterns = exclude_patterns or []\n'
        '        self._undo_log: list[dict] = []\n'
        '\n'
        '    def plan(self) -> list[tuple[Path, Path]]:\n'
        '        """Plan file organization without moving anything."""\n'
        '        actions = []\n'
        '        files = self._collect_files()\n'
        '\n'
        '        for file_path in files:\n'
        '            if self._should_exclude(file_path):\n'
        '                continue\n'
        '\n'
        '            category = get_category(file_path.suffix)\n'
        '            if category == "Other":\n'
        '                continue  # Skip uncategorized files\n'
        '\n'
        '            dest_dir = self.target_dir / category\n'
        '            dest_path = dest_dir / file_path.name\n'
        '\n'
        '            # Handle name collisions\n'
        '            if dest_path.exists() and dest_path != file_path:\n'
        '                dest_path = self._resolve_collision(dest_path)\n'
        '\n'
        '            # Don\'t move if already in correct directory\n'
        '            if file_path.parent == dest_dir:\n'
        '                continue\n'
        '\n'
        '            actions.append((file_path, dest_path))\n'
        '\n'
        '        return actions\n'
        '\n'
        '    def execute(self, actions: list[tuple[Path, Path]]) -> int:\n'
        '        """Execute planned file moves."""\n'
        '        moved = 0\n'
        '        self._undo_log = []\n'
        '\n'
        '        for src, dst in actions:\n'
        '            try:\n'
        '                dst.parent.mkdir(parents=True, exist_ok=True)\n'
        '                shutil.move(str(src), str(dst))\n'
        '                self._undo_log.append({\n'
        '                    "action": "move",\n'
        '                    "source": str(src),\n'
        '                    "destination": str(dst),\n'
        '                    "timestamp": datetime.now().isoformat()\n'
        '                })\n'
        '                moved += 1\n'
        '            except (OSError, shutil.Error) as e:\n'
        '                print(f"Warning: Failed to move {src.name}: {e}")\n'
        '\n'
        '        if self._undo_log:\n'
        '            log_path = self.target_dir / ".file-organizer-undo.json"\n'
        '            with open(log_path, "w") as f:\n'
        '                json.dump(self._undo_log, f, indent=2)\n'
        '\n'
        '        return moved\n'
        '\n'
        '    def _collect_files(self) -> list[Path]:\n'
        '        """Collect all files to process."""\n'
        '        if self.recursive:\n'
        '            return self._walk(self.target_dir, 0)\n'
        '        return [f for f in self.target_dir.iterdir() if f.is_file()]\n'
        '\n'
        '    def _walk(self, directory: Path, depth: int) -> list[Path]:\n'
        '        """Recursively collect files with depth limit."""\n'
        '        if depth > self.max_depth:\n'
        '            return []\n'
        '\n'
        '        files = []\n'
        '        try:\n'
        '            for entry in sorted(directory.iterdir()):\n'
        '                if entry.is_file() and not entry.name.startswith("."):\n'
        '                    files.append(entry)\n'
        '                elif entry.is_dir() and not entry.name.startswith("."):\n'
        '                    if entry.name not in self._get_category_dirs():\n'
        '                        files.extend(self._walk(entry, depth + 1))\n'
        '        except PermissionError:\n'
        '            pass\n'
        '\n'
        '        return files\n'
        '\n'
        '    def _should_exclude(self, file_path: Path) -> bool:\n'
        '        """Check if a file matches any exclusion pattern."""\n'
        '        for pattern in self.exclude_patterns:\n'
        '            if file_path.match(pattern):\n'
        '                return True\n'
        '        return False\n'
        '\n'
        '    def _resolve_collision(self, dest_path: Path) -> Path:\n'
        '        """Generate a unique filename to avoid collisions."""\n'
        '        stem = dest_path.stem\n'
        '        suffix = dest_path.suffix\n'
        '        parent = dest_path.parent\n'
        '        counter = 1\n'
        '\n'
        '        while dest_path.exists():\n'
        '            dest_path = parent / f"{stem}_{counter}{suffix}"\n'
        '            counter += 1\n'
        '            if counter > 1000:\n'
        '                raise RuntimeError(\n'
        '                    f"Too many collisions for {stem}{suffix}"\n'
        '                )\n'
        '\n'
        '        return dest_path\n'
        '\n'
        '    def _get_category_dirs(self) -> set[str]:\n'
        '        """Get set of category directory names."""\n'
        '        from .file_types import get_all_categories\n'
        '        return set(get_all_categories())\n'
    )

    # Test file: exercises triple quotes, < operators, escaped strings
    test_main_py = (
        '"""Tests for the main module."""\n'
        '\n'
        'import tempfile\n'
        'import shutil\n'
        'from pathlib import Path\n'
        'from unittest import TestCase\n'
        '\n'
        'from file_organizer.main import (\n'
        '    parse_args, validate_directory, count_files, main\n'
        ')\n'
        '\n'
        '\n'
        'class TestParseArgs(TestCase):\n'
        '    """Test command-line argument parsing."""\n'
        '\n'
        '    def test_organize_basic(self):\n'
        '        args = parse_args(["organize", "/tmp/test"])\n'
        '        self.assertEqual(args.command, "organize")\n'
        '        self.assertEqual(args.directory, Path("/tmp/test"))\n'
        '        self.assertFalse(args.dry_run)\n'
        '\n'
        '    def test_organize_with_flags(self):\n'
        '        args = parse_args(\n'
        '            ["organize", "/tmp/test", "--dry-run", "-r", "--max-depth", "5"]\n'
        '        )\n'
        '        self.assertTrue(args.dry_run)\n'
        '        self.assertTrue(args.recursive)\n'
        '        self.assertEqual(args.max_depth, 5)\n'
        '\n'
        '    def test_organize_with_exclude(self):\n'
        '        args = parse_args(\n'
        '            ["organize", "/tmp/test", "--exclude", "*.log", "*.tmp"]\n'
        '        )\n'
        '        self.assertEqual(args.exclude, ["*.log", "*.tmp"])\n'
        '\n'
        '\n'
        'class TestValidateDirectory(TestCase):\n'
        '    def setUp(self):\n'
        '        self.test_dir = Path(tempfile.mkdtemp())\n'
        '\n'
        '    def tearDown(self):\n'
        '        shutil.rmtree(self.test_dir, ignore_errors=True)\n'
        '\n'
        '    def test_valid_directory(self):\n'
        '        self.assertTrue(validate_directory(self.test_dir))\n'
        '\n'
        '    def test_nonexistent_directory(self):\n'
        '        self.assertFalse(validate_directory(Path("/nonexistent/path")))\n'
        '\n'
        '    def test_file_not_directory(self):\n'
        '        file_path = self.test_dir / "test.txt"\n'
        '        file_path.touch()\n'
        '        self.assertFalse(validate_directory(file_path))\n'
        '\n'
        '    def test_system_directory_blocked(self):\n'
        '        self.assertFalse(validate_directory(Path("/")))\n'
        '\n'
        '\n'
        'class TestCountFiles(TestCase):\n'
        '    def setUp(self):\n'
        '        self.test_dir = Path(tempfile.mkdtemp())\n'
        '        (self.test_dir / "photo.jpg").touch()\n'
        '        (self.test_dir / "photo2.png").touch()\n'
        '        (self.test_dir / "doc.pdf").touch()\n'
        '        (self.test_dir / "script.py").touch()\n'
        '        (self.test_dir / "data.json").touch()\n'
        '        (self.test_dir / "unknown.xyz").touch()\n'
        '\n'
        '    def tearDown(self):\n'
        '        shutil.rmtree(self.test_dir, ignore_errors=True)\n'
        '\n'
        '    def test_count_basic(self):\n'
        '        counts = count_files(self.test_dir)\n'
        '        self.assertEqual(counts.get("Images", 0), 2)\n'
        '        self.assertEqual(counts.get("Documents", 0), 1)\n'
        '        self.assertEqual(counts.get("Code", 0), 1)\n'
        '        self.assertEqual(counts.get("Data", 0), 1)\n'
        '        self.assertEqual(counts.get("Other", 0), 1)\n'
        '\n'
        '    def test_count_empty_dir(self):\n'
        '        empty = self.test_dir / "empty"\n'
        '        empty.mkdir()\n'
        '        counts = count_files(empty)\n'
        '        self.assertEqual(len(counts), 0)\n'
        '\n'
        '    def test_file_size_comparison(self):\n'
        '        """Verify size-based operations use < operator correctly."""\n'
        '        large_file = self.test_dir / "large.pdf"\n'
        '        large_file.write_bytes(b"x" * 2048)\n'
        '        small_file = self.test_dir / "small.pdf"\n'
        '        small_file.write_bytes(b"x" * 100)\n'
        '        self.assertTrue(small_file.stat().st_size < large_file.stat().st_size)\n'
        '        self.assertTrue(large_file.stat().st_size < 1024 * 1024)\n'
        '        self.assertFalse(small_file.stat().st_size < 50)\n'
        '\n'
        '\n'
        'class TestEndToEnd(TestCase):\n'
        '    def setUp(self):\n'
        '        self.test_dir = Path(tempfile.mkdtemp())\n'
        '        (self.test_dir / "vacation.jpg").write_bytes(\n'
        '            b"\\xff\\xd8\\xff" + b"\\x00" * 1000\n'
        '        )\n'
        '        (self.test_dir / "resume.pdf").write_bytes(\n'
        '            b"%PDF-1.4" + b"\\x00" * 500\n'
        '        )\n'
        '        (self.test_dir / "script.py").write_text(\'print("hello")\')\n'
        '        (self.test_dir / "config.json").write_text(\n'
        '            \'{"key": "value"}\'\n'
        '        )\n'
        '        (self.test_dir / "song.mp3").write_bytes(\n'
        '            b"ID3" + b"\\x00" * 2000\n'
        '        )\n'
        '        (self.test_dir / "archive.zip").write_bytes(\n'
        '            b"PK" + b"\\x00" * 300\n'
        '        )\n'
        '\n'
        '    def tearDown(self):\n'
        '        shutil.rmtree(self.test_dir, ignore_errors=True)\n'
        '\n'
        '    def test_dry_run_no_changes(self):\n'
        '        result = main(["organize", str(self.test_dir), "--dry-run"])\n'
        '        self.assertEqual(result, 0)\n'
        '        self.assertTrue((self.test_dir / "vacation.jpg").exists())\n'
        '\n'
        '    def test_organize_creates_categories(self):\n'
        '        result = main(["organize", str(self.test_dir)])\n'
        '        self.assertEqual(result, 0)\n'
        '        self.assertTrue((self.test_dir / "Images").is_dir())\n'
        '        self.assertTrue((self.test_dir / "Documents").is_dir())\n'
        '\n'
        '    def test_undo_log_created(self):\n'
        '        main(["organize", str(self.test_dir)])\n'
        '        log_path = self.test_dir / ".file-organizer-undo.json"\n'
        '        self.assertTrue(log_path.exists())\n'
        '        import json\n'
        '        with open(log_path) as f:\n'
        '            log = json.load(f)\n'
        '        self.assertTrue(len(log) > 0)\n'
        '\n'
        '\n'
        'class TestEdgeCases(TestCase):\n'
        '    def test_html_content_with_angle_brackets(self):\n'
        '        """Test that < and > in file content don\'t break anything."""\n'
        '        test_dir = Path(tempfile.mkdtemp())\n'
        '        try:\n'
        '            html_file = test_dir / "page.html"\n'
        '            html_file.write_text(\n'
        '                "<!DOCTYPE html>\\n"\n'
        '                "<html>\\n"\n'
        '                "<head><title>Test</title></head>\\n"\n'
        '                "<body>\\n"\n'
        '                "  <div class=\\"container\\">\\n"\n'
        '                "    <script>\\n"\n'
        '                "      if (x < 10 && y > 5) {\\n"\n'
        '                "        console.log(\\"x < 10\\");\\n"\n'
        '                "      }\\n"\n'
        '                "      const items = arr.filter(i => i.size < MAX_SIZE);\\n"\n'
        '                "    </script>\\n"\n'
        '                "  </div>\\n"\n'
        '                "</body>\\n"\n'
        '                "</html>"\n'
        '            )\n'
        '            counts = count_files(test_dir)\n'
        '            self.assertEqual(counts.get("Other", 0), 1)\n'
        '        finally:\n'
        '            shutil.rmtree(test_dir, ignore_errors=True)\n'
        '\n'
        '    def test_filename_with_special_chars(self):\n'
        '        test_dir = Path(tempfile.mkdtemp())\n'
        '        try:\n'
        '            (test_dir / "file (1).jpg").touch()\n'
        '            (test_dir / "data [final].json").touch()\n'
        '            counts = count_files(test_dir)\n'
        '            self.assertEqual(counts.get("Images", 0), 1)\n'
        '        finally:\n'
        '            shutil.rmtree(test_dir, ignore_errors=True)\n'
    )

    # ── Build multi-turn conversation ──────────────────────────────────────────
    call_id_counter = [0]

    def next_call_id():
        call_id_counter[0] += 1
        return f"call_{call_id_counter[0]:032X}"

    workdir = "/private/tmp/opencode-serve-workdir"
    conversation = [{"role": "system", "content": system_prompt}]
    req_bodies = []

    # ─── REQUEST 1: Initial PRD prompt (~12K tokens with 11 tools) ────────────
    conversation.append({
        "role": "user",
        "content": (
            "Build a Python CLI file organizer tool. Requirements:\n\n"
            "1. Take a directory path as input\n"
            "2. Scan all files and categorize them by extension (Images, "
            "Documents, Code, Data, Media, Archives)\n"
            "3. Move files into category subdirectories\n"
            "4. Support --dry-run to preview without moving\n"
            "5. Support --recursive with configurable --max-depth\n"
            "6. Support --exclude patterns\n"
            "7. Have an undo command using a JSON log\n"
            "8. Handle edge cases: name collisions, permission errors, "
            "system directories\n"
            "9. Include comprehensive tests\n\n"
            "Create the project structure:\n"
            "```\n"
            "file-organizer/\n"
            "    __init__.py\n"
            "    __main__.py\n"
            "    main.py\n"
            "    organizer.py\n"
            "    file_types.py\n"
            "    tests/\n"
            "        __init__.py\n"
            "        test_main.py\n"
            "        test_organizer.py\n"
            "```\n\n"
            "Start by writing main.py with the CLI argument parser, directory "
            "validation, file counting, and the main entry point. Make it "
            "production quality with type hints and docstrings."
        )
    })
    req_bodies.append({
        "model": "test",
        "messages": [m.copy() for m in conversation],
        "tools": tools,
        "max_tokens": 4096,
        "temperature": 0
    })

    # ─── REQUEST 2: After write main.py, ask for file_types.py (~14K tokens) ──
    cid = next_call_id()
    conversation.append({
        "role": "assistant", "content": None,
        "tool_calls": [{"id": cid, "type": "function", "function": {
            "name": "write",
            "arguments": json.dumps({
                "filePath": f"{workdir}/file_organizer/main.py",
                "content": main_py
            })
        }}]
    })
    conversation.append({
        "role": "tool", "tool_call_id": cid,
        "content": "File written successfully to file_organizer/main.py"
    })
    conversation.append({
        "role": "user",
        "content": (
            "Good. Now create file_types.py with comprehensive "
            "extension-to-category mappings. Include at least 60 extensions "
            "across all categories."
        )
    })
    req_bodies.append({
        "model": "test",
        "messages": [m.copy() for m in conversation],
        "tools": tools,
        "max_tokens": 4096,
        "temperature": 0
    })

    # ─── REQUEST 3: After write file_types.py, ask for organizer.py (~16K) ────
    cid = next_call_id()
    conversation.append({
        "role": "assistant", "content": None,
        "tool_calls": [{"id": cid, "type": "function", "function": {
            "name": "write",
            "arguments": json.dumps({
                "filePath": f"{workdir}/file_organizer/file_types.py",
                "content": file_types_py
            })
        }}]
    })
    conversation.append({
        "role": "tool", "tool_call_id": cid,
        "content": "File written successfully to file_organizer/file_types.py"
    })
    conversation.append({
        "role": "user",
        "content": (
            "Now write organizer.py with the core FileOrganizer class. It "
            "needs: scanning, planning, executing moves, name collision "
            "resolution, and undo log generation."
        )
    })
    req_bodies.append({
        "model": "test",
        "messages": [m.copy() for m in conversation],
        "tools": tools,
        "max_tokens": 4096,
        "temperature": 0
    })

    # ─── REQUEST 4: After write organizer.py, read main.py back (~18K) ────────
    cid = next_call_id()
    conversation.append({
        "role": "assistant", "content": None,
        "tool_calls": [{"id": cid, "type": "function", "function": {
            "name": "write",
            "arguments": json.dumps({
                "filePath": f"{workdir}/file_organizer/organizer.py",
                "content": organizer_py
            })
        }}]
    })
    conversation.append({
        "role": "tool", "tool_call_id": cid,
        "content": "File written successfully to file_organizer/organizer.py"
    })
    conversation.append({
        "role": "user",
        "content": (
            "Read main.py back to verify the imports and validate_directory "
            "function. Then fix the missing 'import os' that validate_directory "
            "needs for os.access()."
        )
    })
    req_bodies.append({
        "model": "test",
        "messages": [m.copy() for m in conversation],
        "tools": tools,
        "max_tokens": 1024,
        "temperature": 0
    })

    # ─── REQUEST 5: After read, ask for edit (< operator, import fix) (~20K) ──
    cid = next_call_id()
    conversation.append({
        "role": "assistant", "content": None,
        "tool_calls": [{"id": cid, "type": "function", "function": {
            "name": "read",
            "arguments": json.dumps({
                "filePath": f"{workdir}/file_organizer/main.py"
            })
        }}]
    })
    # Realistic read result with line numbers (like OpenCode returns)
    read_lines = []
    for i, line in enumerate(main_py.split('\n'), 1):
        read_lines.append(f"{i}: {line}")
    read_result = (
        f"<path>{workdir}/file_organizer/main.py</path>\n"
        f"<type>file</type>\n"
        f"<content>\n" + '\n'.join(read_lines) + "\n</content>"
    )
    conversation.append({
        "role": "tool", "tool_call_id": cid, "content": read_result
    })
    conversation.append({
        "role": "user",
        "content": (
            "I see the bug: validate_directory uses os.access() but 'import os'"
            " is missing. Fix it by adding the import. Also add a size limit "
            "check: skip files where file.stat().st_size > 1073741824 (1GB) "
            "with a warning. Use the < operator properly in the condition. "
            "Add this to the _walk_with_depth function."
        )
    })
    req_bodies.append({
        "model": "test",
        "messages": [m.copy() for m in conversation],
        "tools": tools,
        "max_tokens": 1024,
        "temperature": 0
    })

    # ─── REQUEST 6: After edit, ask to write tests with triple quotes (~21K) ──
    cid = next_call_id()
    conversation.append({
        "role": "assistant", "content": None,
        "tool_calls": [{"id": cid, "type": "function", "function": {
            "name": "edit",
            "arguments": json.dumps({
                "filePath": f"{workdir}/file_organizer/main.py",
                "oldString": "import argparse\nimport sys\nfrom pathlib import Path",
                "newString": (
                    "import argparse\nimport os\nimport sys\n"
                    "from pathlib import Path"
                )
            })
        }}]
    })
    conversation.append({
        "role": "tool", "tool_call_id": cid,
        "content": "Edit applied successfully."
    })
    conversation.append({
        "role": "user",
        "content": (
            "Write comprehensive tests in tests/test_main.py. Include:\n"
            "- Argument parsing tests\n"
            "- Directory validation tests\n"
            "- File counting tests\n"
            "- End-to-end organize tests\n"
            "- Edge cases with HTML content containing < and > characters\n"
            "- Tests with files that have triple-quoted docstrings\n"
            "Use unittest.TestCase."
        )
    })
    req_bodies.append({
        "model": "test",
        "messages": [m.copy() for m in conversation],
        "tools": tools,
        "max_tokens": 4096,
        "temperature": 0
    })

    # ─── REQUEST 7: After write tests, run them with bash (~22K) ──────────────
    cid = next_call_id()
    conversation.append({
        "role": "assistant", "content": None,
        "tool_calls": [{"id": cid, "type": "function", "function": {
            "name": "write",
            "arguments": json.dumps({
                "filePath": f"{workdir}/file_organizer/tests/test_main.py",
                "content": test_main_py
            })
        }}]
    })
    conversation.append({
        "role": "tool", "tool_call_id": cid,
        "content": "File written successfully to file_organizer/tests/test_main.py"
    })
    conversation.append({
        "role": "user",
        "content": "Run the tests to see if they pass. Use python -m pytest -v."
    })
    req_bodies.append({
        "model": "test",
        "messages": [m.copy() for m in conversation],
        "tools": tools,
        "max_tokens": 512,
        "temperature": 0
    })

    # ─── REQUEST 8: After test run, grep + fix (multi-step reasoning) (~24K) ──
    cid = next_call_id()
    test_output = (
        "============================= test session starts "
        "==============================\n"
        "platform darwin -- Python 3.12.4, pytest-8.3.2\n"
        "rootdir: /private/tmp/opencode-serve-workdir\n"
        "collecting ... collected 14 items\n\n"
        "file_organizer/tests/test_main.py::TestParseArgs::"
        "test_organize_basic PASSED\n"
        "file_organizer/tests/test_main.py::TestParseArgs::"
        "test_organize_with_flags PASSED\n"
        "file_organizer/tests/test_main.py::TestParseArgs::"
        "test_organize_with_exclude PASSED\n"
        "file_organizer/tests/test_main.py::TestValidateDirectory::"
        "test_valid_directory PASSED\n"
        "file_organizer/tests/test_main.py::TestValidateDirectory::"
        "test_nonexistent_directory PASSED\n"
        "file_organizer/tests/test_main.py::TestValidateDirectory::"
        "test_file_not_directory PASSED\n"
        "file_organizer/tests/test_main.py::TestValidateDirectory::"
        "test_system_directory_blocked PASSED\n"
        "file_organizer/tests/test_main.py::TestCountFiles::"
        "test_count_basic FAILED\n"
        "file_organizer/tests/test_main.py::TestCountFiles::"
        "test_count_empty_dir PASSED\n"
        "file_organizer/tests/test_main.py::TestCountFiles::"
        "test_file_size_comparison PASSED\n"
        "file_organizer/tests/test_main.py::TestEndToEnd::"
        "test_dry_run_no_changes PASSED\n"
        "file_organizer/tests/test_main.py::TestEndToEnd::"
        "test_organize_creates_categories PASSED\n"
        "file_organizer/tests/test_main.py::TestEndToEnd::"
        "test_undo_log_created PASSED\n"
        "file_organizer/tests/test_main.py::TestEdgeCases::"
        "test_html_content_with_angle_brackets PASSED\n\n"
        "=================================== FAILURES "
        "===================================\n"
        "_______ TestCountFiles.test_count_basic ________\n\n"
        "    def test_count_basic(self):\n"
        "        counts = count_files(self.test_dir)\n"
        "        self.assertEqual(counts.get(\"Images\", 0), 2)\n"
        "        self.assertEqual(counts.get(\"Documents\", 0), 1)\n"
        "        self.assertEqual(counts.get(\"Code\", 0), 1)\n"
        "        self.assertEqual(counts.get(\"Data\", 0), 1)\n"
        ">       self.assertEqual(counts.get(\"Other\", 0), 1)\n"
        "E       AssertionError: 0 != 1\n\n"
        "========================= 1 failed, 13 passed in 0.15s "
        "========================"
    )
    conversation.append({
        "role": "assistant", "content": None,
        "tool_calls": [{"id": cid, "type": "function", "function": {
            "name": "bash",
            "arguments": json.dumps({
                "command": (
                    f"cd {workdir} && python -m pytest "
                    "file_organizer/tests/ -v"
                ),
                "description": "Run the test suite with verbose output"
            })
        }}]
    })
    conversation.append({
        "role": "tool", "tool_call_id": cid, "content": test_output
    })
    conversation.append({
        "role": "user",
        "content": (
            "test_count_basic fails because count_files doesn't count 'Other' "
            "category files (the .xyz file). Search for "
            "'category == \"Other\"' in organizer.py to find where we skip "
            "uncategorized files, then fix it: count_files should report all "
            "files including Other, but plan() should still skip Other during "
            "organize. Use grep to find the exact line, then edit to fix."
        )
    })
    req_bodies.append({
        "model": "test",
        "messages": [m.copy() for m in conversation],
        "tools": tools,
        "max_tokens": 1024,
        "temperature": 0
    })

    # ── Write request bodies ──────────────────────────────────────────────────
    n = min(num_requests, len(req_bodies))
    for i in range(n):
        path = os.path.join(output_dir, f"req{i + 1}.json")
        with open(path, "w") as f:
            json.dump(req_bodies[i], f)

    # Print summary
    for i in range(n):
        body = req_bodies[i]
        msg_count = len(body["messages"])
        tool_count = len(body["tools"])
        body_size = len(json.dumps(body))
        users = [m for m in body["messages"] if m["role"] == "user"]
        last_user = users[-1]["content"][:80] if users else "?"
        print(
            f"  req{i + 1}.json: {msg_count} msgs, {tool_count} tools, "
            f"{body_size:,} bytes — {last_user}..."
        )

    return n


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/afm-realworld-bodies"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    count = generate_workload(out, n)
    print(f"\n  Generated {count} request bodies in {out}/")
