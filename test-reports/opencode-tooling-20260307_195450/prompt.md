# Product Requirements Document (PRD)

## Product: Simple Local File Organizer

### Version

1.0

### Date

March 2026

---

# 1. Product Overview

Simple Local File Organizer is a command-line utility that automatically organizes files in a directory into categorized folders based on file type.

The tool scans a target folder and moves files into folders such as **Images**, **Documents**, **Videos**, and **Archives**. The goal is to provide users with a quick way to clean up cluttered directories like Downloads.

The application runs locally and requires no external services.

---

# 2. Objectives

## Primary Objectives

* Automatically organize files into categorized folders
* Reduce clutter in common directories (e.g., Downloads)
* Provide a simple command-line interface

## Secondary Objectives

* Allow users to preview actions before files are moved
* Provide fast execution for large directories
* Keep the tool lightweight and easy to install

---

# 3. Target Users

### Primary Users

* Developers
* Power users
* Anyone with cluttered directories

### Secondary Users

* Students
* Office workers managing downloaded files

---

# 4. User Scenarios

## Scenario 1 — Organize Downloads Folder

A user wants to clean up their Downloads folder.

Command:

```
file-organizer organize ~/Downloads
```

System behavior:

1. Scans all files in the directory
2. Determines file type
3. Creates category folders if needed
4. Moves files into appropriate folders

Example result:

```
Downloads/
    Images/
    Documents/
    Videos/
    Archives/
```

---

## Scenario 2 — Preview Organization

A user wants to preview changes before moving files.

Command:

```
file-organizer organize ~/Downloads --dry-run
```

System output:

```
report.pdf -> Documents/
photo.png -> Images/
video.mp4 -> Videos/
```

No files are moved.

---

# 5. Functional Requirements

## FR1 — Directory Scanning

The system must:

* scan all files in a specified directory
* ignore folders unless specified otherwise

---

## FR2 — File Type Detection

The system must categorize files using file extensions.

Example mapping:

| Extension   | Category  |
| ----------- | --------- |
| .jpg, .png  | Images    |
| .pdf, .docx | Documents |
| .mp4, .mov  | Videos    |
| .zip, .tar  | Archives  |

---

## FR3 — Folder Creation

The system must create category folders if they do not already exist.

Example:

```
Images/
Documents/
Videos/
Archives/
```

---

## FR4 — File Moving

The system must move files into the appropriate category folder.

Example:

```
photo.png -> Images/photo.png
report.pdf -> Documents/report.pdf
```

---

## FR5 — Dry Run Mode

The system must support a preview mode.

Command:

```
file-organizer organize <folder> --dry-run
```

The tool prints intended actions without moving files.

---

# 6. Command Line Interface

### Organize Files

```
file-organizer organize <directory>
```

---

### Preview Organization

```
file-organizer organize <directory> --dry-run
```

---

# 7. Technical Requirements

### Programming Language

Python

---

### Suggested Libraries

```
argparse
pathlib
shutil
```

---

# 8. Project Structure

```
file-organizer/
│
├── main.py
├── organizer.py
├── file_types.py
└── utils.py
```

---

# 9. Performance Requirements

The system must:

* process directories containing up to **10,000 files**
* complete operations within **a few seconds**

---

# 10. Privacy Requirements

The system must:

* operate locally
* not transmit files externally
* not collect user data

---

# 11. Future Enhancements

Possible future features include:

* recursive folder organization
* customizable file categories
* graphical user interface
* duplicate file detection

---

# 12. Success Criteria

The project is considered successful if:

* users can organize directories with a single command
* files are categorized correctly based on type
* dry-run mode accurately previews actions
* the tool works reliably on large directories
