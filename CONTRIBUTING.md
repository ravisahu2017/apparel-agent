# Contributing to Apparel Agent

First off, thank you for considering contributing to Apparel Agent! It's people like you who make this project better for everyone.

## Table of Contents
1. [Code of Conduct](#code-of-conduct)
2. [How Can I Contribute?](#how-can-i-contribute)
    - [Reporting Bugs](#reporting-bugs)
    - [Suggesting Enhancements](#suggesting-enhancements)
    - [Pull Requests](#pull-requests)
3. [Development Setup](#development-setup)
4. [Style Guides](#style-guides)
    - [Python Style Guide](#python-style-guide)
    - [Commit Messages](#commit-messages)
5. [Project Structure](#project-structure)

## Code of Conduct
This project and everyone participating in it is expected to maintain a professional and respectful environment.

## How Can I Contribute?

### Reporting Bugs
Before creating bug reports, please check the existing issues to see if the problem has already been reported. When creating a bug report, please include:
- A clear and descriptive title
- Steps to reproduce the bug
- Expected vs. actual behavior
- Screenshots if applicable
- Your environment details (OS, Python version, etc.)

### Suggesting Enhancements
Enhancement suggestions are tracked as GitHub issues. When suggesting an enhancement, please:
- Explain why this enhancement would be useful
- Provide a step-by-step description of how the feature should work
- Mention any alternative solutions you've considered

### Pull Requests
1. Fork the repository and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. Ensure the test suite passes.
4. Make sure your code follows the style guidelines.
5. Issue a Pull Request with a clear description of your changes.

## Development Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/ravisahu2017/apparel-agent.git
   cd apparel-agent
   ```

2. **Set up a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables:**
   Create a `.env` file in the root directory and add your API keys:
   ```env
   OPENAI_API_KEY=your_key
   OPENROUTER_API_KEY=your_key
   ```

## Style Guides

### Python Style Guide
- We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code.
- Use `black` for code formatting.
- Ensure all functions and classes have descriptive docstrings.

### Commit Messages
- Use the present tense ("Add feature" not "Added feature").
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...").
- Limit the first line to 72 characters or less.

## Project Structure
- `tools/`: Core logic for image processing, S3, and database connections.
- `prompts/`: JSON and text templates for LLM prompts.
- `docs/`: Documentation and diagrams.
- `test/`: Test cases and sample images.
- `run_agent.py`: CLI entry point for the agent.
- `run_orchestrator.py`: run using gradio web ui