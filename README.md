# Apparel Agent - AI-Powered Apparel Design Extraction & Generation

An intelligent agent that extracts design attributes from apparel images (specifically Kurtis) and generates high-quality, marketplace-ready fashion images using a multi-stage AI pipeline.

## 🚀 Key Features
- **Multi-Image Design Extraction**: Uses Claude 3 Haiku/GPT-4o to analyze multiple reference images.
- **AI-Powered Prompt Engineering**: Generates optimized prompts for fashion synthesis.
- **Reference-Guided Generation**: Uses OpenAI's image editing capabilities to maintain design integrity.
- **Interactive Web UI**: Built with Gradio for an easy-to-use extraction and generation workflow.
- **Feedback Loop**: Integrated critic/verification stage to ensure design accuracy.

## Architecture Overview

The system follows a **three-stage architecture**:

1. **Design Extraction Stage**: Analyzes kurti images to extract design attributes (patterns, colors, neck design, etc.)
2. **Generation Stage**: Uses extracted attributes to generate optimized prompts and create new kurti images.
3. **Critic/Verification Stage**: Compares the generated image with the original reference to ensure accuracy.

![Architecture Diagram](./docs/architecture_diagram.png)

## 🛠️ Installation

**Required Python version:** 3.13+

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ravisahu2017/apparel-agent.git
   cd apparel-agent
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Environment:**
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   ```

## 🚀 How to Run

### 1. Web Interface (Recommended)
Launch the interactive Gradio UI to process images through a browser:
```bash
python run_orchestrator.py
```
- **Upload**: Drop your kurti images (front, back, pattern, etc.).
- **Extract**: Click "Extract" to analyze design details.
- **Generate**: Choose a view (Front, Back, Side) to generate new fashion images.

### 2. CLI Mode
For automated processing:

**To extract attributes:**
```bash
python run_agent.py extract
```

**To generate with critic feedback (requires UUID from extract phase):**
```bash
python run_agent.py generate <uuid>
```

## 📂 Project Structure
- `run_orchestrator.py`: Entry point for the Gradio Web UI.
- `gradio_orchestrator.py`: Logic for the interactive web interface.
- `extractor_chain.py`: Vision-based attribute extraction logic.
- `generater_chain.py`: Prompt engineering and image generation logic.
- `tools/`: Utility modules for S3, database, and image processing.
- `output/`: Directory where generated images are saved.

## ⚙️ Customization

### Modify Extraction
Change vision models in `gradio_orchestrator.py` or `run_agent.py`:
- `anthropic/claude-3-haiku` (Default)
- `openai/gpt-4o`

### Marketplace Templates
The `GeneratorChain` in `generater_chain.py` supports multiple marketplace styles (Meesho, Amazon, Flipkart).

## 🤝 Contributing
Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to help improve this project.
