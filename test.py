from dotenv import load_dotenv
from gradio_orchestrator import GradioOrchestrator
load_dotenv()


if __name__ == "__main__":
    gr = GradioOrchestrator()
    gr.run()
    