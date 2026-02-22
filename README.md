# NPU-Piggy 🐷💰

*For everyone who checks their bank account and needs a moment. AI-powered budget tracking, fully on-device.*

## Collaborators
* **Iris Xu**: LLM LangChain design, Prompt engineering
* **Hamlet Abrahamyan**: React app front end, OCR
* **Yihan Wu**: NPU runtime deployment, OCR

---

## Inspiration & Vision
We've all had that moment: checking the bank account at the end of the month and it becomes a crime scene investigation. Who authorized this? When did this happen? For young adults wanting to save their first bucket of gold but occasionally shopping online at 2 AM, budgeting isn't a knowledge problem—it's a visibility problem!

Many existing budget apps attempt to solve this by asking you to sync your bank and hand over your most sensitive data to cloud servers. But your transaction history is more than just numbers; it reveals where you eat, how you live, and what you’re going through. That data is profoundly personal and should never leave your device. 

That is why we built **NPU-Piggy**, a fully on-device AI budget planner that turns a snapshot of a receipt into structured financial insight with **zero cloud dependency**. No accounts. No clouds. No sharing your privacy with a third party.

## What It Does
NPU-Piggy is an end-to-end expense tracking tool that runs entirely on your local machine. You scan any receipt—physical or digital—and NPU-Piggy will:
* **Extract** the text using on-device OCR.
* **Parse** the text into structured JSON data using a local Large Language Model (LLM).
* **Store** your expenses securely in React local storage.
* **Visualize** your spending patterns on a clean, personal dashboard.
* **Chat** with you through an interactive AI assistant to provide personalized budget planning feedback and suggestions.

## Technical Architecture & How We Built It
Our pipeline operates in four main stages, keeping privacy intact at every step:

1. **Capture**: Users photograph a receipt via their PC camera.
2. **OCR Extraction**: We utilize EasyOCR, uniquely optimized and deployed on-device via the **Qualcomm AI Hub**. The pipeline runs the detector on the **NPU** and the recognizer on the **CPU**, swiftly extracting raw text from the receipt image.
3. **LLM Parsing**: The raw OCR text is passed to a local **Mistral 7B** model via **Ollama**, orchestrated with **LangChain**. A structured prompt with explicit rules converts the messy text into validated JSON, backed by a Pydantic schema. Built-in retry logic and fallback mechanisms handle any malformed or incomplete responses gracefully.
4. **Storage & Insights**: The parsed expense data is stored locally using React local storage—no external database servers required. The React frontend renders a dynamic spending dashboard with interactive visualizations and hosts the local LLM chat interface for budget planning.

### Built With
* **Frontend**: React
* **Backend**: Python, Flask, LangChain
* **AI/ML Models**: Mistral 7B (via Ollama), EasyOCR
* **Runtime**: ONNX Runtime
* **Hardware**: Qualcomm NPU, CPU
