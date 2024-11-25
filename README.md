# HuggingFace with LangChain Integration

This repository demonstrates the integration of HuggingFace models with LangChain to build and use large language models (LLMs) for text generation and question-answering tasks. The implementation explores HuggingFace endpoints, pipelines, and GPU acceleration with LangChain's intuitive APIs.

---

## Prerequisites

Before running the code, ensure the following tools and libraries are installed:

- **Python**: Version 3.8 or higher
- Required Python libraries: `langchain`, `langchain_huggingface`, `transformers`, `dotenv`, `os`

---

## Installation

### Clone the Repository:
```bash
git clone https://github.com/ELAKIYA-SEKAR/langchain-huggingface.git
cd langchain-huggingface


### Install Dependencies:
```bash
pip install -r requirements.txt
```

### Set up Environment Variables:
Create a `.env` file in the project root directory and add your HuggingFace token:
```plaintext
HUGGING_FACE_TOKEN=<your_huggingface_api_token>
```

### Verify GPU Availability (Optional for GPU users):
If you intend to use GPU acceleration, ensure your machine has a compatible GPU and the necessary drivers installed.

---

## Code Overview

### HuggingFace Endpoint Integration

The `HuggingFaceEndpoint` class connects LangChain to a model hosted on HuggingFace.

- **Model in use**: `Qwen/Qwen2.5-Coder-32B-Instruct`

#### Example:
```python
from langchain_huggingface import HuggingFaceEndpoint
import os

hf_token = os.getenv("HUGGING_FACE_TOKEN")
repo_id = "Qwen/Qwen2.5-Coder-32B-Instruct"

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    max_length=128,
    temperature=0.7,
    token=hf_token
)
```

---

### Prompt Creation

LangChain's `PromptTemplate` simplifies prompt engineering for better LLM interaction.

#### Example:
```python
from langchain import PromptTemplate

template = """Question: {question}
Answer: Let's think step by step,"""
prompt = PromptTemplate(template=template, input_variables=['question'])
```

---

### HuggingFace Pipeline Usage

For offline/local model inference, HuggingFace pipelines are integrated with LangChain.

#### Example:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline

model_id = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)
hf = HuggingFacePipeline(pipeline=pipe)
```

---

### GPU Acceleration with LangChain

Leverage GPU for faster inference:

#### Example:
```python
gpu_llm = HuggingFacePipeline.from_model_id(
    model_id="gpt2",
    task="text-generation",
    device=-1,  # Use 0 for GPU
    pipeline_kwargs={"max_new_tokens": 100}
)
```

---

## Example Use Cases

### Answering Questions using HuggingFace Endpoint:
```python
question = "Who played the most innings in the history of Cricket?"
print(llm_chain.invoke(question))
```

---

### Pipeline Inference:
```python
print(hf.invoke("Langchain is a company"))
```

---

### GPU Acceleration:
```python
question = "What is artificial intelligence?"
print(chain.invoke({"question": question}))
```

---

## Notes on Device Usage

- **`device=0`**: Utilizes GPU for faster computation.
- **`device=-1`**: Falls back to CPU (default behavior).

To unlock GPU capabilities, ensure your system has the required CUDA libraries installed.

