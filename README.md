# Project for Intel Unnati Industrial Training 2024
 ## PDF Chatbot using OpenVINO and RAG

 A PDF-based chatbot leveraging OpenVINO and RAG techniques for efficient question-answering, developed as part of the Intel Unnati Industrial Training 2024.

## Description

This project demonstrates how to create a chatbot that can answer questions related to a given PDF document using the Retrieval Augmented Generation (RAG) technique. The chatbot is implemented in Google Colab and uses various libraries including OpenVINO for efficient inference.

The project consists of a single notebook that performs the following tasks:
1. Reads and processes a PDF file
2. Generates a vector store from the PDF content
3. Uses a Language Model (LLM) to answer questions based on the vector store

## Components

| Component | Description |
|-----------|-------------|
| PDF Processing | Extracts text from a PDF file using PyPDF2 |
| Vector Store Generation | Creates a FAISS index from text chunks using sentence-transformers |
| LLM Integration | Uses TinyLlama model for generating responses |
| OpenVINO Optimization | Leverages Intel's OpenVINO toolkit for optimized model inference |
| User Interface | Implements a Gradio interface for easy interaction |

## How to Run

1. Open the notebook in Google Colab.
2. Run all cells in order.
3. Upload a PDF file when prompted.
4. Ask questions about the PDF content using the Gradio interface.

## Prerequisites

Before running the [notebook](https://colab.research.google.com/github/paharipratyush/intelunnati/blob/main/pdfchatbotopenvino.ipynb), you need to install the required packages and import the necessary libraries. Run the following commands in a code cell:

```python
# Install required packages
!pip install -q transformers sentence-transformers faiss-cpu PyPDF2 openvino-nightly
!pip install -q optimum[openvino]
!pip install numpy PyPDF2 sentence-transformers faiss-cpu optimum[intel] transformers nltk gradio

# Import required libraries
import numpy as np
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer
from optimum.intel import OVModelForCausalLM
import gc
import torch
import nltk
import gradio as gr
import tempfile
import os
import traceback
from google.colab import files

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
```

## Functionality

1. **PDF Processing**: The notebook reads the uploaded PDF and extracts its text content.
2. **Semantic Chunking**: The extracted text is divided into semantic chunks for better context preservation.
3. **Vector Store Creation**: Chunks are embedded using a sentence transformer model and stored in a FAISS index for quick retrieval.
4. **Question Answering**: When a user asks a question, the system:
   - Finds the most relevant chunks using semantic similarity
   - Constructs a prompt with the question and relevant context
   - Generates an answer using the TinyLlama model

## Configuration

The notebook uses default configurations, but you can modify the following:

- Chunk size and overlap in the `create_semantic_chunks` function
- Number of relevant chunks retrieved (k) in the `chatbot` function
- Model used for embeddings and language generation

## OpenVINO Integration

This project leverages Intel's OpenVINO toolkit for optimized inference:

 - The TinyLlama model is loaded and exported using `OVModelForCausalLM` from the `optimum.intel` package.
 - This allows for hardware-specific optimizations, potentially improving inference speed and efficiency, especially on Intel hardware.
 - OpenVINO optimizations are particularly beneficial for larger models or high-volume query processing.

## System Requirements

- Google Colab environment (or local setup with similar specifications)
- Internet connection for downloading models and libraries
- For optimal performance with OpenVINO, Intel CPU is recommended

## Performance Notes

- The use of OpenVINO optimizations may significantly improve performance, especially on Intel hardware
- Performance benefits may be more noticeable with larger models or when processing many queries
- The TinyLlama model is relatively small, which allows for quick responses but may limit the complexity of answers

## Troubleshooting

- If CUDA out of memory errors is encountered, try restarting the runtime or using a CPU-only version.
- Ensure all required libraries are correctly installed. Check the error message for missing packages.

## Limitations

- Performance depends on the quality and length of the uploaded PDF.
- Uses a small language model (TinyLlama) which may limit response quality for complex queries.

## Future Improvements

- Support for multiple PDF uploads
- Integration with more powerful language models
- Implementation of conversation history and context awareness
- Fine-tuning options for specific domains

## Team Members and their contributions

This project was developed by a team of 5 members as part of the Intel Unnati Industrial Training 2024:

- [Swetakshi Nanda](https://github.com/SwetakshiNanda18): Project lead, architecture design
- [Pratyush Pahari](https://github.com/paharipratyush): LLM integration and OpenVINO optimization
- [Arpan Bag](https://github.com/ArpanBag02): PDF Processing, Embedding generation
- [Akashdeep Mitra](https://github.com/Akashdeep099): User interface development and integration
- [Tulika Chakraborty](https://github.com/Tulika2003): Documentation of the complete Project

We would like to thank our mentor [Abhishek Nandy](https://github.com/AbhiLegend) and the Intel Unnati program for their guidance and support throughout this project.

## Contributing

Contributions to improve the chatbot are welcome. Please feel free to fork the repository and submit a Pull Request or open an issue to discuss about your changes.

## License

[MIT License](https://opensource.org/licenses/MIT)

## Acknowledgements

This project uses several open-source libraries and models:

- [Sentence-Transformers](https://github.com/UKPLab/sentence-transformers)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Transformers](https://github.com/huggingface/transformers)
- [TinyLlama](https://github.com/jzhang38/TinyLlama)
- [OpenVINO](https://github.com/openvinotoolkit/openvino)
- [Gradio](https://github.com/gradio-app/gradio)
