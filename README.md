# PDF Q&A with RAG and Gemini

A Retrieval-Augmented Generation (RAG) application that allows users to ask questions about a PDF and receive answers based on relevant information retrieved from the document.

## Live Demo

[Open the deployed Streamlit app](PASTE-YOUR-STREAMLIT-URL-HERE)

## Overview

This project combines semantic search with a large language model to create a question-answering system for PDF documents.

Instead of sending the entire PDF directly to the language model, the application:

1. Extracts text from the PDF.
2. Splits the text into smaller chunks.
3. Converts the chunks into vector embeddings.
4. Retrieves the most relevant chunks for a user's question.
5. Sends the retrieved context to Gemini through OpenRouter.
6. Generates an answer based on the retrieved document context.
7. Displays relevant source pages.

## Architecture

```text
PDF Document
     |
     v
Text Extraction
     |
     v
Text Chunking
     |
     v
Sentence Transformer Embeddings
     |
     v
Vector Similarity Search
     |
     v
Relevant Context
     |
     v
Gemini 2.5 Flash
     |
     v
Answer + Sources
Features

* Upload and process PDF documents
* Semantic search using sentence embeddings
* Retrieval-Augmented Generation (RAG)
* Gemini 2.5 Flash for answer generation
* Source/page references
* Streamlit web interface
* Answers grounded in the uploaded document

Tech Stack

* Python
* Streamlit
* PyMuPDF
* NumPy
* Sentence Transformers
* OpenRouter
* Gemini 2.5 Flash

How It Works

1. PDF Processing

The application extracts text from the uploaded PDF using PyMuPDF.

2. Text Chunking

The extracted text is divided into smaller chunks so that relevant sections can be retrieved efficiently.

3. Embeddings

Each chunk is converted into a numerical vector using a Sentence Transformer model.

4. Retrieval

When a user asks a question, the question is converted into an embedding. The application compares it with the document chunk embeddings and retrieves the most relevant sections.

5. Generation

The retrieved sections are provided as context to Gemini 2.5 Flash through OpenRouter.

The model generates an answer using the retrieved context.
