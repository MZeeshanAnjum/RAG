from config import OPENAI_API_KEY

import os
import io
from io import BytesIO
import base64

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import fitz 

from openai import OpenAI

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

def process_image(file):
    print("Processing image")

    client = OpenAI()
    img_bytes = file.read()

    # Encode to base64
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")

    # Send to GPT-4o-mini for OCR
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are an intelligent OCR and image understanding assistant. "
                        "1. First, determine if the uploaded image contains only text or contains charts, graphs, diagrams, or any kind of visual data. "
                        "2. If it only contains text, extract the text exactly as it appears. "
                        "3. If it contains visuals (charts, infographics, diagrams, images, etc.), provide a clear, concise description or summary of the useful information."
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                }
            ]
        }
    ],
)


    extracted_text = response.choices[0].message.content

    print("Image processed successfully.")
    print(f"Extracted text: {extracted_text}")
    return extracted_text


# def process_pdf(file):
#     print("Processing PDF file...")

#     pdf_reader = PdfReader(file)
#     text = ""
#     for page in pdf_reader.pages:
#         text += page.extract_text() + "\n"

#     print("PDF file processed successfully.")
#     return text

def process_pdf(file_obj):
    print("Processing PDF file...")

    # Read uploaded file bytes
    pdf_bytes = file_obj.read()

    pdf_text = ""
    # Open PDF from bytes for image detection & text extraction
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    for page_num, page in enumerate(doc, start=1):
        # 1. Extract selectable text
        page_text = page.get_text()

        # 2. Check for images
        images = page.get_images(full=True)
        ocr_text = ""

        if images:
            print(f"Page {page_num} contains {len(images)} image(s). Running OCR...")
            for img_index, img in enumerate(images, start=1):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]

                # Process the image using GPT-4o-mini OCR
                ocr_text += process_image(io.BytesIO(image_bytes)) + "\n"

        # 3. Merge text from both sources
        combined_text = (page_text or "") + "\n" + ocr_text
        pdf_text += combined_text + "\n"

    print("PDF file processed successfully.")
    return pdf_text

def Initialize_vector_store():

    print("Initializing vector store...")

    vector_store = Chroma(
        collection_name="Business_Information",
        embedding_function=embeddings,
        persist_directory="chroma_db"
    )

    print("Vector store initialized.")

    return vector_store

def chunk_text(text, chunk_size=1000, chunk_overlap=200):

    print("Chunking text...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_text(text)

    print(f"Text chunked into {len(chunks)} parts.")

    return chunks

def create_and_store_embeddings(chunks, vector_store, filename=None):

    print("Creating and storing embeddings...")

    documents = [
        Document(page_content=chunk, metadata={"source": filename, "chunk_index": i})
        for i, chunk in enumerate(chunks)
    ]

    vector_store.add_documents(documents)

    print("Embeddings created and stored successfully.")

def is_file_already_embedded(vector_store, filename):
    results = vector_store.get(
        where={"source": filename}
    )
    return len(results["documents"]) > 0

def process_document(uploaded_file, chunk_size=1000, chunk_overlap=200):

    vector_store = Initialize_vector_store()

    mime_type = uploaded_file.type
    file_name = uploaded_file.name

    #  Check if file already exists
    if is_file_already_embedded(vector_store, file_name):
        print(f"File '{file_name}' is already embedded. Skipping...")
        return

    if mime_type == "text/plain":
        print("It's a TXT file")

        text_content = uploaded_file.read().decode("utf-8")
        text_chunks = chunk_text(text_content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        create_and_store_embeddings(text_chunks, vector_store, filename=file_name)

    
    elif mime_type == "application/pdf":
        print("It's a PDF file")

        text_content = process_pdf(uploaded_file)
        text_chunks = chunk_text(text_content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        create_and_store_embeddings(text_chunks, vector_store, filename=file_name)

    elif mime_type.startswith("image/"):
        print("It's an image file")

        text_content = process_image(uploaded_file)
        text_chunks = chunk_text(text_content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        create_and_store_embeddings(text_chunks, vector_store, filename=file_name)
