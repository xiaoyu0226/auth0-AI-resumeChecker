import os
from langchain.schema import Document
from typing import List
from pathlib import Path
from PyPDF2 import PdfReader

"""
Reads and returns a list of documents (resumes).
It looks for documents in a folder and assigns an access level based on file name
The supported document formats are PDF and TXT.
"""
def read_documents() -> List[Document]:
    current_dir = os.path.dirname(__file__)    # get current directory 
    resumes_dir = os.path.join(current_dir, "../uploads/")

    # List all files in the resumes directory
    resume_files = os.listdir(resumes_dir)

    documents = []

    for resume_file in resume_files:
        file_path = os.path.join(resumes_dir, resume_file)

        # Skip directories
        if os.path.isdir(file_path):
            continue

        # Read the resume content
        file_extension = resume_file.split('.')[-1].lower()
        resume_id = resume_file.split('.')[0]
        try:
            if file_extension == "pdf":
                content = read_pdf(file_path)
            elif file_extension == "txt":
                content = read_txt(file_path)
            else:
                continue  # Skip unsupported file types

            # Assuming the file name contains the access level or some other rule for classification
            access_level = "public" if "public" in resume_file else "private"
            
            # Create a document object
            document = Document(
                page_content=content,
                metadata={"id": resume_id, "access": access_level},    # id for the document is crucial for FGA filter
            )
            documents.append(document)

        except Exception as e:
            print(f"Error reading {resume_file}: {e}")

    return documents

"""
Reads a PDF file and extracts its text content.
"""
def read_pdf(file_path: str) -> str:
    content = ""
    try:
        with open(file_path, "rb") as file:
            reader = PdfReader(file)
            for page in reader.pages:
                content += page.extract_text()
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
    return content

"""
Reads a plain TXT file and returns its content.
"""
def read_txt(file_path: str) -> str:
    content = ""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
    except Exception as e:
        print(f"Error reading TXT {file_path}: {e}")
    return content
