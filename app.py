# =========================================
# RULE EXTRACTOR (LLM + API) - CHECKLIST STYLE
# =========================================

pip install -q openai langchain langchain-openai langchain-text-splitters faiss-cpu python-docx pypdf

import os
from google.colab import files
from docx import Document
from pypdf import PdfReader
import json

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ----------------------------
# 1. SET OPENAI API KEY
# ----------------------------
os.environ["OPENAI_API_KEY"] = "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"

# ----------------------------
# 2. UPLOAD DOCUMENT
# ----------------------------
print("Upload your rule guideline file (.txt, .pdf, .docx)")
uploaded = files.upload()
file_path = list(uploaded.keys())[0]

# ----------------------------
# 3. LOAD FILE
# ----------------------------
def load_file(path):
    if path.endswith(".txt"):
        return open(path, "r", encoding="utf-8").read()
    elif path.endswith(".pdf"):
        reader = PdfReader(path)
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    elif path.endswith(".docx"):
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    else:
        raise ValueError("Unsupported file format")

text = load_file(file_path)
print(f"Loaded {len(text)} characters")

# ----------------------------
# 4. SPLIT INTO CHUNKS
# ----------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
chunks = splitter.split_text(text)
print(f"Created {len(chunks)} chunks")

# ----------------------------
# 5. VECTOR STORE FOR RETRIEVAL
# ----------------------------
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

# ----------------------------
# 6. GPT PROMPT
# ----------------------------
prompt = ChatPromptTemplate.from_template("""
You are a rule extraction AI. Extract ALL rules from the given text. 
Rules may be anywhere in the text, even if unstructured.

Output format: JSON array with a single key "rule" per object, e.g.

[
  {{"rule": "First rule text"}},
  {{"rule": "Second rule text"}}
]

Do NOT add explanations, numbering, or text outside JSON array.

Context:
{context}
""")

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# ----------------------------
# 7. RAG RETRIEVAL & EXTRACTION
# ----------------------------
def extract_rules(query="Extract all rules"):
    docs = retriever.invoke(query)
    context = "\n\n".join(d.page_content for d in docs)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context})

print("Extracting rules with LLM...")
rules_json_text = extract_rules()

# ----------------------------
# 8. PARSE AND SAVE JSON
# ----------------------------
try:
    rules_list = json.loads(rules_json_text)
except:
    print("Warning: Failed to parse JSON. Falling back to line-by-line.")
    rules_list = [{"rule": line.strip()} for line in rules_json_text.split("\n") if line.strip()]

with open("rules.json", "w", encoding="utf-8") as f:
    json.dump(rules_list, f, indent=2)

# ----------------------------
# 9. SAVE CHECKLIST FORMAT
# ----------------------------
with open("rules_checklist.txt", "w", encoding="utf-8") as f:
    for r in rules_list:
        f.write(f"[ ] {r['rule']}\n")

# ----------------------------
# 10. DOWNLOAD FILES
# ----------------------------
files.download("rules.json")
files.download("rules_checklist.txt")

print("✅ Extraction complete! Files downloaded: rules.json, rules_checklist.txt")
