# =========================================
# RULE EXTRACTOR (LLM + API) - STREAMLIT CHECKLIST
# =========================================

import os
import json
import streamlit as st
from docx import Document
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ----------------------------
# 0. Streamlit page config
# ----------------------------
st.set_page_config(page_title="Rule Extractor", layout="wide")
st.title("📄 Rule Extractor - Checklist Style")

# ----------------------------
# 1. Input OpenAI API Key
# ----------------------------
api_key = st.text_input("Enter your OpenAI API Key", type="password")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
else:
    st.warning("Please enter your OpenAI API Key to continue.")
    st.stop()

# ----------------------------
# 2. File uploader
# ----------------------------
uploaded_file = st.file_uploader(
    "Upload your rule guideline file (.txt, .pdf, .docx)", 
    type=["txt", "pdf", "docx"]
)
if not uploaded_file:
    st.stop()

# ----------------------------
# 3. Load file content
# ----------------------------
def load_file(file):
    name = file.name
    if name.endswith(".txt"):
        return file.read().decode("utf-8")
    elif name.endswith(".pdf"):
        reader = PdfReader(file)
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    elif name.endswith(".docx"):
        doc = Document(file)
        return "\n".join(p.text for p in doc.paragraphs)
    else:
        raise ValueError("Unsupported file format")

text = load_file(uploaded_file)
st.success(f"Loaded {len(text)} characters")

# ----------------------------
# 4. Split text into chunks
# ----------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
chunks = splitter.split_text(text)
st.write(f"Created {len(chunks)} chunks for processing")

# ----------------------------
# 5. Vector store (Chroma)
# ----------------------------
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_texts(chunks, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

# ----------------------------
# 6. GPT Prompt
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
# 7. Extraction function
# ----------------------------
def extract_rules(query="Extract all rules"):
    docs = retriever.invoke(query)
    context = "\n\n".join(d.page_content for d in docs)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context})

# ----------------------------
# 8. Run extraction
# ----------------------------
if st.button("Extract Rules"):
    with st.spinner("Extracting rules with LLM..."):
        rules_json_text = extract_rules()

        # Parse JSON safely
        try:
            rules_list = json.loads(rules_json_text)
        except:
            st.warning("Failed to parse JSON, falling back to line-by-line")
            rules_list = [{"rule": line.strip()} for line in rules_json_text.split("\n") if line.strip()]

        # Save files in memory
        rules_json_str = json.dumps(rules_list, indent=2)
        checklist_str = "\n".join(f"[ ] {r['rule']}" for r in rules_list)

        # Display sample rules
        st.subheader("Sample Extracted Rules")
        st.write(rules_list[:10])

        # Download buttons
        st.download_button("Download JSON", data=rules_json_str, file_name="rules.json")
        st.download_button("Download Checklist", data=checklist_str, file_name="rules_checklist.txt")

        st.success("✅ Extraction complete!")
