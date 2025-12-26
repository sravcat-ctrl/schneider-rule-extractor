import os
import json
import streamlit as st
from docx import Document
from pypdf import PdfReader
import openai

st.title("📄 Rule Extractor (Checklist)")

api_key = st.text_input("Enter OpenAI API Key", type="password")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
else:
    st.stop()

uploaded_file = st.file_uploader("Upload rule file (.txt, .pdf, .docx)", type=["txt","pdf","docx"])
if not uploaded_file:
    st.stop()

def load_file(file):
    if file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    elif file.name.endswith(".pdf"):
        reader = PdfReader(file)
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    elif file.name.endswith(".docx"):
        doc = Document(file)
        return "\n".join(p.text for p in doc.paragraphs)
    else:
        raise ValueError("Unsupported file format")

text = load_file(uploaded_file)
chunks = text.split("\n\n")  # simple paragraph chunks

if st.button("Extract Rules"):
    rules_list = []
    for chunk in chunks:
        if chunk.strip():
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role":"system","content":"Extract rules from this text into JSON array [{\"rule\": \"...\"}]"},
                          {"role":"user","content": chunk}],
                temperature=0
            )
            content = response['choices'][0]['message']['content']
            try:
                rules = json.loads(content)
                rules_list.extend(rules)
            except:
                pass  # ignore parse errors

    # Save files
    rules_json = json.dumps(rules_list, indent=2)
    checklist = "\n".join(f"[ ] {r['rule']}" for r in rules_list)

    st.download_button("Download JSON", data=rules_json, file_name="rules.json")
    st.download_button("Download Checklist", data=checklist, file_name="rules_checklist.txt")
    st.success("✅ Extraction complete!")
