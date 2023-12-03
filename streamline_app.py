import streamlit as st
import PyPDF2
import spacy
import io

# Download the spaCy model data
spacy.cli.download("en_core_web_sm")

def read_pdf(file_content):
    with io.BytesIO(file_content) as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
    return text

def compare_job_and_resume(job_description, resume_text):
    nlp = spacy.load('en_core_web_sm')
    job_description_doc = nlp(job_description)
    resume_text_doc = nlp(resume_text)
    similarity_score = job_description_doc.similarity(resume_text_doc)
    return similarity_score * 100

def suggest_skills(job_description, resume_text):
    nlp = spacy.load('en_core_web_sm')
    job_description_doc = nlp(job_description)
    resume_text_doc = nlp(resume_text)

    # Identify potential skills based on POS tagging (verbs and nouns)
    job_skills = set(token.text.lower() for token in job_description_doc if token.pos_ in ['VERB', 'NOUN'])

    # Get the skills that are in the job description but not in the resume
    missing_skills = job_skills - set(token.text.lower() for token in resume_text_doc if token.is_alpha)

    return missing_skills

st.title("Resume Analyzer")

job_description = st.text_area("Enter the job description:")
resume_file = st.file_uploader("Upload Resume PDF", type=["pdf"])

if job_description and resume_file:
    resume_text = read_pdf(resume_file.read())
    similarity_score = compare_job_and_resume(job_description, resume_text)
    st.write(f"Similarity Score: {similarity_score:.2f}%")

    if similarity_score < 100:
        suggested_skills = suggest_skills(job_description, resume_text)
        st.write("Suggested Skills to Improve Score:")
        st.write(suggested_skills)
