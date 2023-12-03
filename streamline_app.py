import os
import re
import ssl
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import streamlit as st
import PyPDF2
import pdfplumber
import tensorflow as tf
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Define skill keywords and their weights for each job field
skill_keywords_dict = {
    "Data Science": {
    "machine learning": 0.8,
    "Python": 0.7,
    "data analysis": 0.6,
    "statistical modeling": 0.5,
    "deep learning": 0.7,
    "data visualization": 0.6,
    "SQL": 0.5,
    "R": 0.4,
    "big data": 0.6,
    "natural language processing": 0.7,
    "data mining": 0.6,
    "data cleansing": 0.5,
    "feature engineering": 0.6,
    "time series analysis": 0.6,
    "predictive modeling": 0.7,
    "data preprocessing": 0.5,
    "Hadoop": 0.4,
    "Spark": 0.4,
    # Add more relevant keywords and weights as needed
},
    "Software Developer":{
    "programming": 0.8,
    "Java": 0.7,
    "C++": 0.6,
    "software development": 0.7,
    "coding": 0.6,
    "algorithm": 0.6,
    "web development": 0.5,
    "debugging": 0.5,
    "API": 0.4,
    "software engineering": 0.7,
    "agile methodology": 0.6,
    "version control": 0.6,
    "object-oriented programming": 0.7,
    "unit testing": 0.6,
    "database management": 0.5,
    "cloud computing": 0.5,
    "continuous integration": 0.6,
    "problem-solving": 0.7,
    # Add more relevant keywords and weights as needed
} ,
    "Cybersecurity": {
    "cybersecurity": 0.8,
    "network security": 0.7,
    "firewall": 0.6,
    "penetration testing": 0.7,
    "incident response": 0.6,
    "security audit": 0.5,
    "vulnerability assessment": 0.5,
    "encryption": 0.4,
    "ethical hacking": 0.7,
    "SIEM (Security Information and Event Management)": 0.6,
    "threat intelligence": 0.6,
    "identity and access management": 0.6,
    "cybersecurity policies": 0.5,
    "risk management": 0.5,
    "security awareness training": 0.4,
    "malware analysis": 0.6,
    "network monitoring": 0.7,
    # Add more relevant keywords and weights as needed
},
    "Web Developer": {
    "web development": 0.8,
    "HTML": 0.7,
    "CSS": 0.6,
    "JavaScript": 0.7,
    "front-end development": 0.6,
    "back-end development": 0.6,
    "React": 0.5,
    "Angular": 0.5,
    "Node.js": 0.4,
    "responsive design": 0.6,
    "RESTful API": 0.6,
    "web frameworks": 0.7,
    "UI/UX design": 0.6,
    "cross-browser compatibility": 0.5,
    "web security": 0.5,
    "performance optimization": 0.6,
    "version control (e.g., Git)": 0.7,
    "web hosting": 0.5,
    "content management systems": 0.5,
    # Add more relevant keywords and weights as needed
},
}
email_regex = r'\S+@\S+'
phone_regex = r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}'
# Define your existing functions here (e.g., extract_text_from_pdf, process_resumes_for_field, etc.)
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    doc = nlp(text)
    tokens = [token.text for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

# Function to calculate cosine similarity between two texts with weighted skills
def calculate_similarity(job_description, resume):
    vectorizer = CountVectorizer().fit_transform([job_description, resume])
    vectors = vectorizer.toarray()
    return cosine_similarity([vectors[0]], [vectors[1]])[0][0]

# Function to suggest missing skills based on weighted skills
def suggest_missing_skills(job_description, resume):
    # Preprocess job description and resume
    job_description = preprocess_text(job_description)
    resume_text = preprocess_text(resume)

    # Create a set of unique skills in the job description
    job_skills = set(job_description.split())

    # Calculate missing skills based on weighted skills
    missing_skills = []
    for skill, weight in skill_keywords.items():
        if skill in job_skills and skill not in resume_text:
            missing_skills.append(skill)

    return missing_skills

def extract_information(text):
    email = ""
    phone = ""

    # Find emails and phone numbers
    emails = re.findall(email_regex, text)
    phones = re.findall(phone_regex, text)

    # Extract the first email and phone number found (you can modify this logic as needed)
    if emails:
        email = emails[0]
    if phones:
        phone = phones[0]

    return email, phone

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to process resumes for a selected field
def process_resumes_for_field(selected_field, job_description, resumes_folder, count, sender_email, sender_password, email_subject, email_message):
    selected_resumes = []
    rejected_resumes = []

    # Retrieve the skill keywords and weights for the selected field
    skill_keywords = skill_keywords_dict.get(selected_field, {})

    for resume_file in os.listdir(resumes_folder):
        if resume_file.endswith(".pdf"):
            resume_path = os.path.join(resumes_folder, resume_file)
            resume_text = extract_text_from_pdf(resume_path)

            # Preprocess the extracted text
            resume_text = resume_text.strip().lower()

            # Tokenize and pad the resume text
            tokenizer = Tokenizer(oov_token="<OOV>")
            tokenizer.fit_on_texts([resume_text])
            vocab_size = len(tokenizer.word_index) + 1
            max_seq_length = 1159
            padded_sequence = pad_sequences(tokenizer.texts_to_sequences([resume_text]), maxlen=max_seq_length, padding="post")

            # Make predictions using the loaded model
            predictions = model.predict(padded_sequence)

            # Extract email and phone information using the loaded model
            email, phone = extract_information(resume_text)

            # Calculate similarity score based on selected field's skill keywords and weights
            similarity_score = calculate_similarity(job_description, resume_text)

            # Check if the resume is selected or rejected
            if len(selected_resumes) < count:
                selected_resumes.append({"resume_file": resume_file, "similarity_score": similarity_score, "email": email, "phone": phone})
            else:
                # Check if the current resume has a higher similarity score than the lowest in selected resumes
                min_similarity_score = min(selected_resumes, key=lambda x: x["similarity_score"])["similarity_score"]
                if similarity_score > min_similarity_score:
                    # Replace the lowest similarity score resume with the current resume
                    lowest_sim_index = selected_resumes.index(next(item for item in selected_resumes if item["similarity_score"] == min_similarity_score))
                    selected_resumes[lowest_sim_index] = {"resume_file": resume_file, "similarity_score": similarity_score, "email": email, "phone": phone}
                else:
                    # Add the current resume to the rejected list
                    rejected_resumes.append({"resume_file": resume_file, "similarity_score": similarity_score, "email": email, "phone": phone})

    # Sort selected resumes by similarity score in descending order
    selected_resumes.sort(key=lambda x: x["similarity_score"], reverse=True)
    
    # Sort rejected resumes by similarity score in descending order
    rejected_resumes.sort(key=lambda x: x["similarity_score"], reverse=True)
    
    # Send emails to selected resumes and display "Rejected" for rejected ones
    st.markdown(f"### Selected Resumes for {selected_field} :")
    for resume_info in selected_resumes:
        resume_file = resume_info["resume_file"]
        similarity_score = resume_info["similarity_score"]
        email = resume_info["email"]
        phone = resume_info["phone"]
        
        st.write(f"**Selected Resume:** {resume_file}")
        st.write(f"**Similarity Score:** {similarity_score * 100:.2f}%")
        st.write(f"**Email:** {email}")
        st.write(f"**Phone:** {phone}")
        st.write("***")  # Add a line separator after each resume

    st.markdown(f"### Rejected Resumes for {selected_field} :")
    for resume_info in rejected_resumes:
        resume_file = resume_info["resume_file"]
        similarity_score = resume_info["similarity_score"]
        email = resume_info["email"]
        phone = resume_info["phone"]
        
        st.write(f"**Rejected Resume:** {resume_file}")
        st.write(f"**Similarity Score:** {similarity_score * 100:.2f}%")
        st.write(f"**Email:** {email}")
        st.write(f"**Phone:** {phone}")
        st.write("***")  # Add a line separator after each resume

def send_email(sender_email, sender_password, recipient_email, subject, message):
    try:
        context = ssl.create_default_context()

        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender_email, sender_password)
            msg = MIMEMultipart()
            msg["From"] = sender_email
            msg["To"] = recipient_email
            msg["Subject"] = subject
            msg.attach(MIMEText(message, "plain"))

            server.sendmail(sender_email, recipient_email, msg.as_string())
            print(f"Email sent to {recipient_email} successfully!")

    except smtplib.SMTPAuthenticationError as e:
        print(f"SMTP Authentication Error: {str(e)}")
    except smtplib.SMTPException as e:
        print(f"SMTP Exception: {str(e)}")
    except Exception as e:
        print(f"Error sending email: {str(e)}")


# Load the trained model for email and phone extraction
model_path = "/Users/suka/Downloads/Resume_deployment"
model = tf.keras.models.load_model(model_path)

# Main Streamlit app
def main():
    st.title("Resume Processing App")

    selected_field = st.selectbox("Select a field", ["Web Developer", "Data Science", "Cybersecurity", "Software Developer"])
    job_description = st.text_area("Enter the job description")
    resumes_folder = st.text_input("Enter the folder path containing resumes")
    count = st.number_input("Enter the number of top matching resumes to select", min_value=1)
    sender_email = st.text_input("Enter your email address (sender)")
    sender_password = st.text_input("Enter your email password (sender)", type="password")
    email_subject = st.text_input("Enter the email subject")
    email_message = st.text_area("Enter the email message")

    if st.button("Process Resumes"):
        # Call your existing code here with the provided inputs, including the loaded model
        process_resumes_for_field(selected_field, job_description, resumes_folder, count, sender_email, sender_password, email_subject, email_message)

if __name__ == "__main__":
    main()
