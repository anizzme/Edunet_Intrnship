import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf=PdfReader(file)
    text= ""
    for page in pdf.pages:
        text += page.extract_text()
        
#Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    job_description = str(job_description) if job_description else ""

    # Filter out None or empty resumes
    resumes = [str(resume) for resume in resumes if resume]

    # If no valid resumes exist, return an empty list
    if not resumes:
        print("Error: No valid resumes found!")
        return []
    # Combine job description with resumes
    documents=[str(job_description)] + [str(resume) for resume in resumes if resume is not None]
    vectorizer=TfidfVectorizer().fit_transform(documents)
    vectors=vectorizer.toarray()
    
    #calculate cosine similarity
    job_description_vector=vectors[0]
    resume_vectors=vectors[1:]
    cosine_similarities=cosine_similarity([job_description_vector],resume_vectors).flatten()
    
    return cosine_similarities

#streamlit app
st.title("AI-powered Resume Ranking and Screening System")

#Job description input
st.header("Job Description")
job_description=st.text_area("Enter the job description")

#File uploader
st.header("Upload Resumes")
uploaded_files=st.file_uploader("Upload PDF files", type=["pdf"],accept_multiple_files=True)

if uploaded_files and job_description:
    st.header("Ranking Resumes")
    
    resumes=[]
    for file in uploaded_files:
        text=extract_text_from_pdf(file)
        resumes.append(text)
        
    #Rank resumes
    scores=rank_resumes(job_description,resumes)
    
    #Display scores
    results=pd.DataFrame({"Resume":[file.name for file in uploaded_files],"Score":scores})
    results=results.sort_values(by="Score",ascending=False)
    
    st.write(results)
    
    