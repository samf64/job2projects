#Overcommenting to make sure I learn these concepts :)

# Import libraries
import json  # to load and work with the project database (projects.json)
from sentence_transformers import SentenceTransformer, util  # for embedding text and computing similarity
import spacy  # NLP library for extracting keywords
import os

# Load spaCy English model
# en_core_web_sm is a small, fast English model for tokenization, POS tagging, and named entity recognition
nlp = spacy.load("en_core_web_sm")

# Construct the path to projects.json relative to this script
# This ensures the file can be found no matter what folder the script is run from
json_path = os.path.join(os.path.dirname(__file__), "projects.json")

# Load the project database from a JSON file
# This file contains project ideas with tags for matching
with open(json_path) as f:
    projects = json.load(f)

# Load a sentence transformer model
# This converts text into embeddings (vectors) to compare similarity
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to extract keywords from a job description using spaCy
def extract_keywords(text):
    """
    Extracts keywords from text using spaCy.
    Only alphabetic tokens are kept (ignores punctuation/numbers).
    Returns a single string for embedding.
    """
    doc = nlp(text)           
    keywords = []             
    for token in doc:         
        if token.is_alpha:    
            keywords.append(token.text.lower())  
    return " ".join(keywords)  

# Function to recommend top projects based on job description
def recommend_projects(job_description, top_k=3):
    # Extract keywords from job description
    job_keywords = extract_keywords(job_description)
    # Convert keywords into an embedding vector
    job_embedding = model.encode(job_keywords, convert_to_tensor=True)

    recommendations = []  # list to store similarity scores and project info
    # Compare job description embedding with each project’s tags
    for project in projects:
        # Convert project tags into a string and then into an embedding
        project_text = " ".join(project["tags"])
        project_embedding = model.encode(project_text, convert_to_tensor=True)
        # Compute cosine similarity between job description and project
        score = util.cos_sim(job_embedding, project_embedding).item()
        # Store score and project info
        recommendations.append((score, project))

    # Sort projects by similarity score (highest first)
    recommendations.sort(reverse=True, key=lambda x: x[0])
    # Return only the top_k projects
    return [proj for _, proj in recommendations[:top_k]]

# Main block: run only when this file is executed directly
if __name__ == "__main__":
    # Prompt user to paste a job description
    jd = input("Paste job description here:\n")
    # Get top project recommendations
    top_projects = recommend_projects(jd)
    # Print the results
    print("\nTop Project Recommendations:\n")
    for i, proj in enumerate(top_projects, 1):
        print(f"{i}. {proj['title']}: {proj['description']}")
        print(f"{proj['tags']}")
