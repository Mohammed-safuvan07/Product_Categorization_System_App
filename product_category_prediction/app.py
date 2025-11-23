# import libraries

import streamlit as st
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Title
st.title('Product Category Prediction App ')

# Dataset Loading
data = pd.read_csv('category_tree_.csv')
unique_categories = data['category_path'].unique()

# Hugging Face model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Data Cleaning

def clean_text(text):
    lower_text = text.lower()
    cln_text = re.sub('[^a-zA-Z0-9]',' ',lower_text)
    return cln_text.strip()

# Embedding
category_embedding = model.encode([clean_text(i) for i in unique_categories],convert_to_tensor=False)

# predicting user query with cosine similarity

def predict_category(user_query):
    user_embedding = model.encode([user_query])[0]
    similarity = cosine_similarity([user_embedding],category_embedding)
    best_index = np.argmax(similarity)
    predicted_category = unique_categories[best_index]
    confidence = float(np.max(similarity))
    return predicted_category, confidence

# Streamlit UI
title = st.text_input('Product Title')
description = st.text_area('Product Description')
tags = st.text_input('Tags')


if st.button('Predict Category'):

    cln_title = clean_text(title)
    cln_description = clean_text(description)
    cln_tags = clean_text(tags)
    query = f"{cln_title} {cln_description} {cln_tags}"

    if query.strip() == "":
        st.error("Please enter at least one input!")
    else:
        category,confidence = predict_category(query)
        st.success(f"### Predicted Category\n{category}")
        st.info(f"### Confidence Score: {confidence:.2f}")