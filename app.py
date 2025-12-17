import streamlit as st
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
    
@st.cache_resource
def load_system():  
    try:  
        required_files = ['knowledge_base.pkl', 'ecommerce_index.faiss', 'model_name.txt']  
        for file in required_files:  
            if not os.path.exists(file):
                st.error(f"Missing: {file}")  
                return None, None, None 
        with open('model_name.txt', 'r') as f: 
            model_name = f.read().strip() 
            model = SentenceTransformer(model_name) 
        with open('knowledge_base.pkl', 'rb') as f: 
            knowledge_base = pickle.load(f) 
        index = faiss.read_index('ecommerce_index.faiss') 
        return model, knowledge_base, index 
    except Exception as e: 
        st.error(f"Error: {str(e)}") 
        return None, None, None 

def get_answer(query, model, knowledge_base, index):  
    try:
        query_embedding = model.encode([query])  
        faiss.normalize_L2(query_embedding) 
        scores, indices = index.search(query_embedding.astype('float32'), 3) 
        best_idx = indices[0][0] 
        best_score = scores[0][0] 
        if best_score < 0.3: 
            return get_fallback(query) 
        best_match = knowledge_base[best_idx] 
        return { 
            'answer': best_match['answer'], 
            'confidence': "High" if best_score > 0.7 else "Medium" 
        } 
    except Exception as e: 
        return {'answer': f"Sorry, error occurred: {str(e)}", 'confidence': 'Low'} 

def main():
    st.title("💬 Customer Support") 
    model, knowledge_base, index = load_system() 
    if not all([model, knowledge_base, index]): 
        st.stop() 
    if "messages" not in st.session_state: 
        st.session_state.messages = []
    # Display chat 
    for message in st.session_state.messages: 
        with st.chat_message(message["role"]): 
            st.write(message["content"]) 

    # Chat input 
    if prompt := st.chat_input("Ask me anything..."): 
        st.session_state.messages.append({"role": "user", "content": prompt}) 
    
        with st.chat_message("user"): 
            st.write(prompt) 
    
        with st.chat_message("assistant"): 
            response = get_answer(prompt, model, knowledge_base, index) 
            st.write(response['answer']) 
    
            st.session_state.messages.append({ 
                "role": "assistant", 
                "content": response['answer'] 
            })

if __name__ == "__main__":
    main()
