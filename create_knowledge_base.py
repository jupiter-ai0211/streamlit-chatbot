import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from datasets import load_dataset
import os

# Write the code below
def create_ecommerce_knowledge_base(): 
    print("Loading Bitext e-commerce dataset...") 
    # Load the dataset 
    dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset") 
    print(f"Dataset loaded: {len(dataset['train'])} examples") 

    knowledge_base = [] 
    for example in dataset['train']: 
        # Clean the response 
        response = example['response'] 
        response = response.replace("{{Order Number}}", "your order") 
        response = response.replace("{{Online Company Portal Info}}", "our website") 
        response = response.replace("{{Online Order Interaction}}", "Order History") 
        response = response.replace("{{Customer Support Hours}}", "business hours") 
        response = response.replace("{{Customer Support Phone Number}}", "our support line") 
        response = response.replace("{{Website URL}}", "our website") 
    
        knowledge_base.append({ 
            'question': example['instruction'], 
            'answer': response, 
            'intent': example['intent'], 
            'category': example['category'] 
        }) 
    
    print(f"Knowledge base created with {len(knowledge_base)} entries")

    print("Loading sentence transformer model...") 
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    print("Creating embeddings...") 
    questions = [item['question'] for item in knowledge_base] 
    embeddings = model.encode(questions, show_progress_bar=True) 

    print("Creating FAISS index...") 
    dimension = embeddings.shape[1] 
    index = faiss.IndexFlatIP(dimension)  # Inner Product for similarity 
    
    # Normalize embeddings for cosine similarity 
    faiss.normalize_L2(embeddings) 
    index.add(embeddings.astype('float32'))
    # Save everything 

    print("Saving knowledge base and index...") 
    
    # Save knowledge base 
    with open('knowledge_base.pkl', 'wb') as f: 
        pickle.dump(knowledge_base, f) 
    
    # Save FAISS index 
    faiss.write_index(index, 'ecommerce_index.faiss') 
    
    # Save model name for later loading 
    with open('model_name.txt', 'w') as f: 
        f.write('sentence-transformers/all-MiniLM-L6-v2') 
    
    print("Knowledge base created successfully!") 
    print("Files created:") 
    print("- knowledge_base.pkl") 
    print("- ecommerce_index.faiss") 
    print("- model_name.txt") 

def get_fallback(query): 
    query_lower = query.lower() 
    responses = { 
        'track': "Track your order in 'My Account' > 'Order History'", 
        'return': "We offer 30-day returns. Start in your account.", 
        'refund': "Refunds process in 5-7 business days", 
        'cancel': "Cancel orders within 1 hour in your account", 
        'shipping': "Standard: 3-5 days, Express: 1-2 days", 
        'payment': "We accept cards, PayPal, Apple Pay, Google Pay" 
    } 
    for keyword, response in responses.items(): 
        if keyword in query_lower: 
            return {'answer': response, 'confidence': 'Medium'} 
    return {
        'answer': "I'm here to help! Ask about orders, shipping, returns, or payments.",
        'confidence': 'Low'
    }

# Don't remove the following code
if __name__ == "__main__":
    create_ecommerce_knowledge_base()
