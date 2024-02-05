# Import Streamlit
import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def rensou(word):
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

    inputs = tokenizer(f"if I say {word}, you say what?", return_tensors="pt")
    outputs = model.generate(**inputs)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True, max_new_tokens=10)

# Streamlit frontend
def main():
    # Title
    st.title('Rensou')

    # User input for the original word and the replacement word
    original_word = st.text_input("Enter the original word:")
    
    # Button to replace word
    if st.button('Rensou'):
        result = rensou(original_word)
        st.success(f'{result[0]}')

if __name__ == "__main__":
    main()
