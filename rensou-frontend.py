import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def rensou(word: str):
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    
    for _ in range(5):
        inputs = tokenizer(f"if I say {word}, you say", return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=10)
        response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        # Check if the response is different from the input word
        if response.lower() != word.lower():
            return response

    return "Unable to generate a new word."


# Streamlit frontend
def main():
    # Title
    st.title('Rensou')

    # User input for the original word and the replacement word
    original_word = st.text_input("If I say")
    
    # Button to replace word
    if st.button('then you say'):
        result = rensou(original_word)
        st.success(f'{result[0]}')

if __name__ == "__main__":
    main()
