import streamlit as st
from PIL import Image
import requests
from io import BytesIO
from transformers import ViltProcessor, ViltForQuestionAnswering

st.set_page_config(layout="wide", page_title="Visual QA with VILT HuggingFace")

# Vilt model
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

def get_answer(image, text):
    try:
        # load and process the image
        img = Image.open(BytesIO(image)).convert("RGB")
        # prepare input
        encoding = processor(img, text, return_tensors="pt")

        # forward pass
        outputs = model(**encoding)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        answer = model.config.id2label.get(idx, "")

    except Exception as e:
        return str(e)

    return answer

st.title("Visual QA Application")
st.write("Upload an image and ask a question about it!")

col1, col2 = st.columns(2)

# Image Upload
with col1:
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = image.convert("RGB")  # Convert the image to RGB mode
        st.image(image, use_column_width=True)

with col2:
    question = st.text_input("question")

    if uploaded_file and question is not None:
        if st.button("Ask question"):
            image = Image.open(uploaded_file)
            image = image.convert("RGB")

            image_byte_array = BytesIO()
            image.save(image_byte_array, format='jpeg')
            image_bytes = image_byte_array.getvalue()

            # get the answer
            answer = get_answer(image_bytes, question)

            # check if answer is not None before displaying
            if answer is not None:
                st.info("Your question: " + question)
                st.success("Answer: " + answer)
            else:
                st.error("Failed to get an answer. Please check your input.")
