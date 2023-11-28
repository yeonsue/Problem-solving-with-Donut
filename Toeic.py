from dotenv import load_dotenv
import streamlit as st
import pandas as pd
from io import StringIO
from PIL import Image
from donut import DonutModel
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage
load_dotenv()
model = DonutModel.from_pretrained("finetune-model")
#이미지 입력 받기
st.title('Get Answer!')

uploaded_file = st.file_uploader("Upload the image")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    #st.write(bytes_data)
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image)

if st.button('Answer'):
    with st.spinner('정답 불러오는 중..'):
        #donut parsing
        image = Image.open(uploaded_file).convert("RGB")
        model.eval()
        output = model.inference(image=image,prompt="<s_dataset>")
        json_file=output['predictions'][0]

        #answer with openai
        answer_model = ChatOpenAI()
        question = "passage : {0}, A : {1}, B : {2}, C : {3}, D : {4}".format(json_file['passage'],json_file['A'],json_file['B'],json_file['C'],json_file['D'])
        messages = [
            SystemMessage(content="Read the passage and select the answer best fit in ------, answer with only Alphabet and word."),
            HumanMessage(content=question),
        ]

        answer = answer_model.invoke(messages)
        st.write(answer.content)
