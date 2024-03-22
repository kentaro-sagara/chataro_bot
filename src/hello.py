from openai import OpenAI
import streamlit as st
from PIL import Image

icon = Image.open('chataro.png')

with st.sidebar:
    openai_api_key = st.text_input(
        "OpenAI API Key", key="chatbot_api_key", type="password"
    )
    st.write("[Get an OpenAI API key](https://platform.openai.com/account/api-keys)")

st.title("Hello Streamlit!")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "何か気になることはありますか？"}]

for msg in st.session_state.messages:
    if msg["role"] == "assistant":
        st.chat_message(msg["role"], avatar=icon).write(msg["content"])
    elif msg["role"] == "user":
        st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    client = OpenAI(
        # This is the default and can be omitted
        api_key=openai_api_key,
    )
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    response = client.chat.completions.create(
        messages=st.session_state.messages,
        model="gpt-3.5-turbo",
    )
    msg = response.choices[0].message
    st.session_state.messages.append({"role": "assistant", "content": msg.content})
    st.chat_message("assistant", avatar=icon).write(msg.content)
