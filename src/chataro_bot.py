import streamlit as st
import shutil
import os
import sys
from PIL import Image

from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ChatPromptTemplate, load_index_from_storage, Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex

from tools_chataro import CustomPrompt, ChataroBot
from tools_general import Categorize, GetIndex


icon = Image.open('src/chataro.png')

st.title("チャ太郎 アシスト")

with st.sidebar:
    openai_api_key = st.text_input(
        "OpenAI API Key", key="chatbot_api_key", type="password"
    )
    st.write("[Get an OpenAI API key](https://platform.openai.com/account/api-keys)")
    OPNEAI_API_KEY = openai_api_key

tab1, tab2 = st.tabs(["チャ太郎に質問する", "チャ太郎に文書を与える"])
st.session_state.subject = None

with tab1:
    Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)

    def load_index_to_engin(custom_prompt:CustomPrompt) -> VectorStoreIndex:
        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        index = load_index_from_storage(storage_context)
        query_engine = index.as_query_engine(
            similarity_top_k=5,
            text_qa_template=custom_prompt._chat_text_qa_prompt(),
            refine_template =custom_prompt._chat_refine_prompt(),
        )
        return query_engine

    def load_index() -> VectorStoreIndex:
        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        index = load_index_from_storage(storage_context)
        return index

    if "messages" not in st.session_state:
        system_prompt = (
            "あなたは数研チャ太郎です。日本語を話します。"
            "あなたの役割は、文書に関する知識に基づいてユーザーの質問に答えることです。"
            "文書の範囲外の質問の場合は、丁寧にお断りしてください。"
            "適切なお断りの文「すみません、私にはその件についての知識が不足しております。申し訳ありません。」"
        )
        first_message = "こんにちは、数研チャ太郎です！先生のアシスタントとして頑張ります、よろしくお願いいたします！"
        st.session_state['messages'] = [
            ChatMessage(content=system_prompt, role=MessageRole.SYSTEM),
            ChatMessage(content=first_message, role=MessageRole.ASSISTANT)
        ]

    #prompt_chataro = CustomPrompt(st.session_state['messages'])

    if "query_engine" not in st.session_state:
        index = load_index()
        st.session_state.query_engine = ChataroBot(index)

    for msg in st.session_state.messages:
        if msg.role == MessageRole.ASSISTANT:
            st.chat_message("assistant", avatar=icon).write(msg.content)
        elif msg.role == MessageRole.USER:
            st.chat_message("user").write(msg.content)
        else:
            continue

    prompt = st.chat_input("質問を入力してください。")
    if prompt:
        if st.session_state.subject is None:
            subject = Categorize._categorize_subject(prompt)
            if subject != "雑談":
                st.session_state.subject = subject
        else:
            subject = st.session_state.subject
        st.chat_message("user").write(prompt)
        response = st.session_state.query_engine._get_response(
            prompt, st.session_state.messages, subject
        )
        st.chat_message("assistant", avatar=icon).write(f"{response}")


with tab2:
    save_dir = "updata"
    uploaded_file = st.file_uploader("テキストファイルを選んでください", type=["txt"])
    if st.button('チャ太郎に渡す'):
        if uploaded_file is not None:
            content = uploaded_file.getvalue().decode("utf-8")
            # アップロードされたファイルを指定のディレクトリに保存
            file_path = os.path.join(save_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        if not os.path.exists("storage"):
            documents = SimpleDirectoryReader(input_dir="./updata").load_data()
            index = VectorStoreIndex.from_documents(documents)
            index.storage_context.persist()
        else:
            storage_context = StorageContext.from_defaults(persist_dir="./storage")
            index = load_index_from_storage(storage_context)
            documents = SimpleDirectoryReader("updata").load_data()
            for document in documents:
                index.insert(document)
            index.storage_context.persist()

        # ディレクトリ内のすべてのファイルおよびディレクトリの名前を取得
        for filename in os.listdir(save_dir):
            file_path = os.path.join(save_dir, filename)
            # ファイルパスがファイルである場合のみ削除
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)

        st.chat_message("assistant", avatar=icon).write("文書を受け取りました！")
