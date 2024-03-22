from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ChatPromptTemplate
from llama_index.llms.openai import OpenAI
from llama_index.core.chat_engine.condense_plus_context import CondensePlusContextChatEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter, FilterCondition
from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters

class CustomPrompt:
    def __init__(self, past_contexts:list=None):
        """
        カスタムプロンプトを保持するクラス
        チャットQAプロンプトとチャットRefineプロンプトを保持する
        """
        TEXT_QA_SYSTEM_PROMPT = ChatMessage(
            content=(
                "あなたは信頼された教員用のQAシステム、数研チャ太郎です。\n"
                "事前知識ではなく、常に提供されたコンテキスト情報を使用してクエリに回答してください。\n"
                "従うべきいくつかのルール:\n"
                "1. 回答内で指定されたコンテキストを直接参照しないでください。\n"
                "2. 「コンテキストに基づいて、...」や「コンテキスト情報は...」、またはそれに類するような記述は避けてください。\n"
                "3. 文書の範囲外の質問には「申し訳ありません、まだそれについての知識をもっていません。」と回答してください。"
            ),
            role=MessageRole.SYSTEM,
        )

        past_contexts = "\n".join([f'"{message["role"]}":"{message["content"]}"'for message in past_contexts[1:]])

        # QAプロンプトテンプレートメッセージ
        TEXT_QA_PROMPT_TMPL_MSGS = [
            TEXT_QA_SYSTEM_PROMPT,
            ChatMessage(
                content=(
                    "過去のコンテキスト情報は以下のとおりです。\n"
                    "---------------------\n"
                    ""
                    "コンテキスト情報は以下のとおりです。\n"
                    "---------------------\n"
                    "{context_str}\n"
                    "---------------------\n"
                    "事前知識ではなくコンテキスト情報を考慮して、クエリに答えます。\n"
                    "Query: {query_str}\n"
                    "Answer: "
                ),
                role=MessageRole.USER,
            ),
        ]

        # チャットQAプロンプト
        CHAT_TEXT_QA_PROMPT = ChatPromptTemplate(message_templates=TEXT_QA_PROMPT_TMPL_MSGS)

        # チャットRefineプロンプトテンプレートメッセージ
        CHAT_REFINE_PROMPT_TMPL_MSGS = [
            ChatMessage(
                content=(
                    "あなたは、既存の回答を改良する際に2つのモードで厳密に動作するQAシステムのエキスパートです。\n"
                    "1. 新しいコンテキストを使用して元の回答を**書き直す**。\n"
                    "2. 新しいコンテキストが役に立たない場合は、元の回答を**繰り返す**。\n"
                    "回答内で元の回答やコンテキストを直接参照しないでください。\n"
                    "疑問がある場合は、元の答えを繰り返してください。"
                    "New Context: {context_msg}\n"
                    "Query: {query_str}\n"
                    "Original Answer: {existing_answer}\n"
                    "New Answer: "
                ),
                role=MessageRole.USER,
            )
        ]

        # チャットRefineプロンプト
        CHAT_REFINE_PROMPT = ChatPromptTemplate(message_templates=CHAT_REFINE_PROMPT_TMPL_MSGS)

        self.CHAT_TEXT_QA_PROMPT = CHAT_TEXT_QA_PROMPT
        self.CHAT_REFINE_PROMPT  = CHAT_REFINE_PROMPT

    def _chat_text_qa_prompt(self):
        return self.CHAT_TEXT_QA_PROMPT

    def _chat_refine_prompt(self):
        return self.CHAT_REFINE_PROMPT


class ChataroBot:
    def __init__(self, index):
        self.index = index

    def _get_response(self, query_content:str, chat_history:list, subject:str) -> str:
        llm = OpenAI(temperature=0, model="gpt-3.5-turbo", max_tokens=1024)
        filters = MetadataFilters(
            filters=[
                #MetadataFilter(key="subject", value=subject, operator="in"),
                ExactMatchFilter(key="subject", value=subject),
            ],
            condition="or"
        )

        chat_engine = CondensePlusContextChatEngine.from_defaults(
            self.index.as_retriever(similarity_top_k=5, filters=filters,),
            llm = llm,
            system_prompt="あなたは信頼された教員用のQAシステム、数研チャ太郎です。\n"
                "事前知識ではなく、常に提供されたコンテキスト情報を使用してクエリに回答してください。\n"
                "従うべきいくつかのルール:\n"
                "1. 回答内で指定されたコンテキストを直接参照しないでください。\n"
                "2. 「コンテキストに基づいて、...」や「コンテキスト情報は...」、またはそれに類するような記述は避けてください。\n"
                "3. 文書の範囲外の質問には「申し訳ありません、まだそれについての知識をもっていません。」と回答してください。",
            chat_history=chat_history,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
            verbose=True,
        )
        response = chat_engine.chat(query_content)
        return response
