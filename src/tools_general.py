import logging
from time import sleep
import json
import os
from openai import OpenAI as OpenAI_client

from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import Document

class Categorize():
    def __init__(cls):
        None

    @staticmethod
    def _categorize_subject(query_content):
        openai_client = OpenAI_client()
        message_list = [{
            "role": "user", "content": query_content
        }]
        functions = [
            {
                "name": "categorize_query",
                "description": "質問の内容から、質問を'国語','数学','英語','理科','社会','情報'のいずれかのカテゴリーに分類する。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subject": {
                            "type": "string",
                            "description": "質問の分類結果('国語','数学','英語','物理','化学','生物','地学','地理','歴史','公民','情報'のいずれか)",
                        }
                    },
                    "required": ["subject"],
                },
            }
        ]
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=message_list,
            temperature=0,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stream=False,
            max_tokens=128,
            functions=functions,
            function_call="auto"
        )
        response_args = response.choices[0].message.function_call
        if response_args is None:
            return "雑談"
        else:
            return json.loads(response_args.arguments)["subject"]


class GetIndex():
    def __init__(cls):
        None

    @staticmethod
    def _get_index():
        dir_path = os.path.dirname(os.path.realpath(__file__)).replace("/src", "")

        with open(os.path.join(dir_path, "data", "article_info.json")) as f:
            article_json = json.load(f)
        # add a new field to each film in the json. type = "film"
        for i in range(len(article_json["article_list"])):
            article_json["article_list"][i]["type"] = "article"

        document_list = []
        for article in article_json["article_list"]:
            document = Document(
                text=article["article"],
                metadata={key: article[key] for key in article if key != "article"},
            )
            document_list.append(document)

        pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(chunk_size=512, chunk_overlap=0)
            ]
        )

        nodes = pipeline.run(documents=document_list)

        index = VectorStoreIndex(nodes)
        index.storage_context.persist(persist_dir=os.path.join(dir_path, "storage"))
