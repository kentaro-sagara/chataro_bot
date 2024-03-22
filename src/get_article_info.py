import os
import json
from tools_general import GetIndex

folder_path = "data"
articles_list = []

for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)

        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

            article_dict = {
                "title": lines[0].strip().replace("\u3000", "").replace("\u200b", ""),
                "subject": lines[1].strip().replace("\u3000", "").replace("\u200b", ""),
                "author": lines[2].strip().replace("\u3000", "").replace("\u200b", ""),
                "year": lines[3].strip().replace("\u3000", "").replace("\u200b", ""),
                "article": "".join(lines[4:]).strip().replace("\u3000", "").replace("\u200b", "")
            }

            articles_list.append(article_dict)

articles_info = {'article_list': articles_list}

output_file_path = os.path.join(folder_path, 'article_info.json')

with open(output_file_path, 'w', encoding='utf-8') as json_file:
    json.dump(articles_info, json_file, indent=4, ensure_ascii=False)

GetIndex()._get_index()