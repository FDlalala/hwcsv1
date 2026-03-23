import warnings
warnings.filterwarnings("ignore")
import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# ========== 第一步：读取 JSON 案例库 ==========
CASE_JSON_PATH = "./cases.json"  # 改成你的 JSON 文件路径

print("读取案例库 JSON...")
with open(CASE_JSON_PATH, "r", encoding="utf-8") as f:
    case_data = json.load(f)

documents = []
skipped = []

for case_id, case_info in case_data.items():
    case_name = case_info.get("case_name", "")
    text_list = case_info.get("text", [])

    # 将 text 列表拼接成完整文本（每段之间用换行分隔）
    full_text = "\n".join(text_list).strip()

    if full_text:
        documents.append(Document(
            page_content=full_text,
            metadata={
                "case_id": case_id,
                "case_name": case_name,
                "source": CASE_JSON_PATH
            }
        ))
        print(f"  ✅ [{case_id}] {case_name[:40]:<40} ({len(full_text):>6} 字符)")
    else:
        skipped.append(case_id)
        print(f"  ⏭️  跳过空案例: {case_id}")

print(f"\n成功加载: {len(documents)} 个案例")
print(f"跳过/失败: {len(skipped)} 个案例")

# ========== 第二步：写入向量库（每个案例即一个 chunk）==========
print("\n加载 Embedding 模型...")
embeddings = HuggingFaceEmbeddings(model_name="./bge-small-zh-v1.5")

print("清空旧知识库，重新构建...")
vectordb = Chroma(
    embedding_function=embeddings,
    persist_directory="./chroma_cases"
)
vectordb.delete_collection()

vectordb = Chroma(
    embedding_function=embeddings,
    persist_directory="./chroma_cases"
)

# 批量写入
batch_size = 100
total = len(documents)
for i in range(0, total, batch_size):
    batch = documents[i:i+batch_size]
    vectordb.add_documents(batch)
    print(f"  写入进度: {min(i+batch_size, total)}/{total} ({min(i+batch_size, total)*100//total}%)")

final_count = vectordb._collection.count()
print(f"\n✅ 案例知识库构建完成！共 {final_count} 条记录")
