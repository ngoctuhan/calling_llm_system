# Graph RAG v2

Graph RAG v2 là một framework Retrieval Augmented Generation (RAG) dựa trên đồ thị tri thức, được thiết kế để kết hợp các khả năng tìm kiếm ngữ nghĩa của các mô hình embedding với sức mạnh biểu diễn mối quan hệ của đồ thị tri thức (knowledge graph).

## Cấu trúc Module

Module được tổ chức thành các thành phần chính sau:

1. **GraphRAG**: Lớp chính cho phép truy vấn và tìm kiếm thông tin dựa trên đồ thị, thừa kế từ BaseRAG.
2. **GraphBuilder**: Lớp xử lý việc xây dựng đồ thị tri thức từ các tài liệu.
3. **SimpleNeo4jConnection**: Lớp xử lý kết nối và truy vấn cơ sở dữ liệu Neo4j.
4. **GraphExtractor**: Lớp trích xuất thông tin từ văn bản thành các bộ ba tri thức (knowledge triplets).

## Chức năng

### 1. GraphRAG

Tập trung vào việc truy xuất thông tin từ đồ thị tri thức:

- Kế thừa từ BaseRAG để tương thích với các phương thức RAG tiêu chuẩn
- Hỗ trợ tìm kiếm ngữ nghĩa dựa trên vector embedding
- Hỗ trợ tìm kiếm dựa trên cấu trúc đồ thị
- Kết hợp kết quả từ cả hai phương pháp để cung cấp kết quả phong phú hơn

### 2. GraphBuilder

Tập trung vào việc xây dựng và duy trì đồ thị tri thức:

- Xử lý tài liệu để trích xuất bộ ba tri thức
- Tạo embedding cho các thực thể
- Lưu trữ thông tin vào cơ sở dữ liệu Neo4j
- Hỗ trợ xử lý nhiều tài liệu đồng thời

**Lưu ý**: GraphBuilder không xử lý việc chia nhỏ tài liệu (chunking), mà xử lý trực tiếp văn bản đầu vào. Nếu cần chia nhỏ văn bản dài, hãy làm điều đó trước khi truyền vào GraphBuilder hoặc sử dụng module chunking riêng biệt.

### 3. SimpleNeo4jConnection

Quản lý kết nối và truy vấn đến Neo4j:

- Thiết lập và duy trì kết nối đến cơ sở dữ liệu Neo4j
- Thực hiện các truy vấn cơ bản và phức tạp
- Quản lý vector embedding của các thực thể
- Xử lý việc lưu trữ bộ ba tri thức

### 4. GraphExtractor

Trích xuất thông tin từ văn bản:

- Sử dụng LLM để trích xuất các bộ ba tri thức từ văn bản
- Chuyển đổi đầu ra của LLM thành các đối tượng KnowledgeTriplet
- Hỗ trợ xử lý hàng loạt các văn bản

## Ví dụ sử dụng

### Tạo đồ thị tri thức

```python
from retrieval_engine.knowledge_retrieval.graph_rag_v2 import GraphBuilder

# Khởi tạo GraphBuilder
builder = GraphBuilder(
    neo4j_uri="neo4j://localhost:7687",
    neo4j_username="neo4j",
    neo4j_password="password",
    graph_extractor_model="gpt-4o"
)

# Xử lý tài liệu
async def process():
    await builder.initialize()
    result = await builder.process_document(
        text="Albert Einstein was a German physicist who developed the theory of relativity.",
        document_id="doc1",
        document_metadata={"source": "Wikipedia"}
    )
    print(f"Processed document: {result}")
    await builder.close()

# Chạy xử lý
import asyncio
asyncio.run(process())
```

### Truy vấn đồ thị tri thức

```python
from retrieval_engine.knowledge_retrieval.graph_rag_v2 import GraphRAG

# Khởi tạo GraphRAG
rag = GraphRAG(
    neo4j_uri="neo4j://localhost:7687",
    neo4j_username="neo4j",
    neo4j_password="password"
)

# Truy vấn đồ thị
results = rag.retrieve(
    query="What did Einstein develop?",
    top_k=5,
    use_semantic=True
)

# Hiển thị kết quả
for result in results:
    print(f"- {result['text']}")
```

## Xử lý tài liệu dài

Đối với tài liệu dài, bạn có thể sử dụng module chunking riêng biệt hoặc xử lý trước khi truyền vào GraphBuilder:

```python
from retrieval_engine.knowledge_retrieval.graph_rag_v2 import GraphBuilder
from text_processing import TextChunker  # Module chia nhỏ văn bản riêng biệt

# Khởi tạo
builder = GraphBuilder(
    neo4j_uri="neo4j://localhost:7687",
    neo4j_username="neo4j",
    neo4j_password="password"
)

async def process_long_document(text, document_id, metadata=None):
    # Chia nhỏ văn bản thành các chunk
    chunker = TextChunker(chunk_size=1000, chunk_overlap=100)
    chunks = chunker.split_text(text)
    
    # Xử lý từng chunk
    results = []
    for i, chunk in enumerate(chunks):
        chunk_id = f"{document_id}_chunk_{i}"
        chunk_metadata = metadata.copy() if metadata else {}
        chunk_metadata["chunk_id"] = chunk_id
        chunk_metadata["chunk_index"] = i
        
        # Xử lý chunk bằng GraphBuilder
        result = await builder.process_document(
            text=chunk,
            document_id=chunk_id,
            document_metadata=chunk_metadata
        )
        results.append(result)
    
    return results
```

## Tài liệu API

Xem mô tả phương thức chi tiết trong docstrings của từng lớp.

## Yêu cầu

- Python 3.7+
- Neo4j database (có thể sử dụng Neo4j Desktop hoặc Neo4j Aura Cloud)
- Các thư viện Python: neo4j, openai (hoặc thư viện embedding khác được hỗ trợ)

## Lưu ý cài đặt

Để sử dụng vector search trong Neo4j, bạn cần sử dụng Neo4j 5.0+ đối với các kịch bản xử lý tiếng Việt.
