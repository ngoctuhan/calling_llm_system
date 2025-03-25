from data_ingestion.readers import CMSReader
from data_ingestion.chunkers import TextChunker, TokenTextChunker, Tokenizer
import tiktoken

# --- Ví dụ với TextChunker thông thường (dựa trên số ký tự) ---
print("--- Chunking theo số ký tự ---")

cms_reader = CMSReader(credentials={})
content = cms_reader.read("https://en.wikipedia.org/wiki/Nguy%E1%BB%85n_Tr%C3%A3i")
text = cms_reader.get_text(content)
metadata = cms_reader.get_metadata("https://en.wikipedia.org/wiki/Nguy%E1%BB%85n_Tr%C3%A3i")

char_chunker = TextChunker(min_chunk_size=500, max_chunk_size=1000, chunk_overlap=150)
char_chunks = char_chunker.chunk(text, metadata)

print(f"Số lượng chunk theo ký tự: {len(char_chunks)}")
print(f"Chunk đầu tiên: {char_chunks[0]}")
print(f"Độ dài chunk đầu tiên: {len(char_chunks[0].text)} ký tự")
print()

# --- Ví dụ với TokenTextChunker (dựa trên số token) ---
print("--- Chunking theo số token ---")

# Sử dụng tiktoken để chia nhỏ dựa trên token
encoding = tiktoken.get_encoding("cl100k_base")  # OpenAI's encoding

def encode_fn(text):
    return encoding.encode(text)

def decode_fn(tokens):
    # Để xử lý trường hợp trong constructor của TokenTextChunker
    if tokens and isinstance(tokens[0], str) and tokens[0] == "test":
        return "test string"
        
    # Đảm bảo tokens là list các số nguyên
    if isinstance(tokens, list):
        # Kiểm tra xem tokens có phải là list các số nguyên không
        if all(isinstance(token, int) for token in tokens):
            return encoding.decode(tokens)
        else:
            # Nếu không phải số nguyên, chuyển đổi
            try:
                return encoding.decode([int(token) for token in tokens if token])
            except ValueError:
                # Nếu không thể chuyển đổi, trả về chuỗi rỗng
                print(f"Warning: Could not convert tokens to int: {tokens[:5]}...")
                return ""
    else:
        # Trường hợp tokens là string hoặc kiểu dữ liệu khác
        print(f"Warning: Unexpected token type: {type(tokens)}")
        return ""

# Tạo chunker dựa trên token
token_chunker = TokenTextChunker(
    tokenizer_fn=encode_fn,
    detokenizer_fn=decode_fn,
    tokens_per_chunk=1000,  # Số lượng token tối đa mỗi chunk
    chunk_overlap=200  # Số token chồng lấp
)

# Chia văn bản thành các chunk theo token
token_chunks = token_chunker.chunk(text, metadata)

print(f"Số lượng chunk theo token: {len(token_chunks)}")
if token_chunks:
    print(f"Chunk đầu tiên: {token_chunks[0].text}")
    print(f"Số token trong chunk đầu tiên: {token_chunks[0].token_count}")
print()

# --- Ví dụ với nhiều văn bản ---
print("--- Chunking nhiều văn bản ---")

# Thêm một văn bản khác để demo chunking nhiều văn bản
content2 = cms_reader.read("https://vi.wikipedia.org/wiki/Lê_Lợi")
text2 = cms_reader.get_text(content2)
metadata2 = cms_reader.get_metadata("https://vi.wikipedia.org/wiki/Lê_Lợi")

# Chunking nhiều văn bản với TokenTextChunker
multi_chunks = token_chunker.chunk_multiple_texts(
    texts=[text, text2],
    metadata=[metadata, metadata2]
)

print(f"Tổng số chunk từ nhiều văn bản: {len(multi_chunks)}")
if len(multi_chunks) > 10:
    print(f"Nguồn gốc của chunk thứ 10: Văn bản {multi_chunks[10].source_indices[0]}")
print()

# --- Sử dụng hàm tiện ích ---
print("--- Sử dụng hàm tiện ích chunking ---")

from data_ingestion.chunkers import chunk_with_tokenizer, chunk_multiple_texts_with_tokenizer

# Tạo tokenizer
custom_tokenizer = Tokenizer(
    chunk_overlap=50,
    tokens_per_chunk=300,
    decode=decode_fn,
    encode=encode_fn
)

# Chunking một văn bản
simple_chunks = chunk_with_tokenizer(text, custom_tokenizer)
print(f"Số lượng chunk từ hàm tiện ích: {len(simple_chunks)}")

# Chunking nhiều văn bản với theo dõi nguồn gốc
utility_multi_chunks = chunk_multiple_texts_with_tokenizer(
    texts=[text, text2],
    tokenizer=custom_tokenizer
)
print(f"Số lượng chunk từ nhiều văn bản: {len(utility_multi_chunks)}")
if utility_multi_chunks:
    print(f"Văn bản gốc của chunk cuối: Văn bản {utility_multi_chunks[-1].source_indices}")