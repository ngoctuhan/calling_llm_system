from data_ingestion.readers import CMSReader

cms_reader = CMSReader(credentials={})

content = cms_reader.read("https://vi.wikipedia.org/wiki/Nguy%E1%BB%85n_Tr%C3%A3i")

text = cms_reader.get_text(content)
metadata = cms_reader.get_metadata("https://vi.wikipedia.org/wiki/Nguy%E1%BB%85n_Tr%C3%A3i")

from data_ingestion.chunkers import TextChunker

chunker = TextChunker(min_chunk_size=500, max_chunk_size=1000, chunk_overlap=150)
chunks = chunker.chunk(text, metadata)

print(chunks[0])
print(chunks[1])
print(chunks[2])
print(chunks[4])
print(chunks[5])
print(len(chunks))