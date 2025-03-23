from data_ingestion.readers import CMSReader

cms_reader = CMSReader(credentials={})

content = cms_reader.read("https://vi.wikipedia.org/wiki/Nguy%E1%BB%85n_Tr%C3%A3i")

text = cms_reader.get_text(content)
metadata = cms_reader.get_metadata("https://vi.wikipedia.org/wiki/Nguy%E1%BB%85n_Tr%C3%A3i")

from data_ingestion.chunkers import TextChunker

chunker = TextChunker(min_chunk_size=350, max_chunk_size=600, chunk_overlap=100)
chunks = chunker.chunk(text, metadata)
print(chunks[0])
print(chunks[1])
print(chunks[2])


print(len(chunks))