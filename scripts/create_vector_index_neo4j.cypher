// create vector index
CREATE VECTOR INDEX `entity_embedding`
FOR (n:Entity) ON (n.embedding)
OPTIONS {indexConfig: {
 `vector.dimensions`: 768,
 `vector.similarity_function`: 'cosine'
}};


CREATE VECTOR INDEX `relationship_embedding`
FOR (n:Entity) ON (n.embedding)
OPTIONS {indexConfig: {
 `vector.dimensions`: 768,
 `vector.similarity_function`: 'cosine'
}};