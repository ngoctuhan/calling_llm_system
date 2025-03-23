# API Documentation: Ingest URL

## /api/v1/ingestion/ingest/url

**Description:** Ingests content from a specified URL into both the vector database and the graph database. This endpoint processes the content of the URL, splits it into chunks, and stores the chunks in the databases, allowing for later retrieval and analysis based on semantic similarity and relationships.

**Method:** `POST`

**Endpoint:** `http://127.0.0.1:8000/api/v1/ingestion/ingest/url`

**Headers:**

*   `accept: application/json`  (Indicates that the client expects a JSON response)
*   `Content-Type: application/json` (Indicates that the request body is in JSON format)

**Request Body (JSON):**

The request body is a JSON object with the following properties:

| Property          | Type      | Required | Description                                                                                                                                                              | Default |
| ----------------- | --------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :-----: |
| `url`             | string    | Yes      | The URL of the web page to ingest.                                                                                                                                      |   N/A   |
| `collection_name` | string    | Yes      | The name of the collection to store the ingested data in.  Think of this like a table name or a namespace for the data.                                             |   N/A   |
| `chunk_size`      | integer   | Yes      | The maximum number of tokens/characters in each chunk of text.  This determines how the text from the URL is split for processing and storage.                                  |   N/A   |
| `chunk_overlap`   | integer   | Yes      | The number of overlapping tokens/characters between consecutive chunks. Overlap helps to maintain context across chunks.                                                      |   N/A   |
| `credentials`     | object    | No       | An optional object for any credentials needed to access the URL (e.g., username, password, API key).  If the URL is publicly accessible, this can be an empty object `{}`. |   `{}`   |
| `metadata`        | object    | No       | An optional object for any additional metadata you want to associate with the ingested data.  This can be any arbitrary key-value pairs.                                     |   `{}`   |

**Example Request Body:**

```json
{
  "url": "https://vi.wikipedia.org/wiki/Nguy%E1%BB%85n_Tr%C3%A3i",
  "collection_name": "callcenter",
  "chunk_size": 500,
  "chunk_overlap": 100,
  "credentials": {},
  "metadata": {}
}