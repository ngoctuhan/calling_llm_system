// Sample data for demonstration
const sampleDataSources = {
    json: { type: 'json', chunks: 24, indexed: true },
    csv: { type: 'csv', chunks: 18, indexed: true },
    txt: { type: 'txt', chunks: 32, indexed: true },
    pdf: { type: 'pdf', chunks: 53, indexed: true },
    html: { type: 'html', chunks: 27, indexed: true },
    markdown: { type: 'markdown', chunks: 15, indexed: true }
};

const ragMethods = {
    vector: {
        description: "Dense vector similarity search",
        metrics: {
            recall: 0.87,
            precision: 0.82,
            latency: "120ms"
        }
    },
    graph: {
        description: "Knowledge graph traversal and entity linking",
        metrics: {
            recall: 0.81,
            precision: 0.89,
            latency: "190ms"
        }
    },
    hybrid: {
        description: "Combined vector and semantic search with reranking",
        metrics: {
            recall: 0.92,
            precision: 0.86,
            latency: "210ms"
        }
    },
    text2sql: {
        description: "Natural language to SQL query conversion",
        metrics: {
            recall: 0.79,
            precision: 0.93,
            latency: "170ms"
        }
    }
};

// DOM Elements
let configureBtn;
let testBtn;
let copyBtn;
let downloadBtn;
let resultContainer;
let loadingContainer;
let emptyContainer;
let resultJson;
let metadataToggle;
let metadataFields;

// Initialize the page
document.addEventListener('DOMContentLoaded', function() {
    try {
        // Configure highlight.js
        hljs.configure({
            languages: ['json']
        });
        
        // Get DOM elements
        configureBtn = document.getElementById('configureBtn');
        testBtn = document.getElementById('testBtn');
        copyBtn = document.getElementById('copyBtn');
        downloadBtn = document.getElementById('downloadBtn');
        resultContainer = document.getElementById('resultContainer');
        loadingContainer = document.getElementById('loadingContainer');
        emptyContainer = document.getElementById('emptyContainer');
        resultJson = document.getElementById('resultJson');
        metadataToggle = document.getElementById('metadataToggle');
        metadataFields = document.getElementById('metadataFields');
        
        // Add event listeners
        if (metadataToggle) {
            metadataToggle.addEventListener('change', toggleMetadataFields);
        }
        
        if (configureBtn) {
            configureBtn.addEventListener('click', handleConfigureData);
        }
        
        if (testBtn) {
            testBtn.addEventListener('click', handleTestRag);
        }
        
        if (copyBtn) {
            copyBtn.addEventListener('click', handleCopyJson);
        }
        
        if (downloadBtn) {
            downloadBtn.addEventListener('click', handleDownloadJson);
        }
    } catch (error) {
        console.error('Error initializing page:', error);
    }
});

// Toggle metadata fields visibility
function toggleMetadataFields() {
    if (!metadataFields) {
        console.error('metadataFields element not found');
        return;
    }
    
    if (this.checked) {
        metadataFields.classList.remove('hidden');
    } else {
        metadataFields.classList.add('hidden');
    }
}

// Handle data configuration
function handleConfigureData() {
    const dataUrlElement = document.getElementById('dataUrl');
    const dataTypeElement = document.getElementById('dataType');
    const metadataToggleElement = document.getElementById('metadataToggle');
    
    if (!dataUrlElement) {
        console.error('Element with ID "dataUrl" not found');
        return;
    }
    
    const url = dataUrlElement.value;
    const type = dataTypeElement ? dataTypeElement.value : 'txt';
    const includeMetadata = metadataToggleElement ? metadataToggleElement.checked : false;
    
    if (!url) {
        alert('Please enter a data source URL');
        return;
    }
    
    showLoading();
    
    // Prepare request data
    let metadata = {};
    if (includeMetadata) {
        try {
            const metadataJsonElement = document.getElementById('metadataJson');
            metadata = metadataJsonElement && metadataJsonElement.value ? 
                JSON.parse(metadataJsonElement.value) : {};
        } catch (e) {
            alert('Invalid JSON format for metadata');
            hideLoading();
            return;
        }
    }
    
    const requestData = {
        url: url,
        credentials: {},
        metadata: metadata
    };
    
    // Make API call
    fetch('/api/v1/ingestion/ingest/url', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        },
        body: JSON.stringify(requestData)
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => { 
                throw new Error(err.detail || 'Failed to ingest URL');
            });
        }
        return response.json();
    })
    .then(data => {
        // Display configuration result
        displayResult({
            status: "success",
            operation: "configure",
            timestamp: new Date().toISOString(),
            dataSource: {
                url: url,
                type: type,
                indexed: true
            },
            api_response: data
        });
    })
    .catch(error => {
        console.error('Error:', error);
        // Display error result
        displayResult({
            status: "error",
            operation: "configure",
            timestamp: new Date().toISOString(),
            error: error.message || "Failed to ingest data",
            request: requestData
        });
    })
    .finally(() => {
        hideLoading();
    });
}

// Handle RAG testing
function handleTestRag() {
    const queryInputElement = document.getElementById('queryInput');
    const ragMethodElement = document.getElementById('ragMethod');
    const modelTypeElement = document.getElementById('modelType');
    const chunkingToggleElement = document.getElementById('chunkingToggle');
    
    if (!queryInputElement) {
        console.error('Element with ID "queryInput" not found');
        return;
    }
    
    const query = queryInputElement.value;
    const method = ragMethodElement ? ragMethodElement.value : 'vector';
    const model = modelTypeElement ? modelTypeElement.value : 'gemini-2.0-flash';
    const smartChunking = chunkingToggleElement ? chunkingToggleElement.checked : false;
    
    if (!query) {
        alert('Please enter a query');
        return;
    }
    
    showLoading();
    
    // Get the collection name from the previous configuration or use default
    const collectionName = window.lastConfiguredDataSource ? 
        window.lastConfiguredDataSource.collection_name : 'callcenter';
    
    // Prepare request data
    const requestData = {
        query: query,
        collection_name: collectionName,
        top_k: 10,
        model_name: model
    };
    
    // Determine API endpoint based on method
    let endpoint = '';
    switch(method) {
        case 'vector':
            endpoint = '/api/v1/retrieval/test/vector';
            break;
        case 'graph':
            endpoint = '/api/v1/retrieval/test/graph';
            break;
        case 'hybrid':
            endpoint = '/api/v1/retrieval/test/hybrid';
            break;
        case 'text2sql':
            endpoint = '/api/v1/retrieval/test/text2sql';
            break;
        default:
            endpoint = '/api/v1/retrieval/test/vector';
    }
    
    // Make API call
    fetch(endpoint, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        },
        body: JSON.stringify(requestData)
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => { 
                throw new Error(err.detail || `Failed to process ${method} query`);
            });
        }
        return response.json();
    })
    .then(data => {
        // Format result for display
        const formattedResult = {
            status: data.success ? "success" : "error",
            operation: "test",
            timestamp: new Date().toISOString(),
            query: {
                text: data.query,
                method: data.method
            },
            results: data.results,
            answer: data.answer,
            execution_time: data.execution_time,
            dataSource: window.lastConfiguredDataSource || {
                collection_name: collectionName,
                type: "txt"
            }
        };
        
        if (data.error) {
            formattedResult.error = data.error;
        }
        
        displayResult(formattedResult);
    })
    .catch(error => {
        console.error('Error:', error);
        // Display error result
        displayResult({
            status: "error",
            operation: "test",
            timestamp: new Date().toISOString(),
            error: error.message || `Failed to process ${method} query`,
            request: requestData
        });
    })
    .finally(() => {
        hideLoading();
    });
}

// Copy JSON to clipboard
function handleCopyJson() {
    const jsonStr = resultJson.textContent;
    navigator.clipboard.writeText(jsonStr)
        .then(() => {
            alert('JSON copied to clipboard');
        })
        .catch(err => {
            console.error('Error copying text: ', err);
        });
}

// Download JSON
function handleDownloadJson() {
    const jsonStr = resultJson.textContent;
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(jsonStr);
    const downloadAnchorNode = document.createElement('a');
    downloadAnchorNode.setAttribute("href", dataStr);
    downloadAnchorNode.setAttribute("download", "rag-results.json");
    document.body.appendChild(downloadAnchorNode);
    downloadAnchorNode.click();
    downloadAnchorNode.remove();
}

// Display results
function displayResult(result) {
    const jsonStr = JSON.stringify(result, null, 2);
    resultJson.textContent = jsonStr;
    hljs.highlightElement(resultJson);
    
    // Store the data source if this was a successful configuration
    if (result.status === "success" && result.operation === "configure" && result.dataSource) {
        // Always use default collection name
        window.lastConfiguredDataSource = {
            url: result.dataSource.url,
            type: result.dataSource.type,
            collection_name: 'callcenter',
            indexed: true
        };
    }
    
    hideLoading();
    emptyContainer.classList.add('hidden');
    resultContainer.classList.remove('hidden');
}

// Show loading indicator
function showLoading() {
    loadingContainer.classList.remove('hidden');
    emptyContainer.classList.add('hidden');
    resultContainer.classList.add('hidden');
}

// Hide loading indicator
function hideLoading() {
    loadingContainer.classList.add('hidden');
} 