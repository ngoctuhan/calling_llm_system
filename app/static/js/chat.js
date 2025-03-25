// Chat application for the Call Center Information System

document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chat-form');
    const questionInput = document.getElementById('question-input');
    const modeSelect = document.getElementById('mode-select');
    const chatMessages = document.getElementById('chat-messages');
    const thinkingSteps = document.getElementById('thinking-steps');
    const submitButton = document.getElementById('submit-button');
    const thinkingWindow = document.getElementById('thinking-window');
    const thinkingToggle = document.getElementById('thinking-toggle');
    const thinkingIndicator = document.getElementById('thinking-indicator');
    const suggestionChips = document.querySelectorAll('.suggestion-chip');
    
    let socket = null;
    let isProcessing = false;
    let hasThinkingSteps = false;
    
    // Thêm biến toàn cục để lưu trữ sources và SQL data
    let streamSourcesData = null;
    let streamSqlData = null;
    let streamSqlQuery = null;
    
    // Auto-resize textarea as user types
    questionInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
    });
    
    // Handle suggestion chips
    suggestionChips.forEach(chip => {
        chip.addEventListener('click', () => {
            questionInput.value = chip.textContent;
            questionInput.style.height = 'auto';
            questionInput.style.height = (questionInput.scrollHeight) + 'px';
            questionInput.focus();
        });
    });
    
    // Initialize toggle functionality
    thinkingToggle.addEventListener('click', toggleThinkingPanel);
    
    function toggleThinkingPanel() {
        thinkingWindow.classList.toggle('collapsed');
        thinkingToggle.classList.toggle('collapsed');
        
        // If there are thinking steps and the panel is collapsed, show the indicator
        if (hasThinkingSteps && thinkingWindow.classList.contains('collapsed')) {
            thinkingIndicator.classList.add('active');
        } else {
            thinkingIndicator.classList.remove('active');
        }
    }
    
    // Function to initialize WebSocket connection
    function connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
        const wsUrl = `${protocol}://${window.location.host}/api/v1/chat/ws/chat`;
        
        socket = new WebSocket(wsUrl);
        
        socket.onopen = () => {
            console.log('WebSocket connection established');
        };
        
        socket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            console.log('Received message:', data);
            
            if (data.type === 'thinking') {
                // Only show the thinking panel when we start receiving thinking steps
                if (data.step === 'start' && thinkingWindow.classList.contains('collapsed')) {
                    thinkingWindow.classList.remove('collapsed');
                    thinkingToggle.classList.remove('collapsed');
                }
                
                // Display thinking steps
                displayThinkingStep(data.step, data.content);
                hasThinkingSteps = true;
            } else if (data.type === 'stream_start') {
                // Reset sources và SQL data khi bắt đầu stream mới
                streamSourcesData = null;
                streamSqlData = null;
                streamSqlQuery = null;
                
                // Create a new message container for the streamed answer
                createStreamingMessageContainer();
            } else if (data.type === 'stream_chunk') {
                // Append to the streaming container
                appendToStreamingMessage(data.chunk);
            } else if (data.type === 'stream_end') {
                // Finalize the streaming container
                finalizeStreamingMessage();
                
                // Set a flag to indicate we've already displayed a streaming response
                window.hasDisplayedStreamingResponse = true;
                
                // Auto-collapse the thinking panel after receiving the result
                setTimeout(() => {
                    thinkingWindow.classList.add('collapsed');
                    thinkingToggle.classList.add('collapsed');
                    
                    // Show the thinking indicator if there are steps
                    if (hasThinkingSteps) {
                        thinkingIndicator.classList.add('active');
                    }
                }, 1000);
                
                // Enable the form
                enableForm();
                
                // Auto-scroll to the bottom
                chatMessages.scrollTop = chatMessages.scrollHeight;
            } else if (data.type === 'result') {
                // Lưu trữ sources và SQL data để sử dụng trong streaming response
                if (data.sources) streamSourcesData = data.sources;
                if (data.sql) streamSqlQuery = data.sql;
                if (data.data) streamSqlData = data.data;
                
                // Auto-collapse the thinking panel after receiving the result
                setTimeout(() => {
                    thinkingWindow.classList.add('collapsed');
                    thinkingToggle.classList.add('collapsed');
                    
                    // Show the thinking indicator if there are steps
                    if (hasThinkingSteps) {
                        thinkingIndicator.classList.add('active');
                    }
                }, 1000);
                
                // If we didn't use streaming, display the final answer normally
                // Only display if we haven't already displayed a streaming response
                if (!document.querySelector('.bot-message.streaming') && !window.hasDisplayedStreamingResponse) {
                    // Display the final answer
                    displayBotMessage(data);
                }
                
                // Reset the streaming response flag for next message
                window.hasDisplayedStreamingResponse = false;
                
                // Enable the form
                enableForm();
                
                // Auto-scroll to the bottom
                chatMessages.scrollTop = chatMessages.scrollHeight;
            } else if (data.type === 'error') {
                // Display error message
                displayErrorMessage(data.message);
                
                // Enable the form
                enableForm();
            }
        };
        
        socket.onclose = () => {
            console.log('WebSocket connection closed');
            // Try to reconnect after a delay
            setTimeout(connectWebSocket, 3000);
        };
        
        socket.onerror = (error) => {
            console.error('WebSocket error:', error);
            displayErrorMessage('Connection error. Please try again later.');
        };
    }
    
    // Helper functions for streaming responses
    let currentStreamingGroup = null;
    let currentStreamingMessage = null;
    
    function createStreamingMessageContainer() {
        // Create a message group for the bot if not already streaming
        if (!currentStreamingGroup) {
            currentStreamingGroup = document.createElement('div');
            currentStreamingGroup.classList.add('message-group', 'bot');
            
            const messageHeader = document.createElement('div');
            messageHeader.classList.add('message-header');
            
            const avatar = document.createElement('div');
            avatar.classList.add('message-avatar', 'bot');
            
            const avatarSvg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
            avatarSvg.setAttribute('width', '16');
            avatarSvg.setAttribute('height', '16');
            avatarSvg.setAttribute('viewBox', '0 0 24 24');
            avatarSvg.setAttribute('fill', 'none');
            avatarSvg.setAttribute('stroke', 'currentColor');
            avatarSvg.setAttribute('stroke-width', '2');
            avatarSvg.setAttribute('stroke-linecap', 'round');
            avatarSvg.setAttribute('stroke-linejoin', 'round');
            
            const circlePath = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            circlePath.setAttribute('cx', '12');
            circlePath.setAttribute('cy', '12');
            circlePath.setAttribute('r', '10');
            
            const facePath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            facePath.setAttribute('d', 'M16 16s-1.5-2-4-2-4 2-4 2');
            
            const eyePath1 = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            eyePath1.setAttribute('x1', '9');
            eyePath1.setAttribute('y1', '9');
            eyePath1.setAttribute('x2', '9.01');
            eyePath1.setAttribute('y2', '9');
            
            const eyePath2 = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            eyePath2.setAttribute('x1', '15');
            eyePath2.setAttribute('y1', '9');
            eyePath2.setAttribute('x2', '15.01');
            eyePath2.setAttribute('y2', '9');
            
            avatarSvg.appendChild(circlePath);
            avatarSvg.appendChild(facePath);
            avatarSvg.appendChild(eyePath1);
            avatarSvg.appendChild(eyePath2);
            avatar.appendChild(avatarSvg);
            
            const name = document.createElement('div');
            name.classList.add('message-name');
            name.textContent = 'Assistant';
            
            messageHeader.appendChild(avatar);
            messageHeader.appendChild(name);
            
            currentStreamingMessage = document.createElement('div');
            currentStreamingMessage.classList.add('message', 'bot-message', 'streaming');
            
            // Add a cursor element
            const cursor = document.createElement('span');
            cursor.classList.add('streaming-cursor');
            cursor.innerHTML = '&#9612;';
            
            currentStreamingGroup.appendChild(messageHeader);
            currentStreamingGroup.appendChild(currentStreamingMessage);
            currentStreamingMessage.appendChild(cursor);
            
            chatMessages.appendChild(currentStreamingGroup);
            
            // Auto-scroll to the bottom
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    }
    
    function appendToStreamingMessage(chunk) {
        if (currentStreamingMessage) {
            // Remove the cursor if it exists
            const cursor = currentStreamingMessage.querySelector('.streaming-cursor');
            if (cursor) {
                currentStreamingMessage.removeChild(cursor);
            }
            
            // Handle the case where a full message is sent at once (fallback case)
            const isLongChunk = chunk.length > 200;
            
            if (isLongChunk && (chunk.includes('\n\n') || chunk.includes('\n'))) {
                // This is likely a full message - format it nicely
                if (chunk.includes('\n\n')) {
                    // Multiple paragraphs
                    const paragraphs = chunk.split('\n\n').filter(p => p.trim() !== '');
                    currentStreamingMessage.innerHTML = paragraphs.map(p => `<p>${p}</p>`).join('');
                } else if (chunk.includes('\n')) {
                    // Line breaks but not paragraphs
                    const lines = chunk.split('\n').filter(line => line.trim() !== '');
                    currentStreamingMessage.innerHTML = lines.map(line => `<p>${line}</p>`).join('');
                } else {
                    // Just set as is
                    currentStreamingMessage.textContent = chunk;
                }
            } else {
                // Format the chunk with paragraphs if it contains newlines
                if (chunk.includes('\n\n')) {
                    // We need to handle partial chunks, so we'll append text nodes and p elements as needed
                    const lines = chunk.split('\n\n');
                    
                    for (let i = 0; i < lines.length; i++) {
                        const line = lines[i];
                        if (line.trim() === '') continue;
                        
                        // Create a paragraph element
                        const p = document.createElement('p');
                        p.textContent = line;
                        currentStreamingMessage.appendChild(p);
                    }
                } else {
                    // Just append the text
                    const textNode = document.createTextNode(chunk);
                    currentStreamingMessage.appendChild(textNode);
                }
            }
            
            // Add the cursor back
            const newCursor = document.createElement('span');
            newCursor.classList.add('streaming-cursor');
            newCursor.innerHTML = '&#9612;';
            currentStreamingMessage.appendChild(newCursor);
            
            // Auto-scroll to the bottom
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    }
    
    function finalizeStreamingMessage() {
        if (currentStreamingMessage) {
            // Remove the streaming cursor
            const cursor = currentStreamingMessage.querySelector('.streaming-cursor');
            if (cursor) {
                currentStreamingMessage.removeChild(cursor);
            }
            
            // Format the content properly
            const content = currentStreamingMessage.innerHTML;
            
            // If the content doesn't already contain paragraph tags, wrap it in paragraphs
            if (!content.includes('<p>')) {
                // Check if we have multiple paragraphs (separated by double newlines)
                if (content.includes('\n\n')) {
                    const paragraphs = content.split('\n\n').filter(p => p.trim() !== '');
                    currentStreamingMessage.innerHTML = paragraphs.map(p => `<p>${p}</p>`).join('');
                } else if (content.includes('\n')) {
                    // Check for single newlines and treat as separate paragraphs
                    const lines = content.split('\n').filter(line => line.trim() !== '');
                    currentStreamingMessage.innerHTML = lines.map(line => `<p>${line}</p>`).join('');
                } else if (content.length > 100) {
                    // If it's a long single paragraph, wrap it
                    currentStreamingMessage.innerHTML = `<p>${content}</p>`;
                }
            }
            
            // Thêm sources nếu có
            if (streamSourcesData && streamSourcesData.length > 0) {
                let sourcesHTML = '<div class="sources"><h4>Sources:</h4>';
                
                // Tạo accordion cho mỗi source
                sourcesHTML += '<div class="sources-accordion">';
                streamSourcesData.forEach((source, index) => {
                    const sourceId = `source-${Date.now()}-${index}`;
                    const title = source.title || `Source ${index + 1}`;
                    const content = source.content || '';
                    
                    sourcesHTML += `
                        <div class="source-item">
                            <div class="source-header" onclick="toggleSource('${sourceId}')">
                                ${title} <span class="toggle-icon">+</span>
                            </div>
                            <div id="${sourceId}" class="source-content">
                                <p>${content}</p>
                            </div>
                        </div>
                    `;
                });
                sourcesHTML += '</div></div>';
                
                currentStreamingMessage.innerHTML += sourcesHTML;
                
                // Add toggle functionality
                if (!window.toggleSource) {
                    window.toggleSource = function(id) {
                        const content = document.getElementById(id);
                        const header = content.previousElementSibling;
                        const icon = header.querySelector('.toggle-icon');
                        
                        if (content.style.display === 'block') {
                            content.style.display = 'none';
                            icon.textContent = '+';
                        } else {
                            content.style.display = 'block';
                            icon.textContent = '-';
                        }
                    };
                }
            }
            
            // Thêm SQL và data nếu có
            if (streamSqlQuery) {
                let sqlHTML = `<div class="sql-info"><h4>SQL Query:</h4><pre>${streamSqlQuery}</pre>`;
                
                if (streamSqlData && streamSqlData.length > 0) {
                    sqlHTML += '<h4>Query Results:</h4><div class="sql-results">';
                    
                    // Display as a table if the data has consistent keys
                    if (streamSqlData.length > 0) {
                        const keys = Object.keys(streamSqlData[0]);
                        
                        sqlHTML += '<table><thead><tr>';
                        keys.forEach(key => {
                            sqlHTML += `<th>${key}</th>`;
                        });
                        sqlHTML += '</tr></thead><tbody>';
                        
                        streamSqlData.forEach(row => {
                            sqlHTML += '<tr>';
                            keys.forEach(key => {
                                sqlHTML += `<td>${row[key] !== null ? row[key] : ''}</td>`;
                            });
                            sqlHTML += '</tr>';
                        });
                        
                        sqlHTML += '</tbody></table>';
                    }
                    
                    sqlHTML += '</div>';
                } else {
                    sqlHTML += '<p>No data returned from query.</p>';
                }
                
                sqlHTML += '</div>';
                currentStreamingMessage.innerHTML += sqlHTML;
            }
            
            // Remove the streaming class
            currentStreamingMessage.classList.remove('streaming');
            
            // Reset the current streaming elements
            currentStreamingGroup = null;
            currentStreamingMessage = null;
        }
    }
    
    // Connect to WebSocket
    connectWebSocket();
    
    // Function to handle form submission for WebSocket
    function handleSubmitWs(event) {
        event.preventDefault();
        
        // Prevent multiple submissions
        if (isProcessing) return;
        
        const question = questionInput.value.trim();
        const mode = modeSelect.value;
        
        if (!question) return;
        
        // Reset thinking steps state
        hasThinkingSteps = false;
        thinkingIndicator.classList.remove('active');
        
        // Display user message
        displayUserMessage(question);
        
        // Clear thinking steps and add initial message
        clearThinkingSteps();
        
        // Add initial "thinking" badge to the thinking section
        addThinkingBadge();
        
        // Send message via WebSocket
        if (socket && socket.readyState === WebSocket.OPEN) {
            socket.send(JSON.stringify({
                question: question,
                mode: mode
            }));
            
            // Disable form while processing
            disableForm();
            
            // Clear input and reset height
            questionInput.value = '';
            questionInput.style.height = '56px';
            
            // Auto-scroll to the bottom
            chatMessages.scrollTop = chatMessages.scrollHeight;
        } else {
            // Fallback to REST API if WebSocket is not available
            handleSubmitRest();
        }
    }
    
    // Function to handle form submission for REST API (fallback)
    async function handleSubmitRest() {
        // Prevent multiple submissions
        if (isProcessing) return;
        
        const question = questionInput.value.trim();
        const mode = modeSelect.value;
        
        if (!question) return;
        
        // Reset thinking steps state
        hasThinkingSteps = false;
        thinkingIndicator.classList.remove('active');
        
        // Display user message
        displayUserMessage(question);
        
        // Clear thinking steps
        clearThinkingSteps();
        
        // Add initial "thinking" badge to the thinking section
        addThinkingBadge();
        
        // Show the thinking panel when we start processing
        thinkingWindow.classList.remove('collapsed');
        thinkingToggle.classList.remove('collapsed');
        
        try {
            // Disable form while processing
            disableForm();
            
            const response = await fetch('/api/v1/chat/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    question: question,
                    mode: mode
                })
            });
            
            if (!response.ok) {
                throw new Error(`Server responded with status: ${response.status}`);
            }
            
            const data = await response.json();
            
            // Remove thinking badge
            removeThinkingBadge();
            
            // Display thinking steps
            if (data.thinking && data.thinking.length > 0) {
                hasThinkingSteps = true;
                data.thinking.forEach((step, index) => {
                    // Add slight delay for each step to create a sequential appearance
                    setTimeout(() => {
                        displayThinkingStep(step.step, step.content);
                    }, index * 100);
                });
            }
            
            // Display the final answer with a slight delay after thinking steps
            setTimeout(() => {
                displayBotMessage(data);
                
                // Auto-collapse the thinking panel after a delay
                setTimeout(() => {
                    thinkingWindow.classList.add('collapsed');
                    thinkingToggle.classList.add('collapsed');
                    
                    // Show the thinking indicator if there are steps
                    if (hasThinkingSteps) {
                        thinkingIndicator.classList.add('active');
                    }
                }, 1000);
            }, (data.thinking?.length || 0) * 100 + 200);
            
        } catch (error) {
            console.error('Error submitting question:', error);
            displayErrorMessage('Error processing your question. Please try again.');
            
            // Auto-collapse the thinking panel after error
            setTimeout(() => {
                thinkingWindow.classList.add('collapsed');
                thinkingToggle.classList.add('collapsed');
            }, 1000);
        } finally {
            // Enable the form
            enableForm();
            
            // Clear input and reset height
            questionInput.value = '';
            questionInput.style.height = '56px';
            
            // Auto-scroll to the bottom
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    }
    
    // Helper function to display user message
    function displayUserMessage(text) {
        const messageGroup = document.createElement('div');
        messageGroup.classList.add('message-group', 'user');
        
        const messageHeader = document.createElement('div');
        messageHeader.classList.add('message-header');
        
        const avatar = document.createElement('div');
        avatar.classList.add('message-avatar');
        avatar.textContent = 'U';
        
        const name = document.createElement('div');
        name.classList.add('message-name');
        name.textContent = 'You';
        
        messageHeader.appendChild(avatar);
        messageHeader.appendChild(name);
        
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', 'user-message');
        messageElement.textContent = text;
        
        messageGroup.appendChild(messageHeader);
        messageGroup.appendChild(messageElement);
        
        chatMessages.appendChild(messageGroup);
        
        // Auto-scroll to the bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Helper function to display bot message
    function displayBotMessage(data) {
        const messageGroup = document.createElement('div');
        messageGroup.classList.add('message-group', 'bot');
        
        const messageHeader = document.createElement('div');
        messageHeader.classList.add('message-header');
        
        const avatar = document.createElement('div');
        avatar.classList.add('message-avatar', 'bot');
        
        const avatarSvg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        avatarSvg.setAttribute('width', '16');
        avatarSvg.setAttribute('height', '16');
        avatarSvg.setAttribute('viewBox', '0 0 24 24');
        avatarSvg.setAttribute('fill', 'none');
        avatarSvg.setAttribute('stroke', 'currentColor');
        avatarSvg.setAttribute('stroke-width', '2');
        avatarSvg.setAttribute('stroke-linecap', 'round');
        avatarSvg.setAttribute('stroke-linejoin', 'round');
        
        const circlePath = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circlePath.setAttribute('cx', '12');
        circlePath.setAttribute('cy', '12');
        circlePath.setAttribute('r', '10');
        
        const facePath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        facePath.setAttribute('d', 'M16 16s-1.5-2-4-2-4 2-4 2');
        
        const eyePath1 = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        eyePath1.setAttribute('x1', '9');
        eyePath1.setAttribute('y1', '9');
        eyePath1.setAttribute('x2', '9.01');
        eyePath1.setAttribute('y2', '9');
        
        const eyePath2 = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        eyePath2.setAttribute('x1', '15');
        eyePath2.setAttribute('y1', '9');
        eyePath2.setAttribute('x2', '15.01');
        eyePath2.setAttribute('y2', '9');
        
        avatarSvg.appendChild(circlePath);
        avatarSvg.appendChild(facePath);
        avatarSvg.appendChild(eyePath1);
        avatarSvg.appendChild(eyePath2);
        avatar.appendChild(avatarSvg);
        
        const name = document.createElement('div');
        name.classList.add('message-name');
        name.textContent = 'Assistant';
        
        messageHeader.appendChild(avatar);
        messageHeader.appendChild(name);
        
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', 'bot-message');
        
        // Format the answer with paragraphs
        const formattedAnswer = data.answer.split('\n\n').map(para => `<p>${para}</p>`).join('');
        let messageHTML = formattedAnswer;
        
        // Display sources if available
        if (data.sources && data.sources.length > 0) {
            messageHTML += '<div class="sources"><h4>Sources:</h4>';
            
            // Tạo accordion cho mỗi source
            messageHTML += '<div class="sources-accordion">';
            data.sources.forEach((source, index) => {
                const sourceId = `source-${Date.now()}-${index}`;
                const title = source.title || `Source ${index + 1}`;
                const content = source.content || '';
                
                messageHTML += `
                    <div class="source-item">
                        <div class="source-header" onclick="toggleSource('${sourceId}')">
                            ${title} <span class="toggle-icon">+</span>
                        </div>
                        <div id="${sourceId}" class="source-content">
                            <p>${content}</p>
                        </div>
                    </div>
                `;
            });
            messageHTML += '</div></div>';
        }
        
        // Display SQL and data if available
        if (data.sql) {
            messageHTML += `<div class="sql-info"><h4>SQL Query:</h4><pre>${data.sql}</pre>`;
            
            if (data.data && data.data.length > 0) {
                messageHTML += '<h4>Query Results:</h4><div class="sql-results">';
                
                // Display as a table if the data has consistent keys
                if (data.data.length > 0) {
                    const keys = Object.keys(data.data[0]);
                    
                    messageHTML += '<table><thead><tr>';
                    keys.forEach(key => {
                        messageHTML += `<th>${key}</th>`;
                    });
                    messageHTML += '</tr></thead><tbody>';
                    
                    data.data.forEach(row => {
                        messageHTML += '<tr>';
                        keys.forEach(key => {
                            messageHTML += `<td>${row[key] !== null ? row[key] : ''}</td>`;
                        });
                        messageHTML += '</tr>';
                    });
                    
                    messageHTML += '</tbody></table>';
                }
                
                messageHTML += '</div>';
            } else {
                messageHTML += '<p>No data returned from query.</p>';
            }
            
            messageHTML += '</div>';
        }
        
        messageElement.innerHTML = messageHTML;
        
        messageGroup.appendChild(messageHeader);
        messageGroup.appendChild(messageElement);
        
        chatMessages.appendChild(messageGroup);
        
        // Remove thinking badge if it exists
        removeThinkingBadge();
        
        // Auto-scroll to the bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Helper function to display error message
    function displayErrorMessage(message) {
        const errorElement = document.createElement('div');
        errorElement.classList.add('error-message');
        errorElement.textContent = message;
        chatMessages.appendChild(errorElement);
        
        // Remove thinking badge if it exists
        removeThinkingBadge();
        
        // Auto-scroll to the bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Helper function to display thinking step
    function displayThinkingStep(step, content) {
        // Remove the empty state message if it exists
        const emptyStateMessage = thinkingSteps.querySelector('.text-center');
        if (emptyStateMessage) {
            thinkingSteps.removeChild(emptyStateMessage);
        }
        
        const stepElement = document.createElement('div');
        stepElement.classList.add('thinking-step');
        stepElement.innerHTML = `<strong>${step}</strong> ${content}`;
        thinkingSteps.appendChild(stepElement);
        
        // Auto-scroll to the bottom
        thinkingSteps.scrollTop = thinkingSteps.scrollHeight;
    }
    
    // Helper function to clear thinking steps
    function clearThinkingSteps() {
        thinkingSteps.innerHTML = '';
    }
    
    // Helper function to add thinking badge
    function addThinkingBadge() {
        // Make sure there isn't already a thinking badge
        removeThinkingBadge();
        
        // Add empty state message
        thinkingSteps.innerHTML = `
            <div class="thinking-badge">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="12" cy="12" r="10"></circle>
                    <polyline points="12 6 12 12 16 14"></polyline>
                </svg>
                Thinking...
            </div>
            <div class="p-6 text-center text-gray-500">
                <p>Processing your question. Steps will appear here...</p>
            </div>
        `;
    }
    
    // Helper function to remove thinking badge
    function removeThinkingBadge() {
        const thinkingBadge = thinkingSteps.querySelector('.thinking-badge');
        if (thinkingBadge) {
            thinkingBadge.remove();
        }
    }
    
    // Helper functions to disable/enable form
    function disableForm() {
        isProcessing = true;
        questionInput.disabled = true;
        modeSelect.disabled = true;
        submitButton.disabled = true;
        submitButton.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" class="animate-spin" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="10"></circle>
                <path d="M12 6v6l4 2"></path>
            </svg>
        `;
    }
    
    function enableForm() {
        isProcessing = false;
        questionInput.disabled = false;
        modeSelect.disabled = false;
        submitButton.disabled = false;
        submitButton.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z" />
            </svg>
        `;
        
        // Focus on the input field
        questionInput.focus();
    }
    
    // Add click handler to the thinking indicator to show thinking panel
    thinkingIndicator.addEventListener('click', function() {
        thinkingWindow.classList.remove('collapsed');
        thinkingToggle.classList.remove('collapsed');
        thinkingIndicator.classList.remove('active');
    });
    
    // Add keydown event listener for the input field
    questionInput.addEventListener('keydown', function(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            if (!isProcessing) {
                chatForm.dispatchEvent(new Event('submit'));
            }
        } else if (event.key === 'Enter' && event.shiftKey) {
            // Allow line breaks with Shift+Enter
            // No need to prevent default as we want the newline character
        }
    });
    
    // Attach event listener for form submission
    chatForm.addEventListener('submit', handleSubmitWs);
}); 