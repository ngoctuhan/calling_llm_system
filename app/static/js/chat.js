// Chat application for the Call Center Information System

document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chat-form');
    const questionInput = document.getElementById('question-input');
    const modeSelect = document.getElementById('mode-select');
    const chatMessages = document.getElementById('chat-messages');
    const thinkingSteps = document.getElementById('thinking-steps');
    const submitButton = document.getElementById('submit-button');
    
    let socket = null;
    
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
                // Display thinking steps
                const stepElement = document.createElement('div');
                stepElement.classList.add('thinking-step');
                stepElement.innerHTML = `<strong>${data.step}:</strong> ${data.content}`;
                thinkingSteps.appendChild(stepElement);
                
                // Auto-scroll to the bottom
                thinkingSteps.scrollTop = thinkingSteps.scrollHeight;
            } else if (data.type === 'result') {
                // Display the final answer
                const messageElement = document.createElement('div');
                messageElement.classList.add('message', 'bot-message');
                
                let messageHTML = `<p>${data.answer}</p>`;
                
                // Display sources if available
                if (data.sources && data.sources.length > 0) {
                    messageHTML += '<div class="sources"><h4>Sources:</h4><ul>';
                    data.sources.forEach(source => {
                        messageHTML += `<li>${source.title || 'Source'} - ${source.content.substring(0, 100)}...</li>`;
                    });
                    messageHTML += '</ul></div>';
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
                                    messageHTML += `<td>${row[key]}</td>`;
                                });
                                messageHTML += '</tr>';
                            });
                            
                            messageHTML += '</tbody></table>';
                        }
                        
                        messageHTML += '</div>';
                    }
                    
                    messageHTML += '</div>';
                }
                
                messageElement.innerHTML = messageHTML;
                chatMessages.appendChild(messageElement);
                
                // Enable the form
                enableForm();
                
                // Auto-scroll to the bottom
                chatMessages.scrollTop = chatMessages.scrollHeight;
            } else if (data.type === 'error') {
                // Display error message
                const errorElement = document.createElement('div');
                errorElement.classList.add('message', 'error-message');
                errorElement.textContent = data.message;
                chatMessages.appendChild(errorElement);
                
                // Enable the form
                enableForm();
                
                // Auto-scroll to the bottom
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
        };
        
        socket.onclose = () => {
            console.log('WebSocket connection closed');
            // Try to reconnect after a delay
            setTimeout(connectWebSocket, 3000);
        };
        
        socket.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
    }
    
    // Connect to WebSocket
    connectWebSocket();
    
    // Function to handle form submission for WebSocket
    function handleSubmitWs(event) {
        event.preventDefault();
        
        const question = questionInput.value.trim();
        const mode = modeSelect.value;
        
        if (!question) return;
        
        // Display user message
        const userMessageElement = document.createElement('div');
        userMessageElement.classList.add('message', 'user-message');
        userMessageElement.textContent = question;
        chatMessages.appendChild(userMessageElement);
        
        // Clear thinking steps
        thinkingSteps.innerHTML = '';
        
        // Send message via WebSocket
        if (socket && socket.readyState === WebSocket.OPEN) {
            socket.send(JSON.stringify({
                question: question,
                mode: mode
            }));
            
            // Disable form while processing
            disableForm();
            
            // Clear input
            questionInput.value = '';
            
            // Auto-scroll to the bottom
            chatMessages.scrollTop = chatMessages.scrollHeight;
        } else {
            // Fallback to REST API if WebSocket is not available
            handleSubmitRest();
        }
    }
    
    // Function to handle form submission for REST API (fallback)
    async function handleSubmitRest() {
        const question = questionInput.value.trim();
        const mode = modeSelect.value;
        
        if (!question) return;
        
        // Display user message
        const userMessageElement = document.createElement('div');
        userMessageElement.classList.add('message', 'user-message');
        userMessageElement.textContent = question;
        chatMessages.appendChild(userMessageElement);
        
        // Clear thinking steps
        thinkingSteps.innerHTML = '';
        
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
            
            const data = await response.json();
            
            // Display thinking steps
            if (data.thinking && data.thinking.length > 0) {
                data.thinking.forEach(step => {
                    const stepElement = document.createElement('div');
                    stepElement.classList.add('thinking-step');
                    stepElement.innerHTML = `<strong>${step.step}:</strong> ${step.content}`;
                    thinkingSteps.appendChild(stepElement);
                });
            }
            
            // Display the final answer
            const messageElement = document.createElement('div');
            messageElement.classList.add('message', 'bot-message');
            
            let messageHTML = `<p>${data.answer}</p>`;
            
            // Display sources if available
            if (data.sources && data.sources.length > 0) {
                messageHTML += '<div class="sources"><h4>Sources:</h4><ul>';
                data.sources.forEach(source => {
                    messageHTML += `<li>${source.title || 'Source'} - ${source.content.substring(0, 100)}...</li>`;
                });
                messageHTML += '</ul></div>';
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
                                messageHTML += `<td>${row[key]}</td>`;
                            });
                            messageHTML += '</tr>';
                        });
                        
                        messageHTML += '</tbody></table>';
                    }
                    
                    messageHTML += '</div>';
                }
                
                messageHTML += '</div>';
            }
            
            messageElement.innerHTML = messageHTML;
            chatMessages.appendChild(messageElement);
        } catch (error) {
            console.error('Error submitting question:', error);
            
            // Display error message
            const errorElement = document.createElement('div');
            errorElement.classList.add('message', 'error-message');
            errorElement.textContent = 'Error processing your question. Please try again.';
            chatMessages.appendChild(errorElement);
        } finally {
            // Enable the form
            enableForm();
            
            // Clear input
            questionInput.value = '';
            
            // Auto-scroll to the bottom
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    }
    
    // Helper functions to disable/enable form
    function disableForm() {
        questionInput.disabled = true;
        modeSelect.disabled = true;
        submitButton.disabled = true;
        submitButton.textContent = 'Processing...';
    }
    
    function enableForm() {
        questionInput.disabled = false;
        modeSelect.disabled = false;
        submitButton.disabled = false;
        submitButton.textContent = 'Send';
    }
    
    // Attach event listener for form submission
    chatForm.addEventListener('submit', handleSubmitWs);
}); 