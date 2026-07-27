/**
 * SmartAssist – Interactive NLP Chatbot JavaScript Engine
 * --------------------------------------------------------
 * Manages chat interactions, fetch requests to Flask backend API,
 * dynamic message rendering, typewriter animation effects, auto-scrolling,
 * quick suggestion chips, and robust client-side error handling.
 * 
 * Author: Senior AI & NLP Engineer
 */

document.addEventListener('DOMContentLoaded', () => {
    // ------------------------------------------------------------------
    // 1. DOM Element Selection
    // ------------------------------------------------------------------
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatMessages = document.getElementById('chat-messages');
    const typingIndicator = document.getElementById('typing-indicator');
    const sendBtn = document.getElementById('send-btn');
    const clearChatBtn = document.getElementById('clear-chat-btn');
    const charCountDisplay = document.getElementById('char-count');
    const initialTimeDisplay = document.getElementById('initial-time');
    const suggestionChipsContainer = document.getElementById('suggestion-chips');

    const MAX_CHAR_LIMIT = 250;
    const BOT_AVATAR_URL = '/static/chatbot.png';

    // ------------------------------------------------------------------
    // 2. Initialize Timestamps & Controls
    // ------------------------------------------------------------------
    if (initialTimeDisplay) {
        initialTimeDisplay.textContent = getCurrentFormattedTime();
    }

    // Update character counter as user types
    userInput.addEventListener('input', () => {
        const len = userInput.value.length;
        charCountDisplay.textContent = `${len} / ${MAX_CHAR_LIMIT}`;
        
        if (len > MAX_CHAR_LIMIT) {
            charCountDisplay.style.color = '#ef4444';
            sendBtn.disabled = true;
        } else {
            charCountDisplay.style.color = '#64748b';
            sendBtn.disabled = len === 0;
        }
    });

    // Handle Quick Suggestion Chip clicks
    document.addEventListener('click', (e) => {
        const chip = e.target.closest('.chip-btn');
        if (chip) {
            const query = chip.getAttribute('data-query');
            if (query) {
                userInput.value = query;
                handleSendMessage(query);
            }
        }
    });

    // Handle Clear Chat History
    clearChatBtn.addEventListener('click', () => {
        if (confirm("Are you sure you want to clear the chat conversation?")) {
            // Keep only initial greeting message
            const botRows = chatMessages.querySelectorAll('.message-row');
            botRows.forEach((row, index) => {
                if (index > 0) row.remove();
            });
            // Re-append suggestion chips
            if (suggestionChipsContainer) {
                chatMessages.appendChild(suggestionChipsContainer);
            }
            scrollToBottom();
        }
    });

    // ------------------------------------------------------------------
    // 3. Form Submission Event Handler
    // ------------------------------------------------------------------
    chatForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const messageText = userInput.value.trim();
        if (messageText && messageText.length <= MAX_CHAR_LIMIT) {
            handleSendMessage(messageText);
        }
    });

    // Allow submission on Enter key (without Shift)
    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            chatForm.dispatchEvent(new Event('submit'));
        }
    });

    // ------------------------------------------------------------------
    // 4. Main Chat Orchestration Function
    // ------------------------------------------------------------------
    async function handleSendMessage(messageText) {
        // Phase 10: Validate non-empty message
        if (!messageText || !messageText.trim()) return;

        // 1. Render User Message Bubble
        renderUserMessage(messageText);

        // 2. Clear input field & reset char counter
        userInput.value = '';
        charCountDisplay.textContent = `0 / ${MAX_CHAR_LIMIT}`;

        // 3. Hide suggestion chips after user starts chatting
        if (suggestionChipsContainer && suggestionChipsContainer.parentNode === chatMessages) {
            suggestionChipsContainer.remove();
        }

        // 4. Set Loading State (Disable input/button, show typing indicator)
        setLoadingState(true);

        try {
            // 5. Phase 6 & 9: Send POST request to Flask `/chat` API
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify({ message: messageText })
            });

            // Phase 10: Check HTTP Response Status
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                const serverErrorMessage = errorData.response || `Server returned status code ${response.status}`;
                throw new Error(serverErrorMessage);
            }

            const data = await response.json();

            // 6. Render Bot Response with Typewriter Effect
            await renderBotMessage(data.response, data.intent, data.confidence);

        } catch (error) {
            console.error("Chatbot API Error:", error);
            // Phase 10: Gracefully display network / server error message
            renderErrorMessage("⚠️ " + (error.message || "Failed to communicate with the server. Please check your network connection."));
        } finally {
            // 7. Reset Loading State
            setLoadingState(false);
            scrollToBottom();
        }
    }

    // ------------------------------------------------------------------
    // 5. UI Message Rendering Helpers
    // ------------------------------------------------------------------
    function renderUserMessage(text) {
        const timeStr = getCurrentFormattedTime();
        
        const row = document.createElement('div');
        row.className = 'message-row user-row';
        row.innerHTML = `
            <div class="user-avatar-icon">
                <i class="fa-solid fa-user"></i>
            </div>
            <div class="msg-content-col">
                <div class="msg-bubble user-bubble">
                    <div class="msg-header">
                        <span class="sender-name">You</span>
                        <span class="msg-time">${timeStr}</span>
                    </div>
                    <p class="msg-text">${escapeHTML(text)}</p>
                </div>
            </div>
        `;

        chatMessages.appendChild(row);
        scrollToBottom();
    }

    function renderBotMessage(text, intentTag = null, confidence = null) {
        return new Promise((resolve) => {
            const timeStr = getCurrentFormattedTime();
            
            const row = document.createElement('div');
            row.className = 'message-row bot-row';

            // Create metadata pills HTML if intent and confidence are provided
            let metaHtml = '';
            if (intentTag && confidence !== null) {
                const confPercent = (confidence * 100).toFixed(1);
                const confClass = confidence < 0.3 ? 'low' : '';
                metaHtml = `
                    <div class="msg-meta-footer">
                        <span class="meta-badge intent-badge" title="Predicted Intent"><i class="fa-solid fa-tag"></i> ${escapeHTML(intentTag)}</span>
                        <span class="meta-badge confidence-badge ${confClass}" title="TF-IDF Cosine Similarity Score"><i class="fa-solid fa-gauge-high"></i> ${confPercent}% Match</span>
                    </div>
                `;
            }

            row.innerHTML = `
                <div class="msg-avatar-col">
                    <img src="${BOT_AVATAR_URL}" alt="Bot" class="msg-avatar">
                </div>
                <div class="msg-content-col">
                    <div class="msg-bubble bot-bubble">
                        <div class="msg-header">
                            <span class="sender-name">SmartAssist</span>
                            <span class="msg-time">${timeStr}</span>
                        </div>
                        <p class="msg-text" id="typewriter-target"></p>
                        ${metaHtml}
                    </div>
                </div>
            `;

            chatMessages.appendChild(row);
            scrollToBottom();

            // Typewriter Animation Execution
            const targetP = row.querySelector('#typewriter-target');
            targetP.removeAttribute('id'); // Clean ID attribute

            let charIndex = 0;
            const speedMs = 15; // Speed per character in milliseconds

            function typeChar() {
                if (charIndex < text.length) {
                    targetP.textContent += text.charAt(charIndex);
                    charIndex++;
                    scrollToBottom();
                    setTimeout(typeChar, speedMs);
                } else {
                    resolve();
                }
            }

            typeChar();
        });
    }

    function renderErrorMessage(errorMessage) {
        const row = document.createElement('div');
        row.className = 'message-row bot-row';
        row.innerHTML = `
            <div class="msg-avatar-col">
                <img src="${BOT_AVATAR_URL}" alt="Bot" class="msg-avatar">
            </div>
            <div class="msg-content-col">
                <div class="msg-bubble bot-bubble" style="border-color: rgba(239, 68, 68, 0.4); background: rgba(239, 68, 68, 0.1);">
                    <div class="msg-header">
                        <span class="sender-name" style="color: #f87171;">System Error</span>
                        <span class="msg-time">${getCurrentFormattedTime()}</span>
                    </div>
                    <p class="msg-text" style="color: #fca5a5;">${escapeHTML(errorMessage)}</p>
                </div>
            </div>
        `;
        chatMessages.appendChild(row);
        scrollToBottom();
    }

    // ------------------------------------------------------------------
    // 6. Utility Functions
    // ------------------------------------------------------------------
    function setLoadingState(isLoading) {
        if (isLoading) {
            userInput.disabled = true;
            sendBtn.disabled = true;
            typingIndicator.classList.remove('hidden');
            chatMessages.appendChild(typingIndicator); // Ensure it renders at bottom
        } else {
            userInput.disabled = false;
            sendBtn.disabled = userInput.value.trim().length === 0;
            typingIndicator.classList.add('hidden');
            userInput.focus();
        }
        scrollToBottom();
    }

    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function getCurrentFormattedTime() {
        const now = new Date();
        return now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }

    function escapeHTML(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }
});
