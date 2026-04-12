// 等待DOM加载完成
document.addEventListener('DOMContentLoaded', function() {
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatHistory = document.getElementById('chat-history');

    // 监听表单提交事件
    chatForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const question = userInput.value.trim();
        
        if (question) {
            // 添加用户问题到对话历史
            addMessageToHistory('user', question);
            
            // 清空输入框
            userInput.value = '';
            
            // 显示加载状态
            const loadingMessageId = addLoadingMessage();
            
            // 发送请求到后端API
            fetch('/api/rag', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ question: question })
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('网络请求失败');
                }
                return response.json();
            })
            .then(data => {
                // 移除加载状态
                removeLoadingMessage(loadingMessageId);
                
                // 添加AI回答到对话历史
                addMessageToHistory('ai', data.answer);
            })
            .catch(error => {
                // 移除加载状态
                removeLoadingMessage(loadingMessageId);
                
                // 显示错误信息
                addMessageToHistory('ai', `抱歉，处理请求时出现错误：${error.message}`);
                console.error('请求错误:', error);
            });
        }
    });

    // 添加消息到对话历史
    function addMessageToHistory(sender, message) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `flex mb-4 ${sender === 'user' ? 'justify-end' : ''}`;
        
        if (sender === 'user') {
            messageDiv.innerHTML = `
                <div class="bg-primary text-white p-3 rounded-lg shadow-sm max-w-[80%]">
                    <p>${message}</p>
                </div>
                <div class="flex-shrink-0 w-8 h-8 rounded-full bg-gray-300 flex items-center justify-center ml-3">
                    <i class="fa fa-user"></i>
                </div>
            `;
        } else {
            messageDiv.innerHTML = `
                <div class="flex-shrink-0 w-8 h-8 rounded-full bg-primary flex items-center justify-center text-white">
                    <i class="fa fa-robot"></i>
                </div>
                <div class="ml-3 bg-white p-3 rounded-lg shadow-sm max-w-[80%]">
                    <p>${message}</p>
                </div>
            `;
        }
        
        chatHistory.appendChild(messageDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;
        
        return messageDiv;
    }

    // 添加加载状态
    function addLoadingMessage() {
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'flex mb-4';
        loadingDiv.id = 'loading-message';
        loadingDiv.innerHTML = `
            <div class="flex-shrink-0 w-8 h-8 rounded-full bg-primary flex items-center justify-center text-white">
                <i class="fa fa-robot"></i>
            </div>
            <div class="ml-3 bg-white p-3 rounded-lg shadow-sm flex items-center">
                <div class="animate-pulse flex space-x-2">
                    <div class="h-2 w-2 bg-gray-400 rounded-full"></div>
                    <div class="h-2 w-2 bg-gray-400 rounded-full"></div>
                    <div class="h-2 w-2 bg-gray-400 rounded-full"></div>
                </div>
            </div>
        `;
        
        chatHistory.appendChild(loadingDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;
        
        return 'loading-message';
    }

    // 移除加载状态
    function removeLoadingMessage(id) {
        const loadingDiv = document.getElementById(id);
        if (loadingDiv) {
            loadingDiv.remove();
        }
    }
});