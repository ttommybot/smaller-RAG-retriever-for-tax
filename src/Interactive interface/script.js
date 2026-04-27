// 等待DOM加载完成
document.addEventListener('DOMContentLoaded', function() {
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatHistory = document.getElementById('chat-history');
    const modelSelect = document.getElementById('model-select');
    const chunkSelect = document.getElementById('chunk-select');
    const rerankerToggle = document.getElementById('reranker-toggle');
    const rerankerIndicator = document.getElementById('reranker-indicator');
    const rerankerStatus = document.getElementById('reranker-status');
    const applyConfigBtn = document.getElementById('apply-config');
    const configStatus = document.getElementById('config-status');
    const configInfo = document.getElementById('config-info');

    // Reranker 状态
    let useReranker = false;

    // 系统是否已初始化
    let systemInitialized = false;

    // ==================== 加载模型列表 ====================
    loadModels();

    function loadModels() {
        fetch('/api/models')
            .then(response => response.json())
            .then(data => {
                modelSelect.innerHTML = '<option value="">请选择模型...</option>';
                data.models.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model.name;
                    option.textContent = model.name;
                    option.dataset.chunkMethods = model.chunk_methods.join(',');
                    modelSelect.appendChild(option);
                });
            })
            .catch(error => {
                console.error('加载模型列表失败:', error);
                modelSelect.innerHTML = '<option value="">加载失败</option>';
            });
    }

    // ==================== 模型选择变化时更新 chunk 选项 ====================
    modelSelect.addEventListener('change', function() {
        const selectedOption = this.options[this.selectedIndex];
        if (selectedOption.value && selectedOption.dataset.chunkMethods) {
            const chunkMethods = selectedOption.dataset.chunkMethods.split(',');
            chunkSelect.innerHTML = '';
            chunkMethods.forEach(method => {
                const option = document.createElement('option');
                option.value = method;
                option.textContent = method;
                chunkSelect.appendChild(option);
            });
        } else {
            chunkSelect.innerHTML = `
                <option value="semantic">semantic</option>
                <option value="sliding">sliding</option>
            `;
        }
    });

    // ==================== Reranker 开关 ====================
    rerankerToggle.addEventListener('click', function() {
        useReranker = !useReranker;
        if (useReranker) {
            rerankerToggle.classList.remove('bg-gray-300');
            rerankerToggle.classList.add('bg-primary');
            rerankerIndicator.classList.remove('translate-x-1');
            rerankerIndicator.classList.add('translate-x-6');
            rerankerStatus.textContent = '开启';
            rerankerStatus.classList.remove('text-gray-500');
            rerankerStatus.classList.add('text-primary');
        } else {
            rerankerToggle.classList.remove('bg-primary');
            rerankerToggle.classList.add('bg-gray-300');
            rerankerIndicator.classList.remove('translate-x-6');
            rerankerIndicator.classList.add('translate-x-1');
            rerankerStatus.textContent = '关闭';
            rerankerStatus.classList.remove('text-primary');
            rerankerStatus.classList.add('text-gray-500');
        }
    });

    // ==================== 应用配置 ====================
    applyConfigBtn.addEventListener('click', function() {
        const modelName = modelSelect.value;
        const chunkMethod = chunkSelect.value;

        if (!modelName) {
            showConfigStatus('请先选择模型', false);
            return;
        }

        applyConfigBtn.disabled = true;
        applyConfigBtn.innerHTML = '<i class="fa fa-spinner fa-spin mr-1"></i> 加载中...';

        fetch('/api/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model_name: modelName,
                chunk_method: chunkMethod,
                use_reranker: useReranker
            })
        })
        .then(response => response.json())
        .then(data => {
            applyConfigBtn.disabled = false;
            applyConfigBtn.innerHTML = '<i class="fa fa-check mr-1"></i> 应用配置';

            if (data.success) {
                systemInitialized = true;
                showConfigStatus(
                    `已加载：${modelName} (${chunkMethod}) | Reranker: ${useReranker ? '开启' : '关闭'} | ${data.num_chunks} chunks`,
                    true
                );
                addMessageToHistory('ai', `配置已应用！当前使用 ${modelName} 模型，可以开始提问了。`);
            } else {
                showConfigStatus(`配置失败：${data.error}`, false);
            }
        })
        .catch(error => {
            applyConfigBtn.disabled = false;
            applyConfigBtn.innerHTML = '<i class="fa fa-check mr-1"></i> 应用配置';
            showConfigStatus(`请求失败：${error.message}`, false);
        });
    });

    function showConfigStatus(message, success) {
        configStatus.classList.remove('hidden');
        configInfo.textContent = message;
        if (success) {
            configStatus.classList.remove('text-red-500');
            configStatus.classList.add('text-green-600');
        } else {
            configStatus.classList.remove('text-green-600');
            configStatus.classList.add('text-red-500');
        }
    }

    // ==================== 发送消息 ====================
    chatForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const question = userInput.value.trim();

        if (!question) return;

        if (!systemInitialized) {
            addMessageToHistory('ai', '请先选择模型并点击"应用配置"。');
            return;
        }

        // 添加用户问题到对话历史
        addMessageToHistory('user', question);

        // 清空输入框
        userInput.value = '';

        // 显示加载状态
        const loadingMessageId = addLoadingMessage();

        // 发送请求到后端API
        fetch('/api/rag', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: question })
        })
        .then(response => response.json())
        .then(data => {
            // 移除加载状态
            removeLoadingMessage(loadingMessageId);

            if (data.success) {
                // 添加AI回答到对话历史
                let answerHtml = `<p>${data.answer}</p>`;

                // 显示效率信息
                if (data.efficiency) {
                    answerHtml += `<p class="text-xs text-gray-400 mt-2">
                        检索: ${data.efficiency.retrieval_ms.toFixed(1)}ms |
                        重排: ${data.efficiency.rerank_ms.toFixed(1)}ms |
                        总计: ${data.efficiency.total_ms.toFixed(1)}ms
                    </p>`;
                }

                // 显示来源
                if (data.sources && data.sources.length > 0) {
                    answerHtml += `<p class="text-xs text-gray-500 mt-2">
                        <i class="fa fa-file-text-o mr-1"></i> 来源: ${data.sources.join(', ')}
                    </p>`;
                }

                addMessageToHistory('ai', answerHtml);
            } else {
                addMessageToHistory('ai', `抱歉，处理请求时出现错误：${data.error}`);
            }
        })
        .catch(error => {
            removeLoadingMessage(loadingMessageId);
            addMessageToHistory('ai', `抱歉，处理请求时出现错误：${error.message}`);
            console.error('请求错误:', error);
        });
    });

    // ==================== 辅助函数 ====================

    // 添加消息到对话历史
    function addMessageToHistory(sender, message) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `flex mb-4 ${sender === 'user' ? 'justify-end' : ''}`;

        if (sender === 'user') {
            messageDiv.innerHTML = `
                <div class="bg-primary text-white p-3 rounded-lg shadow-sm max-w-[80%]">
                    <p>${escapeHtml(message)}</p>
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
                    ${message}
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

    // HTML转义
    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
});