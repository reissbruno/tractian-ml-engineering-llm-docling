// Tractian Chat JavaScript

// Configurar marked.js
if (typeof marked !== 'undefined') {
    marked.setOptions({
        breaks: true,
        gfm: true,
        headerIds: false,
        mangle: false
    });
}

// Elementos do DOM
const chatForm = document.getElementById('chatForm');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const chatMessages = document.getElementById('chatMessages');
const newChatBtn = document.getElementById('newChatBtn');
const logoutBtn = document.getElementById('logoutBtn');
const sidebarToggle = document.getElementById('sidebarToggle');
const sidebar = document.getElementById('sidebar');
const suggestionCards = document.querySelectorAll('.suggestion-card');
const userProfile = document.getElementById('userProfile');
const modelSelectorBtn = document.getElementById('modelSelectorBtn');
const modelDropdown = document.getElementById('modelDropdown');
const selectedModelName = document.getElementById('selectedModelName');

// Estado da aplicação
let conversationId = null;
let isWaitingResponse = false;
let hasDocuments = false;
let selectedModel = {
    provider: 'openai',
    model: 'gpt-4o-mini-2024-07-18',
    displayName: 'GPT-4o Mini'
};

// Inicialização
document.addEventListener('DOMContentLoaded', async () => {
    // Verificar autenticação
    checkAuth();

    // Verificar se há documentos disponíveis
    await checkDocuments();

    // Ajustar altura do textarea automaticamente
    messageInput.addEventListener('input', adjustTextareaHeight);

    // Habilitar/desabilitar botão de envio
    messageInput.addEventListener('input', toggleSendButton);

    // Enviar com Enter (sem Shift)
    messageInput.addEventListener('keydown', handleKeyDown);

    // Enviar mensagem
    chatForm.addEventListener('submit', handleSubmit);

    // Nova conversa
    newChatBtn.addEventListener('click', startNewConversation);

    // Logout
    logoutBtn.addEventListener('click', handleLogout);

    // Sidebar toggle
    sidebarToggle.addEventListener('click', toggleSidebar);

    // User profile - navegar para documents
    userProfile.addEventListener('click', () => {
        window.location.href = '/static/documents.html';
    });

    // Suggestion cards
    suggestionCards.forEach(card => {
        card.addEventListener('click', () => {
            const prompt = card.dataset.prompt;
            messageInput.value = prompt;
            toggleSendButton();
            messageInput.focus();
        });
    });

    // Model selector
    modelSelectorBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        toggleModelDropdown();
    });

    // Fechar dropdown ao clicar fora
    document.addEventListener('click', (e) => {
        if (!modelSelectorBtn.contains(e.target) && !modelDropdown.contains(e.target)) {
            closeModelDropdown();
        }
    });

    // Model options
    const modelOptions = document.querySelectorAll('.model-option');
    modelOptions.forEach(option => {
        option.addEventListener('click', () => {
            selectModel(option);
        });
    });

    // Carregar modelo selecionado do localStorage
    loadSelectedModel();
});

// Verificar autenticação
function checkAuth() {
    const token = localStorage.getItem('access_token');

    if (!token) {
        window.location.href = '/static/login.html';
        return;
    }

    // Obter informações do usuário (simulado - você pode fazer uma requisição real)
    const userName = localStorage.getItem('user_name') || 'Usuário';
    document.getElementById('userName').textContent = userName;
    document.getElementById('userInitials').textContent = userName.charAt(0).toUpperCase();
}

// Verificar se há documentos disponíveis
async function checkDocuments() {
    try {
        const token = localStorage.getItem('access_token');
        const response = await fetch('/documents', {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });

        if (response.ok) {
            const data = await response.json();
            hasDocuments = data.documents && data.documents.length > 0;
            updateChatState();
        }
    } catch (error) {
        console.error('Erro ao verificar documentos:', error);
        hasDocuments = false;
        updateChatState();
    }
}

// Atualizar estado do chat baseado na disponibilidade de documentos
function updateChatState() {
    if (!hasDocuments) {
        // Desabilitar input
        messageInput.disabled = true;
        messageInput.placeholder = 'Você precisa enviar documentos antes de fazer perguntas';
        sendBtn.disabled = true;

        // Adicionar mensagem informativa na tela
        const welcomeSection = document.querySelector('.welcome-section');
        if (welcomeSection) {
            const infoDiv = document.createElement('div');
            infoDiv.className = 'no-documents-warning';
            infoDiv.innerHTML = `
                <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="12" cy="12" r="10"></circle>
                    <line x1="12" y1="8" x2="12" y2="12"></line>
                    <line x1="12" y1="16" x2="12.01" y2="16"></line>
                </svg>
                <h3>Nenhum documento encontrado</h3>
                <p>Você precisa fazer upload de documentos antes de começar a fazer perguntas ao assistente.</p>
                <button class="btn-go-documents" onclick="window.location.href='/static/documents.html'">
                    Ir para Documentos
                </button>
            `;
            welcomeSection.appendChild(infoDiv);
        }
    } else {
        // Habilitar input
        messageInput.disabled = false;
        messageInput.placeholder = 'Pergunte qualquer coisa...';

        // Remover mensagem de aviso se existir
        const warning = document.querySelector('.no-documents-warning');
        if (warning) {
            warning.remove();
        }
    }
}

// Ajustar altura do textarea
function adjustTextareaHeight() {
    messageInput.style.height = 'auto';
    messageInput.style.height = Math.min(messageInput.scrollHeight, 150) + 'px';
}

// Habilitar/desabilitar botão de envio
function toggleSendButton() {
    const hasText = messageInput.value.trim().length > 0;
    sendBtn.disabled = !hasText || isWaitingResponse || !hasDocuments;
}

// Toggle da sidebar
function toggleSidebar() {
    sidebar.classList.toggle('collapsed');
}

// Capturar Enter para enviar mensagem
function handleKeyDown(e) {
    // Enter sem Shift = enviar mensagem
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (!sendBtn.disabled) {
            chatForm.dispatchEvent(new Event('submit'));
        }
    }
}

// Enviar mensagem
async function handleSubmit(e) {
    e.preventDefault();

    const message = messageInput.value.trim();

    if (!message || isWaitingResponse) return;

    // Limpar input
    messageInput.value = '';
    adjustTextareaHeight();
    toggleSendButton();

    // Remover welcome section se existir
    const welcomeSection = document.querySelector('.welcome-section');
    if (welcomeSection) {
        welcomeSection.remove();
    }

    // Adicionar mensagem do usuário
    addMessage(message, 'user');

    // Adicionar indicador de loading
    const loadingId = addLoadingMessage();

    // Desabilitar entrada
    isWaitingResponse = true;
    messageInput.disabled = true;

    try {
        const token = localStorage.getItem('access_token');

        const response = await fetch('/question', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`,
                'X-LLM-Provider': selectedModel.provider,
                'X-LLM-Model': selectedModel.model
            },
            body: JSON.stringify({
                question: message
            })
        });

        if (!response.ok) {
            throw new Error('Erro ao enviar mensagem');
        }

        const data = await response.json();

        // Remover loading
        removeLoadingMessage(loadingId);

        // Adicionar resposta do assistente com referências
        if (data.references && data.references.length > 0) {
            // Formatar resposta com referências
            let formattedResponse = data.answer + '\n\n**Referências:**\n';
            data.references.forEach((ref, idx) => {
                formattedResponse += `\n${idx + 1}. ${ref}`;
            });
            addMessage(formattedResponse, 'assistant');
        } else {
            addMessage(data.answer, 'assistant');
        }

    } catch (error) {
        console.error('Erro:', error);
        removeLoadingMessage(loadingId);
        addMessage('Desculpe, ocorreu um erro ao processar sua mensagem. Tente novamente.', 'assistant', true);
    } finally {
        // Reabilitar entrada
        isWaitingResponse = false;
        messageInput.disabled = false;
        messageInput.focus();
        toggleSendButton();
    }
}

// Adicionar mensagem ao chat
function addMessage(text, sender, isError = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';

    if (sender === 'user') {
        const userName = localStorage.getItem('user_name') || 'U';
        avatar.textContent = userName.charAt(0).toUpperCase();
    } else {
        avatar.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>
                <circle cx="12" cy="12" r="3"></circle>
            </svg>
        `;
    }

    const content = document.createElement('div');
    content.className = 'message-content';

    // Se for mensagem do assistente, renderizar markdown
    if (sender === 'assistant' && !isError) {
        content.innerHTML = marked.parse(text);
    } else {
        content.textContent = text;
    }

    if (isError) {
        content.style.color = '#DC2626';
    }

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(content);

    chatMessages.appendChild(messageDiv);

    // Scroll para o final
    scrollToBottom();
}

// Adicionar indicador de loading
function addLoadingMessage() {
    const loadingId = 'loading-' + Date.now();
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant loading';
    messageDiv.id = loadingId;

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.innerHTML = `
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>
            <circle cx="12" cy="12" r="3"></circle>
        </svg>
    `;

    const content = document.createElement('div');
    content.className = 'message-content';
    content.innerHTML = `
        <div class="loading-dot"></div>
        <div class="loading-dot"></div>
        <div class="loading-dot"></div>
    `;

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(content);

    chatMessages.appendChild(messageDiv);
    scrollToBottom();

    return loadingId;
}

// Remover indicador de loading
function removeLoadingMessage(loadingId) {
    const loadingElement = document.getElementById(loadingId);
    if (loadingElement) {
        loadingElement.remove();
    }
}

// Scroll para o final
function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Iniciar nova conversa
function startNewConversation() {
    conversationId = null;

    // Limpar mensagens
    chatMessages.innerHTML = `
        <div class="welcome-section">
            <div class="welcome-icon">
                <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                    <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>
                    <circle cx="12" cy="12" r="3"></circle>
                </svg>
            </div>
            <h2>Bem-vindo ao Tractian RAG</h2>
            <p>Faça perguntas sobre manuais técnicos, equipamentos e procedimentos</p>

            <div class="suggestion-cards">
                <button class="suggestion-card" onclick="useSuggestion('Como funciona o sistema de monitoramento?')">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M12 20h9"></path>
                        <path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"></path>
                    </svg>
                    <span>Como funciona o sistema de monitoramento?</span>
                </button>

                <button class="suggestion-card" onclick="useSuggestion('Quais são os principais sensores?')">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <circle cx="12" cy="12" r="10"></circle>
                        <line x1="12" y1="16" x2="12" y2="12"></line>
                        <line x1="12" y1="8" x2="12.01" y2="8"></line>
                    </svg>
                    <span>Quais são os principais sensores?</span>
                </button>

                <button class="suggestion-card" onclick="useSuggestion('Explique sobre manutenção preditiva')">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
                    </svg>
                    <span>Explique sobre manutenção preditiva</span>
                </button>
            </div>
        </div>
    `;

    messageInput.focus();
}

// Usar sugestão
function useSuggestion(text) {
    messageInput.value = text;
    toggleSendButton();
    messageInput.focus();
}

// Logout
function handleLogout() {
    if (confirm('Tem certeza que deseja sair?')) {
        localStorage.removeItem('access_token');
        localStorage.removeItem('user_name');
        window.location.href = '/static/login.html';
    }
}

// Toggle model dropdown
function toggleModelDropdown() {
    const isOpen = modelDropdown.classList.contains('open');
    if (isOpen) {
        closeModelDropdown();
    } else {
        openModelDropdown();
    }
}

// Open model dropdown
function openModelDropdown() {
    modelDropdown.classList.add('open');
    modelSelectorBtn.classList.add('open');
}

// Close model dropdown
function closeModelDropdown() {
    modelDropdown.classList.remove('open');
    modelSelectorBtn.classList.remove('open');
}

// Select model
function selectModel(optionElement) {
    const provider = optionElement.dataset.provider;
    const model = optionElement.dataset.model;
    const displayName = optionElement.querySelector('.model-option-name').textContent;

    document.querySelectorAll('.model-option').forEach(opt => {
        opt.classList.remove('selected');
    });

    optionElement.classList.add('selected');

    selectedModel = {
        provider: provider,
        model: model,
        displayName: displayName
    };

    selectedModelName.textContent = displayName;

    localStorage.setItem('selected_model', JSON.stringify(selectedModel));

    closeModelDropdown();

    console.log('Modelo selecionado:', selectedModel);
}

// Load selected model from localStorage
function loadSelectedModel() {
    const savedModel = localStorage.getItem('selected_model');
    if (savedModel) {
        try {
            selectedModel = JSON.parse(savedModel);
            selectedModelName.textContent = selectedModel.displayName;

            document.querySelectorAll('.model-option').forEach(opt => {
                opt.classList.remove('selected');
                if (opt.dataset.provider === selectedModel.provider &&
                    opt.dataset.model === selectedModel.model) {
                    opt.classList.add('selected');
                }
            });
        } catch (e) {
            console.error('Erro ao carregar modelo salvo:', e);
        }
    }
}
