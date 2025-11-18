// Verificar autenticação
const token = localStorage.getItem('access_token');
const userName = localStorage.getItem('user_name');

if (!token || !userName) {
    window.location.href = '/static/login.html';
}

// Mapa de conexões SSE ativas para evitar duplicatas
const activeSSEConnections = new Map();

// Configurar perfil do usuário
document.addEventListener('DOMContentLoaded', () => {
    const userNameElement = document.getElementById('userName');
    const userAvatarElement = document.getElementById('userAvatar');

    if (userNameElement && userName) {
        userNameElement.textContent = userName;
    }

    if (userAvatarElement && userName) {
        userAvatarElement.textContent = userName.charAt(0).toUpperCase();
    }

    // Carregar documentos salvos
    loadDocuments();
});

// Logout
document.getElementById('logoutBtn').addEventListener('click', () => {
    localStorage.removeItem('access_token');
    localStorage.removeItem('user_name');
    window.location.href = '/static/login.html';
});

// Botão de upload (ainda não implementado)
document.getElementById('uploadBtn').addEventListener('click', () => {
    document.getElementById('fileInput').click();
});

// Input de arquivo - Upload com processamento completo (vector store)
document.getElementById('fileInput').addEventListener('change', async (e) => {
    const files = e.target.files;
    if (files.length > 0) {
        console.log('Arquivos selecionados:', files);

        // Validar que todos são PDFs
        const invalidFiles = Array.from(files).filter(f => f.type !== 'application/pdf');
        if (invalidFiles.length > 0) {
            alert(`Arquivos não-PDF encontrados: ${invalidFiles.map(f => f.name).join(', ')}. Apenas PDFs são aceitos.`);
            return;
        }

        // Upload em batch (todos de uma vez)
        await uploadDocumentsBatch(files);
    }
});

// Carregar lista de documentos
async function loadDocuments() {
    const documentsList = document.getElementById('documentsList');
    const emptyState = document.getElementById('emptyState');
    const documentsCount = document.getElementById('documentsCount');

    try {
        // Buscar documentos da API
        const response = await fetch('/documents', {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });

        if (!response.ok) {
            throw new Error('Erro ao carregar documentos');
        }

        const data = await response.json();
        const documents = data.documents || [];

        // Preservar cards em processamento que já têm SSE ativo
        const activeUploads = new Set(activeSSEConnections.keys());

        // Limpar cards de documentos que foram deletados
        const allCards = documentsList.querySelectorAll('[data-doc-id]');
        allCards.forEach(card => {
            const cardDocId = card.getAttribute('data-doc-id');
            // Se não está em activeUploads E não está na lista de documentos do servidor, remover
            if (!activeUploads.has(cardDocId) && !documents.find(d => d.id === cardDocId)) {
                card.remove();
            }
        });

        if (documents.length === 0 && activeUploads.size === 0) {
            emptyState.style.display = 'flex';
            documentsList.style.display = 'none';
            documentsCount.textContent = '0 documentos';
        } else {
            emptyState.style.display = 'none';
            documentsList.style.display = 'grid';

            const totalCount = documents.length + activeUploads.size;
            documentsCount.textContent = `${totalCount} documento${totalCount !== 1 ? 's' : ''}`;

            // Renderizar apenas documentos que NÃO estão em upload ativo
            const documentsToRender = documents.filter(doc => !activeUploads.has(doc.id));

            // Se há uploads ativos, adicionar documentos novos sem destruir uploads
            if (activeUploads.size > 0) {
                console.log('Preservando cards de upload ativos:', Array.from(activeUploads));
                // Adicionar apenas novos documentos completed/error
                documentsToRender.forEach(doc => {
                    // Verificar se já existe
                    if (!documentsList.querySelector(`[data-doc-id="${doc.id}"]`)) {
                        documentsList.insertAdjacentHTML('beforeend', createDocumentCard(doc));
                    }
                });
            } else {
                // Sem uploads ativos, recriar tudo normalmente
                documentsList.innerHTML = documentsToRender.map(doc => createDocumentCard(doc)).join('');
            }

            // Reconectar SSE apenas para documentos em processamento SEM conexão ativa
            documents.forEach(doc => {
                if (doc.status === 'processing' && !activeSSEConnections.has(doc.id)) {
                    console.log('Reconectando SSE para documento:', doc.id);
                    monitorDocumentProgress(doc.id, doc.filename);
                }
            });
        }
    } catch (error) {
        console.error('Erro ao carregar documentos:', error);
        emptyState.style.display = 'flex';
        documentsList.style.display = 'none';
        documentsCount.textContent = '0 documentos';
    }
}

// Criar card de documento
function createDocumentCard(doc) {
    const statusMap = {
        'completed': { text: 'Processado', class: 'status-completed' },
        'processing': { text: 'Processando', class: 'status-processing' },
        'error': { text: 'Erro', class: 'status-error' }
    };

    const status = statusMap[doc.status] || { text: 'Desconhecido', class: 'status-unknown' };
    const createdDate = doc.created_at ? new Date(doc.created_at).toLocaleDateString('pt-BR') : 'N/A';
    const fileSize = doc.file_size ? formatFileSize(doc.file_size) : 'N/A';

    // Se está processando, renderizar card de progresso
    if (doc.status === 'processing') {
        return createProgressCard(doc.filename, doc.id);
    }

    // Card normal para documentos completed/error
    return `
        <div class="document-card" data-doc-id="${doc.id}">
            <div class="document-icon">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
            </div>
            <div class="document-info">
                <h4 class="document-name">${doc.filename}</h4>
                <p class="document-meta">${fileSize} • ${createdDate} • ${doc.chunks_count || 0} chunks</p>
                <span class="document-status ${status.class}">${status.text}</span>
            </div>
            <div class="document-actions">
                <button class="action-btn" onclick="viewDocument('${doc.id}')" title="Ver detalhes">
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                    </svg>
                </button>
                <button class="action-btn" onclick="deleteDocument('${doc.id}')" title="Excluir">
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                    </svg>
                </button>
            </div>
        </div>
    `;
}

// Formatar tamanho de arquivo
function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

// Visualizar documento (ainda não implementado)
function viewDocument(docId) {
    console.log('Visualizar documento:', docId);
    // TODO: Implementar visualização
    alert('Visualização de documento será implementada em breve!');
}

// Deletar documento
async function deleteDocument(docId) {
    console.log('Deletar documento:', docId);

    if (!confirm('Tem certeza que deseja excluir este documento? Esta ação não pode ser desfeita.')) {
        return;
    }

    try {
        const response = await fetch(`/documents/${docId}`, {
            method: 'DELETE',
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Erro ao deletar documento');
        }

        const data = await response.json();
        console.log('Documento deletado:', data);

        // Mostrar feedback visual
        showNotification('Documento deletado com sucesso!', 'success');

        // Recarregar lista de documentos
        await loadDocuments();

    } catch (error) {
        console.error('Erro ao deletar documento:', error);
        showNotification(`Erro ao deletar: ${error.message}`, 'error');
    }
}

// Mostrar notificação
function showNotification(message, type = 'info') {
    // Criar elemento de notificação
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;

    // Adicionar ao body
    document.body.appendChild(notification);

    // Animar entrada
    setTimeout(() => {
        notification.classList.add('notification-show');
    }, 10);

    // Remover após 3 segundos
    setTimeout(() => {
        notification.classList.remove('notification-show');
        setTimeout(() => {
            notification.remove();
        }, 300);
    }, 3000);
}

// Upload de múltiplos documentos com processamento completo (vector store)
async function uploadDocumentsBatch(files) {
    const processorType = document.getElementById('processorType').value || 'fast';
    const fileList = Array.from(files);

    console.log(`Iniciando upload de ${fileList.length} arquivo(s) com processor=${processorType}`);

    // Mostrar indicador de progresso genérico
    const documentsList = document.getElementById('documentsList');
    const emptyState = document.getElementById('emptyState');

    emptyState.style.display = 'none';
    documentsList.style.display = 'grid';

    // Criar card de progresso temporário
    const tempId = `batch-${Date.now()}`;
    const progressCard = `
        <div class="document-card progress-card" data-doc-id="${tempId}">
            <div class="progress-spinner"></div>
            <div class="document-info">
                <h4 class="document-name">Processando ${fileList.length} documento(s)...</h4>
                <p class="progress-status">Enviando arquivos e gerando chunks...</p>
                <div class="progress-details">
                    <span>⚡ ${processorType === 'fast' ? 'FastPDF' : 'Docling'}</span>
                    <span>📊 Criando chunks e embeddings</span>
                    <span>💾 Indexando no vector store</span>
                </div>
            </div>
        </div>
    `;

    documentsList.insertAdjacentHTML('afterbegin', progressCard);
    const progressElement = documentsList.querySelector(`[data-doc-id="${tempId}"]`);
    const statusElement = progressElement.querySelector('.progress-status');

    try {
        // Preparar FormData
        const formData = new FormData();
        fileList.forEach(file => {
            formData.append('files', file);
        });
        formData.append('processor', processorType);

        statusElement.textContent = 'Enviando arquivos para o servidor...';

        // Fazer upload (pode demorar - processamento completo)
        const startTime = Date.now();
        const response = await fetch('/documents', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Erro ao processar documentos');
        }

        const result = await response.json();
        const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);

        console.log('Processamento concluído:', result);

        // Remover card de progresso
        progressElement.remove();

        // Mostrar notificação de sucesso
        showNotification(
            `✅ ${result.documents_indexed} documento(s) processado(s) com sucesso! ` +
            `${result.total_chunks} chunks indexados. (${elapsed}s)`,
            'success'
        );

        // Recarregar lista de documentos
        await loadDocuments();

    } catch (error) {
        console.error('Erro no upload em batch:', error);

        // Remover card de progresso
        if (progressElement) {
            progressElement.remove();
        }

        showNotification(
            `❌ Erro ao processar documentos: ${error.message}`,
            'error'
        );
    }
}

// Upload de documento com SSE (LEGADO - manter para compatibilidade)
async function uploadDocument(file) {
    // Validar tipo de arquivo
    if (file.type !== 'application/pdf') {
        alert(`Arquivo ${file.name} não é PDF. Apenas arquivos PDF são aceitos.`);
        return;
    }

    // Criar card de progresso com ID temporário único
    const tempId = `upload-${Date.now()}`;
    const progressCard = createProgressCard(file.name, tempId);
    const documentsList = document.getElementById('documentsList');
    const emptyState = document.getElementById('emptyState');

    // Esconder empty state e mostrar lista
    emptyState.style.display = 'none';
    documentsList.style.display = 'grid';

    // Adicionar card de progresso no início
    documentsList.insertAdjacentHTML('afterbegin', progressCard);

    // Buscar o card específico pelo ID temporário
    const progressElement = documentsList.querySelector(`[data-doc-id="${tempId}"]`);
    if (!progressElement) {
        console.error('Erro: não conseguiu encontrar card de progresso');
        return;
    }
    const statusElement = progressElement.querySelector('.progress-status');

    try {
        // Fazer upload
        const formData = new FormData();
        formData.append('file', file);

        // Adicionar tipo de processador
        const processorType = document.getElementById('processorType').value || 'fast';
        formData.append('processor', processorType);

        statusElement.textContent = 'Enviando arquivo...';

        const uploadResponse = await fetch('/documents/upload', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`
            },
            body: formData
        });

        if (!uploadResponse.ok) {
            const errorData = await uploadResponse.json();
            throw new Error(errorData.detail || 'Erro ao fazer upload');
        }

        const uploadData = await uploadResponse.json();
        const docId = uploadData.doc_id;

        console.log('Upload iniciado:', uploadData);

        // Atualizar o data-doc-id com o ID real do servidor
        progressElement.setAttribute('data-doc-id', docId);

        statusElement.textContent = 'Processando documento...';

        // Verificar se já existe uma conexão ativa para este documento
        if (activeSSEConnections.has(docId)) {
            console.log('Fechando conexão SSE antiga antes de criar nova para', docId);
            const oldConnection = activeSSEConnections.get(docId);
            oldConnection.close();
            activeSSEConnections.delete(docId);
        }

        // Conectar ao SSE para progresso em tempo real
        console.log('Criando conexão SSE durante upload para', docId);
        const eventSource = new EventSource(`/documents/progress/${docId}`);
        let isClosedByClient = false; // Flag para evitar erro após fechamento intencional

        // Registrar conexão ativa IMEDIATAMENTE
        activeSSEConnections.set(docId, eventSource);

        eventSource.onopen = () => {
            console.log('Conexão SSE aberta com sucesso para', docId);
        };

        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            console.log(`SSE ${docId}:`, data.status, '-', data.message);

            if (data.status === 'processing') {
                statusElement.textContent = data.message;
            } else if (data.status === 'completed') {
                isClosedByClient = true;
                eventSource.close();
                activeSSEConnections.delete(docId);
                statusElement.textContent = 'Documento processado com sucesso!';
                progressElement.classList.add('completed');

                // Remover card de progresso após 2s
                setTimeout(() => {
                    progressElement.remove();
                    loadDocuments();
                }, 2000);
            } else if (data.status === 'error') {
                isClosedByClient = true;
                eventSource.close();
                activeSSEConnections.delete(docId);
                statusElement.textContent = `Erro: ${data.message}`;
                progressElement.classList.add('error');

                // Remover card de erro após 5s
                setTimeout(() => {
                    progressElement.remove();
                    loadDocuments();
                }, 5000);
            } else if (data.status === 'cancelled') {
                isClosedByClient = true;
                eventSource.close();
                activeSSEConnections.delete(docId);
                statusElement.textContent = 'Processamento cancelado';
                progressElement.classList.add('error');

                setTimeout(() => {
                    progressElement.remove();
                    loadDocuments();
                }, 3000);
            }
        };

        eventSource.onerror = (error) => {
            console.error('Erro no SSE durante upload:', error);
            console.log('ReadyState:', eventSource.readyState, 'isClosedByClient:', isClosedByClient);

            // Se já fechamos intencionalmente, ignorar erro
            if (isClosedByClient) {
                console.log('Ignorando erro - conexão foi fechada intencionalmente');
                return;
            }

            // Se a conexão foi fechada pelo servidor normalmente (readyState === 2)
            if (eventSource.readyState === 2) {
                console.log('Conexão SSE fechada normalmente pelo servidor');
                return;
            }

            // Erro real de conexão (readyState === 0 = CONNECTING, 1 = OPEN com erro)
            console.error('Erro real de conexão SSE durante upload');
            eventSource.close();
            activeSSEConnections.delete(docId);
            statusElement.textContent = 'Erro de conexão - Verifique o servidor';
            progressElement.classList.add('error');

            setTimeout(() => {
                progressElement.remove();
                loadDocuments();
            }, 5000);
        };

    } catch (error) {
        console.error('Erro no upload:', error);
        statusElement.textContent = `❌ Erro: ${error.message}`;
        progressElement.classList.add('error');

        setTimeout(() => {
            progressElement.remove();
            loadDocuments();
        }, 5000);
    }
}

// Criar card de progresso
function createProgressCard(filename, docId = null) {
    const dataAttr = docId ? `data-doc-id="${docId}"` : '';
    return `
        <div class="progress-card" ${dataAttr}>
            <div class="document-icon">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
            </div>
            <div class="document-info">
                <h4 class="document-name">${filename}</h4>
                <div class="progress-spinner"></div>
                <p class="progress-status">Processando documento...</p>
            </div>
            ${docId ? `
            <button class="cancel-btn" onclick="cancelProcessing('${docId}')" title="Cancelar e remover">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                </svg>
            </button>
            ` : ''}
        </div>
    `;
}

// Cancelar processamento
async function cancelProcessing(docId) {
    console.log('Cancelando processamento:', docId);

    if (!confirm('Tem certeza que deseja cancelar e remover este documento?')) {
        return;
    }

    try {
        // Fechar conexão SSE se existir
        if (activeSSEConnections.has(docId)) {
            const connection = activeSSEConnections.get(docId);
            connection.close();
            activeSSEConnections.delete(docId);
        }

        // Deletar documento do servidor
        const response = await fetch(`/documents/${docId}`, {
            method: 'DELETE',
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });

        if (!response.ok) {
            throw new Error('Erro ao deletar documento');
        }

        // Remover card
        const card = document.querySelector(`[data-doc-id="${docId}"]`);
        if (card) {
            card.remove();
        }

        showNotification('Processamento cancelado e documento removido', 'success');

        // Recarregar lista
        await loadDocuments();

    } catch (error) {
        console.error('Erro ao cancelar:', error);
        showNotification('Erro ao cancelar processamento', 'error');
    }
}

// Monitorar progresso de documento via SSE
function monitorDocumentProgress(docId, filename) {
    console.log('Iniciando monitoramento SSE para:', docId);

    // Verificar se já existe uma conexão ativa para este documento
    if (activeSSEConnections.has(docId)) {
        console.log('Conexão SSE já existe para', docId, '- reutilizando');
        return;
    }

    // Buscar o card do documento pelo docId
    const progressElement = document.querySelector(`[data-doc-id="${docId}"]`);
    if (!progressElement) {
        console.error('Card de progresso não encontrado para:', docId);
        return;
    }

    const statusElement = progressElement.querySelector('.progress-status');

    if (!statusElement) {
        console.error('Elemento de status não encontrado');
        return;
    }

    // Fechar qualquer conexão antiga primeiro
    if (activeSSEConnections.has(docId)) {
        console.log('Fechando conexão SSE antiga antes de criar nova para', docId);
        const oldConnection = activeSSEConnections.get(docId);
        oldConnection.close();
        activeSSEConnections.delete(docId);
    }

    // Conectar ao SSE para progresso em tempo real
    console.log('Criando nova conexão SSE para', docId);
    const eventSource = new EventSource(`/documents/progress/${docId}`);
    let isClosedByClient = false; // Flag para evitar erro após fechamento intencional

    // Registrar conexão ativa IMEDIATAMENTE
    activeSSEConnections.set(docId, eventSource);

    eventSource.onopen = () => {
        console.log('Conexão SSE aberta com sucesso para', docId);
    };

    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log(`SSE Monitor ${docId}:`, data.status, '-', data.message);

        if (data.status === 'processing') {
            statusElement.textContent = data.message;
        } else if (data.status === 'completed') {
            isClosedByClient = true;
            eventSource.close();
            activeSSEConnections.delete(docId);
            statusElement.textContent = 'Documento processado com sucesso!';
            progressElement.classList.add('completed');

            // Remover card de progresso e recarregar lista após 2s
            setTimeout(() => {
                progressElement.remove();
                loadDocuments();
            }, 2000);
        } else if (data.status === 'error') {
            isClosedByClient = true;
            eventSource.close();
            activeSSEConnections.delete(docId);
            statusElement.textContent = `Erro: ${data.message}`;
            progressElement.classList.add('error');

            // Remover card de erro após 5s
            setTimeout(() => {
                progressElement.remove();
                loadDocuments();
            }, 5000);
        } else if (data.status === 'cancelled') {
            isClosedByClient = true;
            eventSource.close();
            activeSSEConnections.delete(docId);
            statusElement.textContent = 'Processamento cancelado';
            progressElement.classList.add('error');

            setTimeout(() => {
                progressElement.remove();
                loadDocuments();
            }, 3000);
        }
    };

    eventSource.onerror = (error) => {
        console.error('Erro no SSE para', docId, ':', error);
        console.log('ReadyState:', eventSource.readyState, 'isClosedByClient:', isClosedByClient);

        // Se já fechamos intencionalmente, ignorar erro
        if (isClosedByClient) {
            console.log('Ignorando erro - conexão foi fechada intencionalmente');
            return;
        }

        // Se a conexão foi fechada pelo servidor normalmente (readyState === 2)
        if (eventSource.readyState === 2) {
            console.log('Conexão SSE fechada normalmente pelo servidor');
            return;
        }

        // Erro real de conexão (readyState === 0 = CONNECTING, 1 = OPEN com erro)
        console.error('Erro real de conexão SSE');
        eventSource.close();
        activeSSEConnections.delete(docId);
        statusElement.textContent = 'Erro de conexão - Verifique o servidor';
        progressElement.classList.add('error');

        setTimeout(() => {
            progressElement.remove();
            loadDocuments();
        }, 5000);
    };
}

// Drag and drop - Implementado com upload real
const uploadCard = document.querySelector('.upload-card');

uploadCard.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadCard.classList.add('dragover');
});

uploadCard.addEventListener('dragleave', () => {
    uploadCard.classList.remove('dragover');
});

uploadCard.addEventListener('drop', async (e) => {
    e.preventDefault();
    uploadCard.classList.remove('dragover');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        console.log('Arquivos arrastados:', files);

        // Processar cada arquivo
        for (let i = 0; i < files.length; i++) {
            await uploadDocument(files[i]);
        }
    }
});
