document.getElementById('registerForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const user_name = document.getElementById('user_name').value;
    const senha = document.getElementById('senha').value;
    const confirm_senha = document.getElementById('confirm_senha').value;
    const messageDiv = document.getElementById('message');
    const submitButton = e.target.querySelector('.btn-primary');

    // Limpar mensagens anteriores
    messageDiv.textContent = '';
    messageDiv.className = '';

    // Validar campos
    if (!user_name || !senha || !confirm_senha) {
        messageDiv.className = 'alert alert-error';
        messageDiv.textContent = 'Por favor, preencha todos os campos';
        return;
    }

    // Validar tamanho mínimo da senha
    if (senha.length < 4) {
        messageDiv.className = 'alert alert-error';
        messageDiv.textContent = 'A senha deve ter no mínimo 4 caracteres';
        return;
    }

    // Validar senhas
    if (senha !== confirm_senha) {
        messageDiv.className = 'alert alert-error';
        messageDiv.textContent = 'As senhas não coincidem';
        return;
    }

    // Desabilitar botão durante o envio
    submitButton.disabled = true;
    submitButton.textContent = 'Criando conta...';

    try {
        const response = await fetch('/register', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ user_name, senha })
        });

        const data = await response.json();

        if (response.ok) {
            messageDiv.className = 'alert alert-success';
            messageDiv.textContent = 'Conta criada com sucesso! Redirecionando para login...';

            // Redirecionar para login após 2 segundos
            setTimeout(() => {
                window.location.href = 'login.html';
            }, 2000);
        } else {
            messageDiv.className = 'alert alert-error';
            messageDiv.textContent = data.detail || 'Erro ao criar conta';
            submitButton.disabled = false;
            submitButton.textContent = 'Criar Conta';
        }
    } catch (error) {
        messageDiv.className = 'alert alert-error';
        messageDiv.textContent = 'Erro de conexão com o servidor';
        submitButton.disabled = false;
        submitButton.textContent = 'Criar Conta';
    }
});
