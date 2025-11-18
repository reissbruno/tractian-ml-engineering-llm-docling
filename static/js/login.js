document.getElementById('loginForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const user_name = document.getElementById('user_name').value;
    const senha = document.getElementById('senha').value;
    const messageDiv = document.getElementById('message');
    const submitButton = e.target.querySelector('.btn-primary');

    // Desabilitar botão durante o envio
    submitButton.disabled = true;
    submitButton.textContent = 'Entrando...';

    try {
        const response = await fetch('/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ user_name, senha })
        });

        const data = await response.json();

        if (response.ok) {
            messageDiv.className = 'alert alert-success';
            messageDiv.textContent = 'Login realizado com sucesso! Redirecionando...';

            // Armazenar token e nome do usuário
            localStorage.setItem('access_token', data.access_token);
            localStorage.setItem('user_name', user_name);

            // Redirecionar após 1 segundo
            setTimeout(() => {
                window.location.href = '/static/chat.html';
            }, 1000);
        } else {
            messageDiv.className = 'alert alert-error';
            messageDiv.textContent = data.detail || 'Erro ao fazer login';
            submitButton.disabled = false;
            submitButton.textContent = 'Entrar';
        }
    } catch (error) {
        messageDiv.className = 'alert alert-error';
        messageDiv.textContent = 'Erro de conexão com o servidor';
        submitButton.disabled = false;
        submitButton.textContent = 'Entrar';
    }
});
