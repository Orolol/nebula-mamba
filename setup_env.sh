#!/bin/bash
set -e

# 1. Setup Pyenv
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"

if command -v pyenv 1>/dev/null 2>&1; then
    eval "$(pyenv init -)"
    eval "$(pyenv virtualenv-init -)"
else
    echo "Pyenv not found in path. Trying to locate..."
    if [ -d "$HOME/.pyenv" ]; then
        echo "Found .pyenv directory."
    else
        echo "Installing pyenv..."
        curl https://pyenv.run | bash
    fi
    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init -)"
    eval "$(pyenv virtualenv-init -)"
fi

# 2. Install Python 3.12.3
if pyenv versions | grep -q "3.12.3"; then
    echo "Python 3.12.3 already installed."
else
    echo "Installing Python 3.12.3..."
    pyenv install 3.12.3
fi

# 3. Create Virtualenv
if pyenv virtualenvs | grep -q "nebula-env"; then
    echo "Virtualenv 'nebula-env' already exists."
else
    echo "Creating virtualenv 'nebula-env'..."
    pyenv virtualenv 3.12.3 nebula-env
fi

# 4. Activate and Install Deps
echo "Activating nebula-env..."
pyenv activate nebula-env

echo "Installing dependencies..."
pip install torch transformers datasets tqdm triton mamba-ssm einops

echo "Environment setup complete."
