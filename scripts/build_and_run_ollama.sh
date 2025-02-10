#!/bin/bash

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Ollama is not installed. Please install it first:"
    echo "curl -fsSL https://ollama.com/install.sh | sh"
    exit 1
fi

# Check if model exists
MODEL_NAME="sentiment-bert"

echo "Building $MODEL_NAME model..."
ollama create $MODEL_NAME -f Modelfile

echo "Starting $MODEL_NAME model..."
ollama run $MODEL_NAME

# Example usage:
# curl -X POST http://localhost:11434/api/generate -d '{
#   "model": "sentiment-bert",
#   "prompt": "This movie was absolutely fantastic!",
#   "stream": false
# }' 