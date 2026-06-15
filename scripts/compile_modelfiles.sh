#!/bin/bash
set -e

echo "Compiling Ollama modelfiles..."

echo "Creating ifsca-classifier-3b..."
ollama create ifsca-classifier-3b -f modelfiles/Modelfile.classifier

echo "Creating ifsca-boundary-3b..."
ollama create ifsca-boundary-3b -f modelfiles/Modelfile.boundary

echo "Creating ifsca-extractor-3b..."
ollama create ifsca-extractor-3b -f modelfiles/Modelfile.extractor

echo "Modelfiles compiled successfully!"
