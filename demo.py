import ollama

# Define your prompt
query = "Explain the concept of quantum entanglement in simple terms."

# Send query to the Mistral model running in Ollama
response = ollama.chat(
    model='mistral',
    messages=[
        {'role': 'user', 'content': query}
    ]
)

# Print the response from the model
print("Mistral response:\n", response['message']['content'])
