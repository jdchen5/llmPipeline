# Flask is used to create the app, request to handle incoming requests, and jsonify to send JSON responses.
from flask import Flask, request, jsonify  
from transformers import GPT2LMHeadModel, GPT2Tokenizer  #used to load and interact with the GPT-2 model

app = Flask(__name__) # initializes a new Flask application

tokenizer = GPT2Tokenizer.from_pretrained("gpt2") # convert input text into tokens
model = GPT2LMHeadModel.from_pretrained("gpt2") # pre-trained GPT-2 llm

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.json.get("text", None)
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=100)
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"response": response_text})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
