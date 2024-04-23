
import argparse
import torch

from flask import Flask, json, jsonify, request
from utils import (
    LocalLLMCompletion,
    LocalLLMEmbedding,
)


app = Flask(__name__)

@app.route("/v1/chat/completions/models/<model>", methods=["POST"])
def chat_completion_by_local_llm(model):
    return local_llm_chat_completion_request(request,model)


@app.route("/v1/embeddings/models/<model>", methods=["POST"])
def embedding_by_local_llm(model):
    return local_llm_embedding_request(request,model)


def local_llm_chat_completion_request(request,model):
    # global model,tokenizer

    request_data = request.data
    json_data = json.loads(request_data)

    try:
        prompt = json_data["inputs"]
        if "context" in json_data:
            context = json_data["context"]
        else:
            context = ""

        if "max_tokens" in json_data:
            max_tokens = json_data["max_tokens"]
        else:
            max_tokens = 200

        if "temperature" in json_data:
            temperature = json_data["temperature"]
        else:
            temperature = 0.6

        localLLMChatCompetetion = LocalLLMCompletion.LocalLLMCompletion(model)

        chat_completion_text = localLLMChatCompetetion.call_local_llm_chat(prompt, context, max_tokens,temperature)

        data = [{"generated_text": chat_completion_text}]

        return jsonify(data)



    except Exception as e:
        print(e)
        return "Sorry, unable to perform sentence completion with model {}".format(
            model
        )


def local_llm_embedding_request(request,model):

    request_data = request.data
    json_data = json.loads(request_data)

    try:
        sentences = json_data["inputs"]


        localLLmEmbedding = LocalLLMEmbedding.LocalLLMEmbedding(model)
        embeddings, num_prompt_tokens = localLLmEmbedding.call_local_llm_embeddings(sentences)

        index = 0
        data_entries = []
        for embedding in embeddings:
            data_entries.append(
                {"object": "embedding", "index": index, "embedding": embedding.tolist()}
            )
            index = index + 1

        
        data = {
            "object": "list",
            "data": data_entries,
            "usage": {
                "prompt_tokens": num_prompt_tokens,
                "total_tokens": num_prompt_tokens,
            },
        }
        
        return jsonify(data)
    except Exception as e:
        print(e)
        return "Sorry, unable to generate embeddings with model {}".format(model)


if __name__ == "__main__":

    if torch.backends.mps.is_available():
        torch.set_default_device("mps")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--ip", default="0.0.0.0"
    )
    parser.add_argument(
        "-p",
        "--port",
        default=5002,
        type=int,
    )
    args = parser.parse_args()

    host_ip = args.ip
    port = args.port

    app.run(host=host_ip, debug=True, port=port)


