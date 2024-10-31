# Ollama

In this project `Ollama` will serve up local models for use by `Fabric`.


Ollama is used to server local LLMs for Fabric. While something like GPT4o could be used, when learning it is more afordable to use local models.  One can always switch the default model.


I cloned this repository [https://github.com/valiantlynx/ollama-docker/tree/main](ollama-docker) in case I wanted to change anything.  I then did a:

```.bash
    docker-compose -f docker-compose-ollama-gpu.yaml up -d

```

and it provided a super interface located at http://localhost:8000


# Even Better

I came across this [OpenWebUI](https://github.com/open-webui/open-webui) which is even better.  

```.bash
docker run -d -p 3000:8080 --gpus=all -v ollama:/root/.ollama -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:ollama

```
