# OWUI (Open Web UI)

This container composition  is comporised of [https://ollama.com/](Ollama), [Open WebUI](https://github.com/open-webui/open-webui), [Fabric](https://github.com/danielmiessler/fabric), content from the [Recommenders team](https://github.com/recommenders-team/recommenders/tree/main), and [SUBER](https://github.com/SUBER-Team/SUBER).  

The theme is advancements in recommenders which is covered very well by the recommenders team.  SUBER combines large language models (LLMs) and reinforcement learning to provide recommenders.  

This specific work advanced SUBER with optimal LLM tooling, containerization, and prompt engineering using Open Source products mentioned.

## Getting Started

Make a directory to hold the models called `~/ollama_models`. Models will be pulled from Ollama and stored here for use by the system.

Clone the [cicero]() Repo.  Currently, most work is done in [cicero/owui](../owui/) but clone the whole thing for now. Both `Fabric` and `Open WebUI` have there own docker files that can be built when executing `docker-compose up`. `Ollama` uses the `ollama container` so arguments are just added.  Please review the docker  and the docker-compose - `compose.yml`.

Please pay close attention to the environment variables set in the [docker file for Open WebUI](./Dockerfile.owui).  Cross reference with what is currently set with what you want and what is listed on the [env-configuraiton page for Open WebUI](https://docs.openwebui.com/getting-started/env-configuration/).


```.bash

    git clone https://github.com/acsheller/cicero

    # Build and bring up all containers.

    docker-compuse up --build -d



```

After its up go to http://localhost:8080 for the Open WebUI interface.  Ollama is available on http://localhost:11434.
Both these are preconfigured with settings that are in the `openwebui-data` folder and the `fabric-config` folder.
Remember that models are stored in `~/ollama-models`  -- this can change.  

When the system comes up, if all of these are empty configure them as needed.  


## Stopping things

```
docker-compose down

```

## Exec-ing into the fabric container

While developing the fabric REST API it was necessary to `docker exec` into it. 

```.bash

asheller: owui$ docker exec -it fabric-container /bin/bash
fab_user@b0afe54a6854:~$ 

```

## Fabric Container 

Currently launching it by hand as development of the rest-api is happening. One can start the rest API by:

```.bash

fab_user@b0afe54a6854:~$ python3 fabric_rest_svc.py 
 * Serving Flask app 'fabric_rest_svc'
 * Debug mode: off
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:8000
 * Running on http://172.18.0.4:8000
Press CTRL+C to quit
172.18.0.1 - - [07/Nov/2024 22:08:31] "GET / HTTP/1.1" 200 -
172.18.0.1 - - [07/Nov/2024 22:08:32] "GET /favicon.ico HTTP/1.1" 404 -
172.18.0.1 - - [07/Nov/2024 22:08:37] "GET /html HTTP/1.1" 200 -

```

`Patterns` are engineered prompts intended to do a certain action that a human might be interested in such as summarization, or extract_wisdom.  I'm a novice at this but it is very appealing because they are written in markdown.  The REST interface is custom for now and is specific to the needs of:

1. Generating synthetic users that can be used in SUBER.
2. Generating the rating or a recommend/not recommend value given a description of a user and a news article. 

If one is inside the fabric container they can play around with it like this:

```.bash

"Generate 100 Synthetic Users with an American background" |fabric -p generate_synthetic_news_user

```

