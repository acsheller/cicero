# CICERO -- 

This is a Docker Compose project comporised of [https://ollama.com/](Ollama), [Open WebUI](https://github.com/open-webui/open-webui), , content from the [Recommenders team](https://github.com/recommenders-team/recommenders/tree/main), [Pydantic AI](https://ai.pydantic.dev/), and [SUBER](https://github.com/SUBER-Team/SUBER).  

The theme is advancements in recommenders which is covered very well by the recommenders team.  SUBER combines large language models (LLMs) and reinforcement learning to train recommendations in sparse environments where labeled data is limited.  This work expands on that with some modifications.

## Getting Started

Clone the [cicero](https://github.com/acsheller/cicero) Repo. Review the `compose.yml` file as there is a section for `Ollama`, `Open WebUI`, `Fabric`, `suber`, `nrms`, and `pyai`.

Setup 

Get the MIND datasets and set it up as described in the [datasets documentation for Cicero](/datasets/README.md).


For a time-saver create some aliases like this, place them in your `.bashrc` and source it. The UID and GID are set because the containers mount in a local directory.  the permissions on the files need to be set appropriately. Do a `sudo apt install id-utils` to install the `id` command.

```.bash

export UID=$(id -u)
export GID=$(id -g) 

alias dc='docker-compose'
alias dcup='docker-compose up --build -d'

```

After sourcing your `.bashrc` or opening a new terminal type dcup in the cicero folder. 

```.bash

asheller: cicero$ dcup
Creating network "cicero_cicero" with driver "bridge"
Building suber
[+] Building 0.5s (15/15) FINISHED                                                                       docker:default
 => [internal] load build definition from Dockerfile.suber                                                         0.0s
 => => transferring dockerfile: 2.17kB                                                                             0.0s
....


 => => writing image sha256:3720d94741cfb90951cf655213d755cba5041c69af7a76eb106698db42f4dc80                       0.0s
 => => naming to docker.io/library/cicero_gnuradio                                                                 0.0s
Creating pyai-container  ... done
Creating ollama          ... done
Creating open-webui      ... done
Creating suber-container ... done
asheller: cicero$

```

## Now I'm up, Now What? 

An LLM will need to be loaded

These services are available to you now:

- [Ollama - http://localhost:11434/](http://localhost:11434/)
- [Open WebUI - http://localhost:8080/](http://localhost:8080/)
- [SUBERX - http://localhost:8889/lab/tree/SUBERX/jupyter](http://localhost:8889/lab/tree/SUBERX/jupyter)
- [Pydantic AI - http://localhost:8890/lab/tree/jupyter](http://localhost:8890/lab/tree/jupyter)

Except there are no LLMS loaded.  Review the models at [Ollama Models](https://ollama.com/search).  Note the ones marked `tools`.  Also note the specifications in terms of parameters and consider how much GPU memory is available as a resource.  For example, with one 12GB GPU, a 3b parameter model, such as [`llama3.2`](https://ollama.com/library/llama3.2) will fit into 12GB of GPU memory. Also [`mistral:7b`], with 7 billion parameters will fit into the GPU memory as well.

Go to your [Open WebUI service](http://localhost:8080/), the container is configured to automatically log the user in and is almost ready to go except for a model being loaded. 

- Select the User Icon in upper-right corner of page.
- Select Admin Panel [link for you to localhost version](http://localhost:8080/admin/users.)
- Select Settings [link for you to settings localhost](http://localhost:8080/admin/settings) near the top, towards the center.
- Select `Connections` from the available options.
- You should see the Ollama API Connections for the [localhost Ollama](http://localhost:11434/), select the wrenh to the far right.
- In the `Pull a model from Ollama.com` type in the model you would like to pull down.  `llama3.2`, and/or `minstral:7b` are good options for the hobbiest GPU.
- Click on `models  to see the models you've downloaded. 

### Ollama

`Ollama` is a very good way of serving up models that services like `Open WebUI` can take advantage of. Fortunately, there is an "official" ollama container that makes it surprisingly easy to work with.  Review the `Ollama` section in the [compose.yml](./compose.yml) file and be sure you understand it.  There's not much to configure.

### Open WebUI

`Open WebUI` was a little harder to setup. But there is an official container for it.  I added an `openwebui-data` folder to the project so that it could save its configuration locally.  That way on each shutdown-startup it doesn't come back blank.  Some configuration is necessary so be sure to review the [Open WebUI documentation](https://docs.openwebui.com/) and the settings in the [compose.yml](./compose.yml) file and gain some understanding.  Note the environment variables section -- e.g: `OLLAMA_BASE_URL=http://ollama:11434` -- and how this makes it very simple to configure Open WebUI to use the `Ollama` that was started.

### SUBERX

`SUBERX` is the primary focus of the overall project.  There are a few notebooks to work with. What is special about this, is that PyTorch and Tensorflow are both installed and work in the one container.  Stable Baselines 3 and Recommenders is also installed.  This was precarious and fortunately it is in container format for repatability.

### NRMS

`NRMS`, this was the original notebook from the 2021 MIND news recommenders contest.  I needed to update it because it no longer could pull the datasets, other than the demo dataset.  It also abstracted out the a few key steps for using a real dataset that still needs to be addressed.

### PYAI (Pydantic AI)

I added this one recently bacause it is a very efficient way to use LLMs in contrast to how it was done in SUBER, in my opinion.  I'm still working with it. 


## Stopping things

```
docker-compose down or dc down if you set dc to be an alias for docker-compose

```
