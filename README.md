# OWUI (Open Web UI)

This is a Docker Compose project comporised of [https://ollama.com/](Ollama), [Open WebUI](https://github.com/open-webui/open-webui), [Fabric](https://github.com/danielmiessler/fabric), content from the [Recommenders team](https://github.com/recommenders-team/recommenders/tree/main), and [SUBER](https://github.com/SUBER-Team/SUBER).  

The theme is advancements in recommenders which is covered very well by the recommenders team.  SUBER combines large language models (LLMs) and reinforcement learning to provide recommenders.  

This specific work advanced SUBER with optimal LLM tooling, containerization, and prompt engineering using Open Source products mentioned.

## Getting Started

Clone the [cicero](https://github.com/acsheller/cicero) Repo. Review the `compose.yml` file as there is a section for `Ollama`, `Open WebUI`, `Fabric`, `suber`, `nrms`, and `pyai`.

Make a folder to hold the models called `./ollama_models`. Models will be pulled from Ollama and stored here for use by the system. Do not check this into git as the models can be very large.

Make another folder called `./datasets` and follow the [datasets procedures](./docs/Datasets.md) in the documentation. Note that datasets can be very large for the MIND dataset.  the NRMS Jupyter notebook - isolated in its onwn container will use this folder also.  Review the nrms notebook. 

For a time-saver create some aliases like this, place them in your `.bashrc` and source it.

```.bash

alias dc='docker-compose'
alias dcup='docker-compose up --build -d'

```

After sourcing your `.bashrc` or opening a new terminal type dcup in the cicero folder. 

```.bash

asheller: cicero$ dcup
Creating network "cicero_cicero" with driver "bridge"
Creating network "cicero_default" with the default driver
Building fabric
....
....


Creating nrms-container   ... done
Creating suber-container  ... done
Creating fabric-container ... done
Creating open-webui       ... done
Creating ollama           ... done
Creating pyai-container   ... done
asheller: cicero$

```


## Accessing the Services

- [Ollama - http://localhost:11434/](http://localhost:11434/)
- [Open WebUI - http://localhost:8080/](http://localhost:8080/)
- [NRMS- http://localhost:8888/lab/tree/nrms.ipynb](http://localhost:8888/lab/tree/nrms.ipynb)
- [SUBERX - http://localhost:8889/lab/tree/SUBERX/jupyter](http://localhost:8889/lab/tree/SUBERX/jupyter)
- [Pydantic AI - http://localhost:8890/lab/tree/jupyter](http://localhost:8890/lab/tree/jupyter)


## What is all this stuff?

I kept adding containers to help work on the problem. I'll discuss them in order of current usefullness to the project.

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
