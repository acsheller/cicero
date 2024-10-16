# Fabric

## Overview
Fabric is at this GitHub repository, Daniel Miessler Fabric. The key concept is to make large language models useful.  It achieves this by capturing prompts that users use routinely and turning them into "commands " for use at the command line. The command line or CLI is a crucial concept of Fabric.  

Fabric, originally written in Python, has undergone a significant transformation from Python to the Go or Golang programming language. While I may not be an expert on Go, this transition is a noteworthy development in the Fabric journey. 

### In this Project
In this project, a pattern will be created to craft synthetic users. This one example is good, but we can do more.  Fabric, if containerized, and a rest-api provided could also do the portion of the project that provides the "ground truth" to the model in `SUBER -- Simulated User Behavior Environment for Recommender Systems`.


## Installing Fabric
Now that Go is installed, installing Fabric, a Git-based project, is easy.  Follow the [installation instructions](https://github.com/danielmiessler/fabric#Installation) that are provided.   
    
```    
    go install github.com/danielmiessler/fabric@latest
```

Note that upgrade instructions are also provided.  The project is updated frequently so stay current. 

At this point opne can do a `fabric -h` to get help. There are alot of options so be sure to 

```.bash
    fabric -h
    Usage:
    fabric [OPTIONS]

    Application Options:
    -p, --pattern=             Choose a pattern from the available patterns
    -v, --variable=            Values for pattern variables, e.g. -v=#role:expert -v=#points:30
    -C, --context=             Choose a context from the available contexts
        --session=             Choose a session from the available sessions
    -S, --setup                Run setup for all reconfigurable parts of fabric
        --setup-skip-patterns  Run Setup for all reconfigurable parts of fabric except patterns update
        --setup-vendor=        Run Setup for specific vendor, one of Ollama, OpenAI, Anthropic, Azure, Gemini, Groq,
                                Mistral, OpenRouter, SiliconCloud. E.g. fabric --setup-vendor=OpenAI
    -t, --temperature=         Set temperature (default: 0.7)
    -T, --topp=                Set top P (default: 0.9)
    -s, --stream               Stream
    -P, --presencepenalty=     Set presence penalty (default: 0.0)
    -r, --raw                  Use the defaults of the model without sending chat options (like temperature etc.) and use
                                the user role instead of the system role for patterns.
    -F, --frequencypenalty=    Set frequency penalty (default: 0.0)
    -l, --listpatterns         List all patterns
    -L, --listmodels           List all available models
    -x, --listcontexts         List all contexts
    -X, --listsessions         List all sessions
    -U, --updatepatterns       Update patterns
    -c, --copy                 Copy to clipboard
    -m, --model=               Choose model
    -o, --output=              Output to file
        --output-session       Output the entire session (also a temporary one) to the output file
    -n, --latest=              Number of latest patterns to list (default: 0)
    -d, --changeDefaultModel   Change default model
    -y, --youtube=             YouTube video "URL" to grab transcript, comments from it and send to chat
        --transcript           Grab transcript from YouTube video and send to chat (it used per default).
        --comments             Grab comments from YouTube video and send to chat
    -g, --language=            Specify the Language Code for the chat, e.g. -g=en -g=zh
    -u, --scrape_url=          Scrape website URL to markdown using Jina AI
    -q, --scrape_question=     Search question using Jina AI
    -e, --seed=                Seed to be used for LMM generation
    -w, --wipecontext=         Wipe context
    -W, --wipesession=         Wipe session
        --printcontext=        Print context
        --printsession=        Print session
        --readability          Convert HTML input into a clean, readable view
        --dry-run              Show what would be sent to the model without actually sending it
        --serve                Serve the Fabric Rest API
        --address=             The address to bind the REST API (default: :8080)
        --version              Print current version

    Help Options:
    -h, --help                 Show this help message

```