# cicero
A famous Roman orator, symbolizing communication, structure, and conveying knowledge efficiently.

Many open source projects are combined or blended together to formalize the functionality of `SUBER`.  , these include:

- [Fabric](fabric/README.md)
- [Ollama](ollama/README.md)
- [NRMS](nrms/README.md)
- [Qdrant](qdrant/README.md)
- [SUBER](SUBER/README.md)

## Fabric 

The intent of Fabric in `Cicero` is to provide a clean, fast, and efficient API with access to large langague models. Fabric is written in GO, a compiled language -- so its faster than Python.  It is already equipped with several, what is called `patterns,` which are prompts written in markdown.  The `patterns` are  highly refined.

## Ollama

`Ollama` is intended to serve up private models for Fabric.  It is integrated into many systems that use LLMs, it can provide what might be termed `private` access to a model which contrasts with possibly expensive access to LLM online API such as those provided by Open AI and others.

## NRMS Model.

The NRMS Model is working now, the trick was to install recommenders followed by tensorflow like this `pip install tensorflow[and-cuda]=2.15.1`. There are breaking changes beyond that version.

## Newsfeeds

This component of the project will process the news feed which at the moment will use Qdrant.


