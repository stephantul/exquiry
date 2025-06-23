# exquiry

This is a simple query or document expansion module that implements multiple methods for query or document expansion for sparse information retrieval (IR). This package does very little actual things, but groups a bunch of methods under a common framework, which can be useful for testing or comparisons.

# What is this for?

Sparse document representations suffer from the vocabulary mismatch problem: documents that do not share any terms will have a similarity (or score) of 0, and will not be retrievable. This is especially difficult in the context of information retrieval, because queries and the documents that answer them do not necessarily share any tokens. For example, the query `"heart attack symptoms"` does not match the document `"Myocardial infarction typically presents with chest discomfort, shortness of breath, and sweating."` at all.

Document expansion is a simple set of techniques to counter this shortcoming: at either query or indexing time, we run document expansion models, which augment the document (i.e., either the query or the document) with additional terms. If done at indexing time, document expansion usually adds _terms that are likely to occur in the query_, while if done at query time, it usually adds synonyms and other terms that increase coverage.

For example, `tilde` will augment `"heart attack symptoms"` with:

```
'chest'
'blood'
'include'
'signs'
'may'
'pain'
'weakness'
'heartbeat'
'faint'
'breath'
```

Which leads to a non-zero similarity with the document above, because Tilde already expands the query to the actual symptoms.

# What does it contain?

It currently implements the following expanders:
- [tilde](https://github.com/ielab/TILDE)
- [doctttttquery](https://github.com/castorini/docTTTTTquery) (restyled as t5doc2query)
- [splade](https://github.com/naver/splade)

Note that `doctttttquery` is mostly suited for expanding documents into queries that relate to that documents (i.e., index-time expansion), while `tilde` and `splade` can be used for both.

The default models for all expanders are as follows:

| Expander         | Default Model                                      |
|------------------|----------------------------------------------------|
| TILDE            | [ielab/TILDE](https://huggingface.co/ielab/TILDE)  |
| t5doc2query      | [castorini/doc2query-t5-base-msmarco](https://huggingface.co/castorini/doc2query-t5-base-msmarco) |
| SPLADE           | [naver/splade-cocondenser-ensembledistil](https://huggingface.co/naver/splade-cocondenser-ensembledistil) |

# Installation

The package is not on pypi yet, use the following:

```bash
pip install git+
```

# Example of use

The `DocumentExpander` is the main entrypoint into `exquiry`. A `DocumentExpander` wraps one or more `Expander`s, where an `Expander` is an object that exposes an `expand` function that accepts a document, and which then produces some documents in response.

```python
from exquiry import DocumentExpander

expander = DocumentExpander.from_expansion_types(["tilde", "t5doc2query"])
expansions = expander.expand("Paris is the capital of France. It's where the eiffel tower is")
for name, expansion in expansions.items():
    print(f"Expansions for {name}")
    print(expansion)

```

If you want to instantiate them using one of your own models, you can import the classes directly, and instantiate them using `from_pretrained`.

```python
from exquiry.models import Tilde

# Return 10 expansions
t = Tilde.from_pretrained("ielab/TILDE", k=10)

# Use directly
t.expand("Paris is a great city to live in.")

```

Or you can use it to instantiate a `DocumentExpander`, as follows:

```python
from exquiry import DocumentExpander

expander = DocumentExpander({"tilde": t})
expander.expand("Paris is too hot.")

```
