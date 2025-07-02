<h2 align="center">
  <img width="40%" alt="SKULLLLL" src="assets/skull.png"><br/>
  EXQUIRY
</h2>

This is a simple python module that implements multiple methods for query or document expansion. It is intended to be used with sparse information retrieval (IR). This package does very little actual things, but groups a bunch of methods under a common framework, which can be useful for testing or comparisons.

# Exquiry in 3 lines

```python
from exquiry import get_expander

expander = get_expander("tilde")  # or 't5doc2query'
expansions = expander.expand(["Paris is the capital of France. It's where the eiffel tower is"])

```

Imagine you have a document, and you want to expand it to also include terms that are related to it, so that it is more easily found in an index. This is what query expansion does: it adds related terms to a document, increasing document coverage.

# Explanation

Sparse document models, such as BM25, suffer from the vocabulary mismatch problem: documents that do not share any terms will have a similarity (or score) of 0, and will not be retrieved. This is a big issue, because queries often do not share a lot of terms with the documents that are relevant for that query. For example:

```
"heart attack symptoms"
```

does not share any terms with a document describing the symptoms:

```
"Myocardial infarction typically presents with chest discomfort, shortness of breath, and sweating."
```

_Expansion_ is a simple set of techniques to counter this shortcoming: at either query or indexing time, we run document expansion models, which augment the document (i.e., either the query or the document) with additional terms. If done at indexing time, document expansion usually adds _terms that are likely to occur in the query_, while if done at query time, it usually adds synonyms and other terms that increase coverage.

For example, `tilde` (see below) will augment `"heart attack symptoms"` with:

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

Which leads to a non-zero similarity with the document above, because Tilde is trained to expands the query to the actual symptoms.

# What does `exquiry` contain?

It currently implements the following expanders:
- [tilde](https://github.com/ielab/TILDE)
- [doctttttquery](https://github.com/castorini/docTTTTTquery) (restyled as t5doc2query)

Note that `doctttttquery` is mostly suited for expanding documents into queries that relate to that documents (i.e., index-time expansion), while `tilde` can be used for both.

The default models for these expanders are as follows:

| Expander         | Default Model                                      |
|------------------|----------------------------------------------------|
| TILDE            | [ielab/TILDE](https://huggingface.co/ielab/TILDE)  |
| t5doc2query      | [castorini/doc2query-t5-base-msmarco](https://huggingface.co/castorini/doc2query-t5-base-msmarco) |

# Installation

The package is not on pypi yet, use the following:

```bash
pip install git+https://github.com/stephantul/exquiry.git
```

# Example of use

`get_expander` is the main entrypoint into `exquiry`. `get_expander` is a function that returns one of the expanders implemented in the package, using the default model and arguments implemented in the package. An expander is simply something that takes in a document and produces expansions.

```python
from exquiry import get_expander

expander = get_expander("tilde")  # or 't5doc2query'
expansions = expander.expand(["Paris is the capital of France. It's where the eiffel tower is"])
print(expansions)

```

## Expander

An expander is for all intents and purposes a simple interface that exposes three functions:

### `expand`

Accepts `documents`, a string or list of strings, and `k`, an integer, and produces either a list of list of `k` strings. Note that this function always produces a list, because using sum types as returns is bad for your health.

```python
from exquiry import get_expander

expander = get_expander("tilde")
expansions = expander.expand(["heart attack symptoms"], k=5)
print(expansions)
# list of list of strings
# [['chest', 'blood', 'include', 'signs', 'may']]

```

### `expand_as_tokens`

Some sparse embedder models, including [coil](https://arxiv.org/abs/2104.07186) and [deep impact](dl.acm.org/doi/pdf/10.1145/3404835.3463030) use expansions during inference or indexing. Instead of concatenating these directly with the original text, these are instead added as a text input pair, i.e., separated by a `[SEP]` token from the original input text. To support this use-case, `exquiry` offers a simple helper to create these text pair encodings.

The usage is the same as `expand`, except that it also takes a tokenizer as argument. This tokenizer should be _the tokenizer used in your model_, not the tokenizer used by your expander. Here's an example using [`bge-m3`](https://huggingface.co/BAAI/bge-m3).

```python
from transformers import AutoTokenizer

from exquiry import get_expander

tokenizer = AutoTokenizer.from_pretrained("baai/bge-m3")
expander = get_expander("tilde")
expansions = expander.expand_as_tokens(tokenizer, "Paris is the capital of France. It's where the eiffel tower is", k=5)

# Expansions is a BatchEncoding ready to process by your model.

print(tokenizer.decode(expansions.input_ids[0]))
# "<s> Paris is the capital of France. It's where the eiffel tower is</s></s> city located french de world</s>"
```

### `expand_as_strings`

This functions the same as the `expand` function, except it returns the original document concatenated with the expansions, separated by a single whitespace. This is useful if your method does not consider expansions separately from the original document, e.g., in bag of words methods.

```python
from transformers import AutoTokenizer

from exquiry import get_expander

tokenizer = AutoTokenizer.from_pretrained("baai/bge-m3")
expander = get_expander("tilde")
expansions = expander.expand_as_strings("Paris is the capital of France. It's where the eiffel tower is", k=5)
print(expansions)
# ["Paris is the capital of France. It's where the eiffel tower is city french world located de"]
```


## Manual instantiation

If you want to instantiate an expander using one of your own models, you can use `get_expander_class`, and instantiate them using `from_pretrained`.

```python
from exquiry import get_expander_class

TildeClass = get_expander_class()
t = TildeClass.from_pretrained("ielab/TILDE")

# Use directly, get 10 expansions
t.expand("Paris is a great city to live in.", k=10)

```

## What about the name?

The name is a play on words: "inquiry" is semantically related to "query", and this package introduces words from outside of the query into the query. So "exquiry".

## What about the skull?

idk

# License

MIT

# Author

Stéphan Tulkens
