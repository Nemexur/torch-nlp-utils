# Torch Data Utils

Inspired by AllenNLP library I have decided to built my own for pure PyTorch.
This package is its essence is a collection of useful utils for development and data processing in PyTorch.

* **Data** - for processing datasets for DL Model in PyTorch.
* **Common** - just a bunch of useful classes and functions.

Example:

1. Define your dataset reader.

```python
class MyDatasetReader(DatasetReader):
    def _read(self, file_path: str) -> Union[Iterable[Any], Dataset]:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                yield self._text_to_instance(line.split())

    def _text_to_instance(self, tokens: List[str]):
        return {'tokens': tokens, 'labels': 1}
```

2.Instantiate you train dataset.

```python
reader = MyDatasetReader(lazy=True)
train_dataset = reader.read('./sample.csv')
```

3.Construct Vocabulary for label encoding.

```python
vocab = Vocabulary(
    datasets={
        'train': MyDatasetReader(lazy=True).read('./sample.csv')
    },
    namespaces={
        'tokens': Namespace(processing_type='padding_oov'),
        'labels': Namespace(processing_type='pass_through')
    }
)
```

4.Encode train dataset with Vocabulary.

```python
train_dataset.encode_with(vocab)
```

5.Instantiate DataIterator.

```python
def collate_fn(sample):
    print(sample, end='\n\n')
    return sample

iterator = DataIterator(
    train_dataset,
    collate_fn=collate_fn,
    batch_size=2
)
```

6.Check results :).

```python
print(next(iter(iterator)))
```

```python
[{'tokens': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 8, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 15, 11, 23, 24, 11, 25, 26, 27, 28], 'labels': 1}, {'tokens': [29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41], 'labels': 1}]

[{'tokens': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 8, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 15, 11, 23, 24, 11, 25, 26, 27, 28], 'labels': 1}, {'tokens': [29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41], 'labels': 1}]
```
