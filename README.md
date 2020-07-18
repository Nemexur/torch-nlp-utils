# Torch NLP Utils

Inspired by [AllenNLP](https://github.com/allenai/allennlp) data processing I have decided to built a slightly different implementation for pure PyTorch models.
My main goal was an easy integration with the existing pipeline and simplification of dataset reading and label encoding for text.

DatasetReader is almost the same as AllenNLP version (mine has support for loading `max_instances_in_memory` each time, it is useful for additional sorting in DataIterator), however, DataIterator and Vocabulary are completely different.

* **Data** - processing data for DL Model in PyTorch.
* **Common** - just a bunch of useful classes and functions for development.

Example of creating dataset reader, encoding it and iterating in lazy manner:

1. Define your dataset reader.

```python
class MyDatasetReader(DatasetReader):
    def _read(self, file_path: str) -> Union[Iterable[Any], Dataset]:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                tokens, target = line.split('\t')
                yield {'tokens': tokens.split(), 'labels': int(target)}
```

2. Instantiate you train dataset.

```python
reader = MyDatasetReader(lazy=True)
train_dataset = reader.read('./sample.csv')
```

3. Construct Vocabulary for label encoding.

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

4. Encode train dataset with Vocabulary.

```python
train_dataset.encode_with(vocab)
```

5. Instantiate DataIterator.

```python
iterator = DataIterator(
    train_dataset,
    collate_fn=lambda x: x,
    batch_size=2
)
```

6. Check results for one batch. DataIterator returns `Batch` instance with the same attributes as namespaces.

```python
next(iter(iterator))
>>> Batch(tokens=[[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 8, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 15, 11, 23, 24, 11, 25, 26, 27], [28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]], labels=[0, 1])
```
