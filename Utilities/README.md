# Utilities

## train_test_split.py
### Description
Used for line-based generation of two datasets `train` and `test` out of a single source dataset, containing all data references per line.
Outputs two files `test_SOURCE` and `train_SOURCE` where `SOURCE` is the name of the source dataset.

> 1st parameter: Expects the path to the source dataset
> 
> 2nd parameter, optional: Ratio for training data, default is 0.6. Therefore the ratio for the test data is `1 - trainRatio`

### Example:
```python
python train_test_split.py dataset.txt
python train_test_split.py dataset.txt 0.6 # same result as above
```

This outputs two files `train_dataset.txt` and `test_dataset.txt`.