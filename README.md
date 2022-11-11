#### Installing requirements
```bash
python -c 'import nltk;nltk.download("punkt")'
```

#### Testing

```bash
coverage run --source=dre --module pytest --verbose ./test/test_*.py && coverage report --show-missing
```