#### Installing requirements
```bash
python -c 'import nltk;nltk.download("punkt")'
```

#### Testing

```bash
#coverage run -m pytest ./test/test_nfasegment.py ./test/test_observer.py  && coverage report -m
coverage run --source=dre --module pytest --verbose ./test/test_*.py && coverage report --show-missing
```