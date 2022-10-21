#### Testing

```bash
coverage run --source=dre --module pytest --verbose ./test/test_*.py && coverage report --show-missing
```