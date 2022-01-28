rm -r dist
python setup.py bdist_wheel
twine upload dist/*
