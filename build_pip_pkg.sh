#!/usr/bin/env bash

echo "Uninstall tf_jieba if exist ..."
pip uninstall -y tf_jieba

echo "Generate whl file ..."
rm -rf build/ tf_jieba.egg-info/ dist/
python setup.py bdist_wheel sdist
