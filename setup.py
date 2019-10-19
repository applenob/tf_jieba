from setuptools import setup, find_packages, Extension
from datetime import date
import os
import sys
from glob import glob
import tensorflow as tf

TF_INCLUDE, TF_CFLAG = tf.sysconfig.get_compile_flags()
TF_INCLUDE = TF_INCLUDE.split('-I')[1]

TF_LIB_INC, TF_SO_LIB = tf.sysconfig.get_link_flags()
TF_SO_LIB = TF_SO_LIB.replace('-l:libtensorflow_framework.1.dylib',
                              '-ltensorflow_framework.1')
TF_LIB_INC = TF_LIB_INC.split('-L')[1]
TF_SO_LIB = TF_SO_LIB.split('-l')[1]

NAME = "tf_jieba"
GITHUB_USER_NAME = "applenob"
AUTHOR = "Javen Chen"
AUTHOR_EMAIL = "applecer@pku.edu.cn"
MAINTAINER = "Javen Chen"
MAINTAINER_EMAIL = "applecer@pku.edu.cn"
INCLUDE_PACKAGE_DATA = True
REPO_NAME = os.path.basename(os.getcwd())
URL = "https://github.com/{0}/{1}".format(GITHUB_USER_NAME, REPO_NAME)
GITHUB_RELEASE_TAG = str(date.today())
DOWNLOAD_URL = "https://github.com/{0}/{1}/tarball/{2}".format(
    GITHUB_USER_NAME, REPO_NAME, GITHUB_RELEASE_TAG)
PLATFORMS = ["Windows", "MacOS", "Unix"]
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Education",
    "License :: OSI Approved :: MIT License",
    "Natural Language :: English",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: MacOS",
    "Operating System :: Unix",
    "Programming Language :: Python :: 3.3",
    "Programming Language :: Python :: 3.4",
    "Programming Language :: Python :: 3.5",
    "Programming Language :: Python :: 3.6",
]
this_directory = os.path.abspath(os.path.dirname(__file__))


def get_short_description():
  try:
    short_description = __import__(NAME).__short_description__  # GitHub Short Description
  except:
    print("'__short_description__' not found in '%s.__init__.py'!" % NAME)
    short_description = "No short description!"
  return short_description


def get_long_description():
  try:
    with open(os.path.join(this_directory, 'README.md'),
              encoding='utf-8') as f:
      long_description = f.read()
  except:
    long_description = "No long description!"
  return long_description


def get_license():
  try:
    with open(os.path.join(this_directory, 'LICENSE'),
              encoding='utf-8') as f:
      license = f.read()
  except:
    print("license not found in '%s.__init__.py'!" % NAME)
    license = ""
  return license


def get_requires():
  try:
    f = open("requirements.txt")
    requires = [i.strip() for i in f.read().split("\n")]
  except:
    print("'requirements.txt' not found!")
    requires = list()
  return requires


complie_args = [TF_CFLAG, "-fPIC", "-shared", "-O2", "-std=c++11"]
if sys.platform == 'darwin':  # Mac os X before Mavericks (10.9)
  complie_args.append("-stdlib=libc++")
cppjieba_includes = ["third_party/cppjieba/deps",
                     "third_party/cppjieba/include"]
include_dirs = ['tf_jieba', TF_INCLUDE] + cppjieba_includes

module = Extension('tf_jieba.x_ops',
                   sources=glob('tf_jieba/cc/*.cc'),
                   depends=glob('tf_jieba/cc/*.h'),
                   extra_compile_args=complie_args,
                   include_dirs=include_dirs,
                   library_dirs=[TF_LIB_INC],
                   libraries=[TF_SO_LIB],
                   language='c++')
long_description = get_long_description()
license_ = get_license()
print(f"long_description: {long_description}")
print(f"license: {license_}")

setup(
    name=NAME,
    description=get_short_description(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    version="0.1.2",
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    packages=find_packages(),
    include_package_data=INCLUDE_PACKAGE_DATA,
    package_data={"": ["x_ops*.so"]},
    url=URL,
    download_url=DOWNLOAD_URL,
    classifiers=CLASSIFIERS,
    platforms=PLATFORMS,
    license=license_,
    install_requires=get_requires(),
    ext_modules=[module]
)
