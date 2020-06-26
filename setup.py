from setuptools import setup, find_packages

setup(name='citation_worthiness',
      version='0.0.1',
      description='Code for citation worthiness paper',
      long_description=open("README.md").read(),
      long_description_content_type="text/markdown",
      classifiers=[
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering :: Artificial Intelligence :: Citation Worthiness',
      ],
      keywords='Citation Worthiness',
      url='https://github.com/',
      author='Tong Zeng and Daniel Acuna',
      author_email='tozeng@syr.edu',
      license='MIT',
      packages=find_packages(exclude=["*.tests", "*.tests.*",
                                      "tests.*", "tests"]),
      install_requires=[
          "allennlp==0.9.0",
          "scikit-learn==0.21.2"
      ],
      tests_require=[
          'pytest',
      ],
      include_package_data=True,
      python_requires='>=3.6.1',
      zip_safe=False)
