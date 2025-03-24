"""Setup file for rag package."""

from setuptools import setup, find_packages

setup(
    name="rag",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain-community>=0.0.10",
        "langchain-core>=0.1.0",
        "opensearch-py>=2.4.0",
        "boto3>=1.34.0",
        "boto3-stubs[bedrock]>=1.34.0",
        "watchtower>=3.0.1",
        "python-dotenv>=1.0.0",
    ],
    python_requires=">=3.9",
)
