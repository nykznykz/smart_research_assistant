from setuptools import setup, find_packages

setup(
    name="smart_research_assistant",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain>=0.1.0",
        "langchain-community>=0.0.10",
        "langchain-core>=0.1.0",
        "faiss-cpu>=1.7.4",
        "playwright>=1.40.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "test": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ],
    },
    python_requires=">=3.9",
) 