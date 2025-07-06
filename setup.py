from setuptools import find_packages, setup

requirements = [
    "fastapi==0.115.12",
    "pydantic==2.11.3",
    "pydantic-settings==2.9.1",
    "uvicorn==0.35.0",
    "langgraph==0.5.1",
    "chromadb==1.0.15",
    "pypdf==5.7.0",
    "textsplitter==1.0.3",
    "requests==2.32.4",
    "langgraph==0.5.1",
    "pytest==8.4.1",
    "python-multipart==0.0.20",
    "scikit-learn==1.7.0"
]

setup(
    name="AI Agent",
    version="1.0",
    description="AI agent for automating tasks and providing assistance",
    author="Vivek Ratna Kansakar",
    author_email="kansakar.vivek@gmail.com",
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
    },
)
