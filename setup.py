from setuptools import find_packages, setup

requirements = [
    "fastapi==0.115.12",
    "uvicorn==0.35.0",
    "langgraph==0.5.1",
    "chromadb==1.0.15",
    "pypdf==5.7.0",
    "textsplitter==1.0.3",
    "requests==2.32.4",
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
