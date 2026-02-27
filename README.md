# Project NLP - Tripadvisor recommendation system

## Description
This project utilises Natural language processing (NLP) techniques to analyze reviews from users and calculate the similarity between different places. The goal is to propose a recommendation system capable of identifying the most semantically close places depending on their description and user reviews


## Instructions
* To use the code in this project, just follow these steps :
```bash
git clone <url> NLP_project
```
```bash
cd NLP_project
```
* Now we just need to create the environment, so we don't have any problems with the library
```bash
bash setup.sh
```
### The following code will depend on your operating system
* If you are on windows : 
```bash
source venv_NLP/Scripts/activate
```
* If you are on Linux/MacOS :
```bash
source venv_NLP/bin/activate
```
### You can now run the code in the notebooks without problem. You should start with the nlp_model.ipynb notebook to understand the model and then run the evaluation.ipynb notebook.

## Main Features
* **Text vectorization (TF-IDF)**
* **Cosine similarity computation** 
* **Trust factor and ranking of recommendations**
* **2 levels of model evaluation** 


## Project Structure
```text
Project_NLP/
│
├── notebooks/               # Notebooks with all the functional code and features
│   ├── nlp_model.ipynb      # TF-IDF training, cosine similarity and export
│   ├── evaluation.ipynb     # Evaluation of resutls (Level 1 and 2)
│   └── src/                 # Used python scripts (Aurele & Charles)
│       ├──Aurele            # Scripts used in nlp_model.ipynb notebook (developped by Aurele)
|       ├──Charles           # Scripts used in Evaluation.ipynb notebook (developped by Charles)
├── .gitignore               # Ignored files 
├── README.md                # Project documentation
├── requirements.txt         # Required libraries to be downloaded (see instructions)
└── setup.sh                 # Environment intialisation script 