## Code Examples for GenAI Design Patterns Book

### Note on hardware requirements
* Almost all of the examples involve Jupyter notebooks with a Python 3 kernel. Readers have successfully run the notebooks on
  - Colab
  - Local Jupyter
  - Vertex AI Workbench
  - (file a PR if you have been able to run on Databricks, Sagemaker, etc.)
* In the case of notebooks that involve local models (e.g. LogitsMasking, AdapterTuning), you will need the equivalentof a L4 GPU and about 32 GB of RAM.

### Directions to run examples
1. In your favorite Jupyter notebook environment, clone this repository:  ```git clone https://github.com/lakshmanok/generative-ai-design-patterns/```
2. Edit examples/keys.env to add your keys for the major foundational models and HuggingFace. You don't need all of them (each notebook will check whether you have the relevant model)
3. Run the notebook(s) corresponding to the pattern.
   - Uncomment the pip install line if necessary
   - Sign up to license agreements on HuggingFace if necessary
