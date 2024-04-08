# Artificial-Intelligence-Guided-Approaches-for-Next-Generation-Immunotherapies
This work, taking advantage of Geometric Deep Learning technologies, proposes a predictive approach based on an AI system capable of classifying chemical compounds according to their activity towards the enzyme **Indoleamine-2,3-dioxygenase 1** (IDO1). Given the key role of this enzyme in the onset of immune resistance by various cancers and given the growing interest in immunotherapy in the treatment of these diseases, this work is shared to help identify potential active compounds. 

Using **Graph Neural Networks** (GNNs), trained on bioactivity data regarding compounds tested as IDO1 inhibitors, obtained from the [ChEMBL database](https://www.ebi.ac.uk/chembl/), a software was built to generate molecular graphs from the SMILES string, and then classify these within four activity classes: 
*	**Class 0**, inactive compounds. 
*	**class 1**, low active compounds. 
*	**class 2**, moderately active compounds. 
*	**class 3**, very active compounds.


## Repository Content
* **src folder**, contains the codes and elements required to run the software, including:
  -	utils.py, holds the scripts necessary for the chemical properties computation and the generation of molecular graphs.
  -	 NNET.py, holds the structure of the Neural Networks used as classifiers.
  -	 model.py, holds the structure of the final classifier based on an assembly of the individual Nets to form a decision tree.
  -	 nn_params, contains the trained parameters for the Neural Networks.
  -	 main.py, holds the main software structure that is executed.

* **requirements.txt**, is a configuration file that must first be installed via Anaconda to create the python environment in which to run the code.

The repository can be downloaded freely from GitHub either manually or by using the terminal command `git clone <repository link>` in the desired PC directory. Example, installation of the repository in the desktop directory:


## Configuring the Environment
To configure the execution environment, it is necessary to download the Python package manager Anaconda, which can be obtained free on Windows/MacOS/Linux OS at the [following link]([INSTALL ANACONDA LINK](https://www.anaconda.com/download)). Once the manager has been installed, via the Anaconda prompt on Windows, or via terminal on MacOS/Linux, a Python environment must be created and then activated using the following command:
```
conda create -n env_name python=3.10
conda activate env_name
```

The required libraries can now be installed using the pip command by referring to the `requirements.txt` file:
```
pip install -r requirements.txt
```

For further information, please refer to the official Anaconda documentation available at the [following link](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#activating-an-environment).
