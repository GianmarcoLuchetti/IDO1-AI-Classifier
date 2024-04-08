# IDO1-AI-Classifier
This work, taking advantage of Geometric Deep Learning technologies, proposes a predictive approach based on an AI system capable of classifying chemical compounds according to their activity towards the enzyme **Indoleamine-2,3-dioxygenase 1** (IDO1). Given the key role of this enzyme in the onset of immune resistance by various cancers and given the growing interest in immunotherapy in the treatment of these diseases, this work is shared to help identify potential active compounds. 

Using **Graph Neural Networks** (GNNs), trained on bioactivity data regarding compounds tested as IDO1 inhibitors, obtained from the [ChEMBL database](https://www.ebi.ac.uk/chembl/), a software was built to generate molecular graphs from the SMILES string, and then classify these within four activity classes: 
*	**Class 0**, inactive compounds. 
*	**class 1**, low active compounds. 
*	**class 2**, moderately active compounds. 
*	**class 3**, very active compounds.
  
![Criteria for assigning activity classes.](https://github.com/GianmarcoLuchetti/IDO1-AI-Classifier/blob/main/img/label.png)

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
To configure the execution environment, it is necessary to download the Python package manager Anaconda, which can be obtained free on Windows/MacOS/Linux OS at the [following link](https://www.anaconda.com/download)). Once the manager has been installed, via the Anaconda prompt on Windows, or via terminal on MacOS/Linux, a Python environment must be created and then activated using the following command:
```
conda create -n env_name python=3.10
conda activate env_name
```

The required libraries can now be installed using the pip command by referring to the `requirements.txt` file:
```
pip install -r requirements.txt
```

For further information, please refer to the official Anaconda documentation available at the [following link](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#activating-an-environment).


## Software Execution
After configuring and activating the environment, the software can be run via the Anaconda prompt on Windows or via terminal on MacOS/Linux. After moving to the directory containing the executable files (e.g. src on the desktop), to launch the software run the command `python main.py`.

![Code to launch the software.](https://github.com/GianmarcoLuchetti/IDO1-AI-Classifier/blob/main/img/run.png)

The software will run and ask the user to enter the path to the dataset containing the samples to be analysed. This dataset must have a .txt extension and must necessarily have a column named 'Smiles'/'SMILES'/'smiles', which contains the SMILES strings of the samples. In addition, if columns named "zinc_id", "Molecule ChEMBL ID" and "Molecule Name" are present, this information will be retained in the final output, but is not required for execution purposes.

![Example of .txt dataset for analysis.](https://github.com/GianmarcoLuchetti/IDO1-AI-Classifier/blob/main/img/data.png)

After execution, the software will ask the user whether to save the results obtained, which will always include the SMILES string of the sample, the predicted activity class, and the prediction probability. In the first case, the user will be asked to specify the path where the results are to be saved and the name to be assigned to the file; in the second case, the results will be displayed in the terminal. The first option is recommended for many samples.

![Example of results obtained from the previous dataset.](https://github.com/GianmarcoLuchetti/IDO1-AI-Classifier/blob/main/img/res.gif)

## References
The following project was developed at the Pharmaceutical Sciences and Engineering Departments of the University of Perugia. The student Gianmarco Luchetti Sfondalmondo followed this development as a master's thesis project, under the supervision of Professor Antonio Macchiarulo and Professor Gianluca Reali.
