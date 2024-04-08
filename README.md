# Artificial-Intelligence-Guided-Approaches-for-Next-Generation-Immunotherapies
This work, taking advantage of Geometric Deep Learning technologies, proposes a predictive approach based on an AI system capable of classifying chemical compounds according to their activity towards the enzyme **Indoleamine-2,3-dioxygenase 1** (IDO1). Given the key role of this enzyme in the onset of immune resistance by various cancers and given the growing interest in immunotherapy in the treatment of these diseases, this work is shared to help identify potential active compounds. 

Using **Graph Neural Networks** (GNNs), trained on bioactivity data regarding compounds tested as IDO1 inhibitors, obtained from the ChEMBL database, a software was built to generate molecular graphs from the SMILES string, and then classify these within four activity classes: 
*	**Class 0**, inactive compounds. 
*	**class 1**, low active compounds. 
*	**class 2**, moderately active compounds. 
*	**class 3**, very active compounds.
