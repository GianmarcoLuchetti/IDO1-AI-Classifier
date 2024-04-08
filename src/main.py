import utils
import model
import pandas as pd


def main(dataset_path):
    graph_dataset, smiles_list = utils.graph_dataset(dataset_path)  # generation of graph dataset
    predictions = model.model(graph_dataset)  # computation of predictions

    results = pd.DataFrame({'Smiles': smiles_list, 'Predicted activity class': predictions[0],
                            'Prediction probability': predictions[1]})  # storing the results

    df = pd.read_csv(path, sep=None, engine='python', header=0)
    if 'zinc_id' in df.columns:  # save ID if found in the original dataset
        z_id = df['zinc_id'].tolist()
        results['Zinc ID'] = z_id
    if 'Molecule ChEMBL ID' in df.columns:
        c_id = df['Molecule ChEMBL ID'].tolist()
        results['Molecule ChEMBL ID'] = c_id
    if 'Molecule Name' in df.columns:  # save name if found in the original dataset
        name = df['Molecule Name'].tolist()
        results['Molecule Name'] = name

    # organising from the best prediction to the worst
    results = results.sort_values(['Predicted activity class', 'Prediction probability'], ascending=[False, False])
    results.reset_index(drop=True, inplace=True)

    return results


if __name__ == '__main__':
    print('')
    print('IDO1-AI-Classifier Version 0.1  \n')
    print('')

    path = input("Enter the path to the dataset: ")
    res = main(path)

    print('')
    save = input("Do you want to save the results? (y/n): ")

    if save == 'y':
        save_path = input("\n Enter the path to save the dataset: ")
        file_name = input("\n Enter the file name with the extension (e.g. .csv, .xlsx, .txt): ")
        res.to_csv(save_path + '/' + file_name)

    if save == 'n':
        pd.set_option('display.max_columns', None)
        print(res)
