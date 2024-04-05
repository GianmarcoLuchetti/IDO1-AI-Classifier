import NNET
import torch

# Loading NN parameters
gin_0 = NNET.GIN_0()
gin_0.load_state_dict(torch.load('nn_params/gin0_state_dict.pt'))
gin_0.eval()

gin_1 = NNET.GIN_1()
gin_1.load_state_dict(torch.load('nn_params/gin1_state_dict.pt'))
gin_1.eval()

gin_2 = NNET.GIN_2()
gin_2.load_state_dict(torch.load('nn_params/gin2_state_dict.pt'))
gin_2.eval()

gin_3 = NNET.GIN_3()
gin_3.load_state_dict(torch.load('nn_params/gin3_state_dict.pt'))
gin_3.eval()

gcn_1 = NNET.GCN_1()
gcn_1.load_state_dict(torch.load('nn_params/gcn1_state_dict.pt'))
gcn_1.eval()

gcn_2 = NNET.GCN_2()
gcn_2.load_state_dict(torch.load('nn_params/gcn2_state_dict.pt'))
gcn_2.eval()

gcn_3 = NNET.GCN_3()
gcn_3.load_state_dict(torch.load('nn_params/gcn3_state_dict.pt'))
gcn_3.eval()


def model(dataset):
    data = list(dataset['Graph'])
    pred = []  # Store predictions
    prob = []  # Store probabilities

    for sample in data:  # Iterate over each sample
        pred01 = gin_0(sample.x, sample.edge_index, sample.batch)
        # GIN predicted class
        out01 = (torch.max(torch.exp(pred01), 1)[1])
        # GIN maximum probability
        prob01 = (torch.max(torch.exp(pred01), 1)[0])

        if out01 == 0:  # Predicted class = 0, end of classification
            pred.append(out01.item())
            prob.append(prob01.item())
        elif out01 == 1:  # Predicted class = 1, classification continues
            out12_gin = gin_1(sample.x, sample.edge_index, sample.batch)
            out13_gin = gin_2(sample.x, sample.edge_index, sample.batch)
            pred12_gin = (torch.max(torch.exp(out12_gin), 1))
            pred13_gin = (torch.max(torch.exp(out13_gin), 1))
            # Average of predictions and probabilities
            out123 = (pred12_gin[1] + pred13_gin[1])/2
            prob123 = (pred12_gin[0] + pred13_gin[0])/2

            if out123 == 1:  # Predicted class = 1, end of classification
                pred.append(out123.item())
                prob.append(prob123.item())

            if out123 == 1.5:  # Average = 1.5, classification 1 Vs. 2
                prob12_gin = torch.exp(out12_gin)
                # First GCN control layer
                out12_gcn = gcn_1(sample.x, sample.edge_index, sample.batch)
                prob12_gcn = torch.exp(out12_gcn)
                prob_12 = (prob12_gin + prob12_gcn)/2
                out_12 = torch.max(prob_12, 1)[1]
                if out_12 == 1:  # Predicted class = 1, end of classification
                    pred.append(out_12.item())
                    prob.append((torch.max(prob_12, 1)[0]).item())
                elif out_12 == 2:  # Predicted class = 2, classification 2 Vs. 3
                    # Predictions from the third classification layer
                    out23_gin = gin_3(sample.x, sample.edge_index, sample.batch)
                    prob23_gin = torch.exp(out23_gin)
                    out23_gcn = gcn_3(sample.x, sample.edge_index, sample.batch)
                    prob23_gcn = torch.exp(out23_gcn)
                    prob_23 = (prob23_gin + prob23_gcn)/2
                    out_23 = (torch.max(prob_23, 1))
                    pred.append(out_23[1].item())
                    prob.append((torch.max(prob_23, 1))[0].item())

            if out123 == 2:  # Average = 2, classification 1 Vs. 3
                prob13_gin = torch.exp(out13_gin)
                # Second GCN control layer
                out13_gcn = gcn_2(sample.x, sample.edge_index, sample.batch)
                prob13_gcn = torch.exp(out13_gcn)
                prob_13 = (prob13_gin + prob13_gcn)/2
                out_13 = torch.max(prob_13, 1)[1]
                if out_13 == 1:  # Predicted class = 1, end of classification
                    pred.append(out_13.item())
                    prob.append((torch.max(prob_13, 1)[0]).item())
                elif out_13 == 3:  # Predicted class = 3, classification 2 Vs. 3
                    out23_gin = gin_3(sample.x, sample.edge_index, sample.batch)
                    prob23_gin = torch.exp(out23_gin)
                    out23_gcn = gcn_3(sample.x, sample.edge_index, sample.batch)
                    prob23_gcn = torch.exp(out23_gcn)
                    prob_23 = (prob23_gin + prob23_gcn)/2
                    out_23 = (torch.max(prob_23, 1))
                    pred.append(out_23[1].item())
                    prob.append((torch.max(prob_23, 1))[0].item())

            if out123 == 2.5:  # Average = 2.5, classification 2 Vs. 3
                out23_gin = gin_3(sample.x, sample.edge_index, sample.batch)
                prob23_gin = torch.exp(out23_gin)
                out23_gcn = gcn_3(sample.x, sample.edge_index, sample.batch)
                prob23_gcn = torch.exp(out23_gcn)
                prob_23 = (prob23_gin + prob23_gcn)/2
                out_23 = (torch.max(prob_23, 1))
                pred.append(out_23[1].item())
                prob.append((torch.max(prob_23, 1))[0].item())

    return pred, prob