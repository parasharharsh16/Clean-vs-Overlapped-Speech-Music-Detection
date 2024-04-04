from dataloader import dataloader, SignalDataset
from utils import (train,evaluate,calculate_metrics,plot_ROC_AUC_Curve,
train_classical_model,evaluate_ml_clf,classical_classification,prepare_data)

from param import (
    hyper_parameters as hp,
    model_path_1,
    targ_dict as out_dict,
    train_model as bool_train_model,
    data_dir,
    sample_universe_size,
    dataset_path,data_file, 
    plot_output_folder,
    classical_model_path_rf,
    classical_model_path_svm,
    combination_file_path,
    eval_file
)
import gc
from model_architecture import MtlCascadeModel
from torch.optim import Adam, SGD
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import os
import pandas as pd
from dataloader import dataloader, SignalDataset
from torch.utils.data import ConcatDataset
from joblib import load,dump

# from torch.optim.lr_scheduler import ExponentialLR


def load_train_test():
    if os.path.exists(data_file):
        combined_dataset = torch.load(data_file)
        combination_paths = prepare_data(dataset_path,combination_file_path)
        sampled_df = combination_paths.sample(
            frac=sample_universe_size, random_state=42, ignore_index=True
        )
    else:
        # Prepare data
        combination_paths = prepare_data(dataset_path,combination_file_path)
        sampled_df = combination_paths.sample(
            frac=sample_universe_size, random_state=42, ignore_index=True
        )
        print("Creating datasets")
        for item in ["speech", "music", "mixture"]:
            indiv_datafile = f"{data_dir}/dataset_{item}_{sample_universe_size}.pth"
            if os.path.exists(indiv_datafile):
                print(f"{item} dataset already exists")
                continue
            print(f"Creating {item} dataset")
            dataset = SignalDataset(sampled_df, data_type=item)
            torch.save(dataset, f"{data_dir}/dataset_{item}_{sample_universe_size}.pth")
            del dataset
            gc.collect()
        # Create a combined dataset
        datasets_music = torch.load(
            f"{data_dir}/dataset_music_{sample_universe_size}.pth"
        )
        datasets_speech = torch.load(
            f"{data_dir}/dataset_speech_{sample_universe_size}.pth"
        )
        datasets_mixture = torch.load(
            f"{data_dir}/dataset_mixture_{sample_universe_size}.pth"
        )

        combined_dataset = ConcatDataset(
            [datasets_music, datasets_speech, datasets_mixture]
        )
        del datasets_music, datasets_speech, datasets_mixture
        torch.save(combined_dataset, data_file)
        print("Saved the combined dataset\n")
    return combined_dataset


def training(model, device, train_loader, hp):
    # Move model to GPU if available
    model.to(device)

    print("Training the model\n")
    # Defining Loss functions
    loss_sp = nn.BCEWithLogitsLoss()
    loss_mu = nn.BCEWithLogitsLoss()
    loss_smr = nn.BCEWithLogitsLoss()

    # Optimizer and learning rate scheduler
    optimizer = Adam(model.parameters(), lr=0.002)

    # optimizer = SGD(model.parameters(), lr=0.002, momentum=0.9)
    # scheduler = ExponentialLR(optimizer, gamma=0.1)

    for epoch in range(1, hp["n_epochs"] + 1):
        train(
            train_loader,
            model,
            epoch,
            out_dict,
            loss_sp,
            loss_mu,
            loss_smr,
            optimizer,
            device,
        )  # noqa

    torch.save(model.state_dict(), model_path_1)
    return model


if __name__ == "__main__":
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Enable cuDNN if available
    if device.type == "cuda":
        cudnn.benchmark = True

    print("Using device:", device)

    # load combination dataset of music, speech and mixture
    combined_dataset = load_train_test()

    train_loader, test_loader = dataloader(
        datasets=combined_dataset,
        train_ratio=hp["train_ratio"],
        train_batch_size=hp["batch_size"],
        test_batch_size=1,
    )

    # Create an instance of your model
    print("Initializing the MTL model\n")
    model = MtlCascadeModel(hp) # without weight

    if bool_train_model:
        # model training, comment if only evaluation is needed
        model = training(model, device, train_loader, hp)
        model.cpu()
        classical_model_rm = train_classical_model(train_loader,model_type="rf")
        classical_model_svm = train_classical_model(train_loader,model_type="svm")
        #with weight

    else:
        # Load the state dict onto the model
        state_dict = torch.load(model_path_1)
        model.load_state_dict(state_dict)
        classical_model_svm = load(classical_model_path_svm)
        classical_model_rm = load(classical_model_path_rf)
        
    
    # Evaluate the model
    print("Evaluating the MTL model\n")
    su_predictions, mu_predictions, smr_predictions, su_target, mu_target, smr_target = evaluate(model, test_loader,device)
    print("Evaluating the classical models\n")
    su_predictions_svm, mu_predictions_svm, smr_predictions_svm, su_target_svm, mu_target_svm, smr_target_svm = evaluate_ml_clf(test_loader,classical_model_svm)
    su_predictions_rf, mu_predictions_rf, smr_predictions_rf, su_target_rf, mu_target_rf, smr_target_rf = evaluate_ml_clf(test_loader,classical_model_rm)
    print("Evaluating the classical thresolding model\n")
    su_predictions_classical, mu_predictions_classical, smr_predictions_classical, su_target_classical, mu_target_classical, smr_target_classical = classical_classification(combination_file_path)
    results = []
    for item in ["speech","music","mixed"]:
        if item == "speech":
            predictions_mtl = su_predictions
            targets_mtl = su_target
            predictions_rf = su_predictions_rf
            targets_rf = su_target_rf
            predictions_svm = su_predictions_svm
            targets_svm = su_target_svm
            precision_classical = su_predictions_classical
            targets_classical = su_target_classical
        elif item == "music":
            predictions_mtl = mu_predictions
            targets_mtl = mu_target
            predictions_rf = mu_predictions_rf
            targets_rf = mu_target_rf
            predictions_svm = mu_predictions_svm
            targets_svm = mu_target_svm
            precision_classical = mu_predictions_classical
            targets_classical = mu_target_classical
        else:
            predictions_mtl = smr_predictions
            targets_mtl = smr_target
            predictions_rf = smr_predictions_rf
            targets_rf = smr_target_rf
            predictions_svm = smr_predictions_svm
            targets_svm = smr_target_svm
            precision_classical = smr_predictions_classical
            targets_classical = smr_target_classical

        print(f"-----------------------Classical Thresolding--------------------------\n")
        precision, recall, f1_score,accuracy  = calculate_metrics(precision_classical, targets_classical)
        print(f"Creating ROC AUC curve for class: {item}")
        plot_ROC_AUC_Curve(predictions_mtl,targets_mtl,f'Thresold Model {item}',plot_output_folder)
        results.append(['Thresold Model', item, precision, recall, f1_score, accuracy])

        print("-----------------------MTL DL Model--------------------------\n")
        precision, recall, f1_score,accuracy  = calculate_metrics(predictions_mtl, targets_mtl)
        print(f"Creating ROC AUC curve for class: {item}")
        plot_ROC_AUC_Curve(predictions_mtl,targets_mtl,f'MTL Model {item}',plot_output_folder)
        results.append(['MTL Model', item, precision, recall, f1_score, accuracy])

        print("-----------------------Classical ML Model: SVM--------------------------\n")
        precision, recall, f1_score,accuracy  = calculate_metrics(predictions_svm, targets_svm)
        print(f"Creating ROC AUC curve for class: {item}")
        plot_ROC_AUC_Curve(predictions_mtl,targets_mtl,f'SVM Model {item}',plot_output_folder)
        results.append(['SVM', item, precision, recall, f1_score, accuracy])
        print("-----------------------Classical ML Model: Random Forest--------------------------\n")
        precision, recall, f1_score,accuracy  = calculate_metrics(predictions_rf, targets_rf)
        print(f"Creating ROC AUC curve for class: {item}")
        plot_ROC_AUC_Curve(predictions_mtl,targets_mtl,f'Random Forest Model {item}',plot_output_folder)
        results.append(['Random Forest', item, precision, recall, f1_score, accuracy])
   
    df = pd.DataFrame(results, columns=['Model', 'Item', 'Precision', 'Recall', 'F1 Score', 'Accuracy'])
    print(df)
    df.to_csv(eval_file, index=False)
