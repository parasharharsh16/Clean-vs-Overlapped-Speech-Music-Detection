from dataloader import dataloader, SignalDataset
from utils import train, evaluate, calculate_metrics, plot_ROC_AUC_Curve
from param import (
    hyper_parameters as hp,
    model_path_1,
    targ_dict as out_dict,
    train_model as bool_train_model,
    data_dir,
    sample_universe_size,
    dataset_path,
    data_file,
    plot_output_folder,
)
import gc
from model_architecture import MtlCascadeModel
from torch.optim import Adam, SGD
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import os
from dataloader import dataloader, SignalDataset, prepare_data
from torch.utils.data import ConcatDataset

# from torch.optim.lr_scheduler import ExponentialLR


def load_train_test():
    if os.path.exists(data_file):
        combined_dataset = torch.load(data_file)
    else:
        # Prepare data
        combination_paths = prepare_data(dataset_path)
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
    print("Initializing the model\n")
    model = MtlCascadeModel(hp)  # without weight

    if bool_train_model:
        # model training, comment if only evaluation is needed
        model = training(model, device, train_loader, hp)
        model.cpu()
        # with weight
    else:
        # Load the state dict onto the model
        state_dict = torch.load(model_path_1)
        model.load_state_dict(state_dict)

    # evaluation for trained model
    (
        su_predictions,
        mu_predictions,
        smr_predictions,
        su_target,
        mu_target,
        smr_target,
    ) = evaluate(model, test_loader, device)

    for item in ["speech", "music", "mixed"]:
        if item == "speech":
            predictions = su_predictions
            targets = su_target
        elif item == "music":
            predictions = mu_predictions
            targets = mu_target
        else:
            predictions = smr_predictions
            targets = smr_target
        precision, recall, f1_score, accuracy = calculate_metrics(predictions, targets)
        print(f"Precision for {item}:", precision)
        print(f"Recall for {item}:", recall)
        print(f"F1 score for {item}:", f1_score)
        print(f"Accuracy for {item}:", accuracy)
        print(f"\nCreating ROC AUC curve for class: {item}")
        plot_ROC_AUC_Curve(predictions, targets, item, plot_output_folder)

        print("-------------------------------------------------\n")
