from dataloader import dataloader, SignalDataset
from utils import prepare_data
from param import (
    dataset_path,
    sample_universe_size,
    data_dir,
    hyper_parameters as hp,
    model_1,
)
from torch.utils.data import ConcatDataset
from MTL_w_cascade_info import MtlCascadeModel
from torch.optim import Adam  # , SGD
from torch.optim.lr_scheduler import ExponentialLR
import torch
import os
from torch import nn

data_file = os.path.join(data_dir, "dataset.pth")
if "dataset.pth" in os.listdir(data_dir):
    combined_dataset = torch.load(data_file)
else:
    # Prepare data
    combination_paths = prepare_data(dataset_path)
    sampled_df = combination_paths.sample(
        frac=sample_universe_size, random_state=42, ignore_index=True
    )
    datasets_music = SignalDataset(sampled_df, data_type="music")
    datasets_speech = SignalDataset(sampled_df, data_type="speech")
    datasets_mixture = SignalDataset(sampled_df, data_type="mixture")
    combined_dataset = ConcatDataset(
        [datasets_music, datasets_speech, datasets_mixture]
    )
    torch.save(combined_dataset, data_file)


print("data_loader")
train_loader, test_loader = dataloader(
    datasets=combined_dataset,
    train_ratio=hp["train_ratio"],
    train_batch_size=hp["batch_size"],
    test_batch_size=1,
)


def train(
    train_loader,
    model,
    epoch,
    out_dict,
    loss_sp_fn,
    loss_mu_fn,
    loss_smr_fn,
    optimizer,
):
    correct = 0
    for data in train_loader:
        feature, label = data
        y = [out_dict[x] for x in label]
        out_sp, out_mu, out_smr = model(feature)

        sp_list = [inner_list[0] for inner_list in y]
        mu_list = [inner_list[1] for inner_list in y]
        smr_list = [inner_list[2:] for inner_list in y]
        y_sp = torch.Tensor(sp_list).unsqueeze(1)
        y_mu = torch.Tensor(mu_list).unsqueeze(1)
        y_smr = torch.Tensor(smr_list).unsqueeze(1)

        loss_sp = loss_sp_fn(out_sp, y_sp)
        loss_mu = loss_mu_fn(out_mu, y_mu)
        loss_smr = loss_smr_fn(out_smr, y_smr)

        total_loss = loss_sp + loss_mu + loss_smr
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        pred_y = torch.cat((out_sp, out_mu, out_smr), dim=1)
        result = (pred_y > 0.5).float()
        target = torch.tensor(y).float()
        for i in range(result.size(0)):
            if torch.all(torch.eq(result[i], target[i])):
                correct += 1
    accuracy = correct / len(train_loader.dataset)
    print(
        f"Epoch: {epoch}, Loss_sp: {loss_sp}, Loss_mu: {loss_mu}, Loss_smr: {loss_smr}, Accuracy: {accuracy}"  # noqa
    )


print("model_init")
model = MtlCascadeModel(hp)
# print(model)
loss_sp = nn.BCEWithLogitsLoss()
loss_mu = nn.BCEWithLogitsLoss()
loss_smr = nn.MSELoss()

out_dict = {"speech": [1, 0, 0, 0], "music": [0, 1, 0, 0], "mixture": [0, 0, 1, 1]}

# Optimizer and learning rate scheduler
optimizer = Adam(model.parameters(), lr=0.002)
# optimizer = SGD(model.parameters(), lr=0.002, momentum=0.9)
scheduler = ExponentialLR(optimizer, gamma=0.1)

print("start_training")
for epoch in range(1, hp["n_epochs"] + 1):
    train(
        train_loader, model, epoch, out_dict, loss_sp, loss_mu, loss_smr, optimizer
    )  # noqa

torch.save(model.state_dict(), model_1)
# torch.save(model, f"{model_1}.model")