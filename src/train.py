import copy
import os
import time

import torch
import wandb
from tqdm import tqdm

WANDB_PROJECT = "huggingface-aiornot"
WANDB_ENTITY = "astockman"


def train_model(
    run_name,
    model,
    dataloaders,
    criterion,
    optimizer,
    train_params,
    scheduler=None,
    output_preds=False,
    device=None,
    seed=None,
    enable_logs=True,
    save_dir=None,
):
    time_start = time.time()

    if enable_logs:
        wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, tags=[])
        wandb.run.name = run_name + "-" + wandb.run.id
        wandb.config.update(
            {
                "training": train_params,
                "seed": seed,
                "run_mode": run_name,
            }
        )
        wandb.watch(model, criterion=criterion)

    val_acc_history = []
    preds_cache = {
        "train": {},
        "valid": {},
    }

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in tqdm(range(train_params["epochs"])):
        for phase in ["train", "valid"]:
            if phase not in dataloaders:
                continue

            outputs = _model_loop(
                phase,
                model=model,
                dataloader=dataloaders[phase],
                optimizer=optimizer,
                criterion=criterion,
                scheduler=scheduler,
                output_preds=output_preds,
                train_params=train_params,
                epoch=epoch,
                device=device,
                enable_logs=enable_logs,
            )

            if output_preds:
                epoch_loss, epoch_acc, epoch_preds = outputs
                preds_cache[phase].update(epoch_preds)
            else:
                epoch_loss, epoch_acc = outputs

            if phase == "train" and scheduler:
                scheduler.step()

            if enable_logs:
                wandb.log({
                    f"{phase}/loss": epoch_loss,
                    f"{phase}/accuracy": epoch_acc,
                    f"{phase}/epoch": epoch + 1,
                })

            if phase == "train":
                pass
            else:
                val_acc_history.append(epoch_acc)

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - time_start
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:4f}")

    if enable_logs:
        wandb.finish()

    # load best model weights
    model.load_state_dict(best_model_wts)

    if save_dir:
        torch.save(model.state_dict(), os.path.join(save_dir, run_name))

    if output_preds:
        return model, preds_cache
    else:
        return model


def train_loop(**kwargs):
    return _model_loop(phase="train", **kwargs)


def eval_loop(**kwargs):
    return _model_loop(phase="valid", **kwargs)


def _model_loop(
    phase,
    model,
    dataloader,
    criterion,
    optimizer=None,
    scheduler=None,
    train_params=None,
    output_preds=False,
    epoch=0,
    device=None,
    enable_logs=False,
):
    if phase == "train":
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    running_corrects = 0
    running_samples = 0
    batch_num = len(dataloader)
    preds_cache = {}

    for batch_idx, (image_idx, inputs, labels) in tqdm(enumerate(dataloader), total=batch_num, leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)

        if phase == "train":
            optimizer.zero_grad()

        with torch.set_grad_enabled(phase == "train"):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            if phase == "train":
                loss.backward()
                optimizer.step()

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        running_samples += inputs.size(0)

        if output_preds:
            preds_cache.update({idx: pred for idx, pred in zip(image_idx, preds.tolist())})

        if phase == "train" and enable_logs:
            wandb.log(
                {
                    f"{phase}/loss": running_loss / running_samples,
                    f"{phase}/learning_rate": scheduler.get_last_lr()
                    if scheduler
                    else train_params["learning_rate"],
                    f"{phase}/epoch": epoch + batch_idx / batch_num,
                }
            )

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = (running_corrects.double() / len(dataloader.dataset)).item()

    if output_preds:
        return epoch_loss, epoch_acc, preds_cache
    else:
        return epoch_loss, epoch_acc
