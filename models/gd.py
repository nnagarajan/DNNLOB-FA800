import logging as log
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm


class GradientDescent():
    def __init__(self, device):
        self.device = device

    # A function to encapsulate the training loop
    def batch(self, model, criterion, optimizer, train_loader, test_loader, model_savepoint, epochs,patience=20):
        train_losses = np.zeros(epochs)
        test_losses = np.zeros(epochs)
        best_test_loss = np.inf
        best_test_epoch = 0
        # counter for epochs without improvement
        no_improve_epochs = 0

        for it in tqdm(range(epochs)):

            model.train()
            t0 = datetime.now()
            train_loss = []
            for inputs, targets in train_loader:
                # move data to GPU
                inputs, targets = inputs.to(self.device, dtype=torch.float), targets.to(self.device, dtype=torch.int64)
                # print("inputs.shape:", inputs.shape)
                # zero the parameter gradients
                optimizer.zero_grad()
                # Forward pass
                # print("about to get model output")
                outputs = model(inputs)
                # print("done getting model output")
                # print("outputs.shape:", outputs.shape, "targets.shape:", targets.shape)
                loss = criterion(outputs, targets)
                # Backward and optimize
                # print("about to optimize")
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())
            # Get train loss and test loss
            train_loss = np.mean(train_loss)  # a little misleading

            model.eval()
            test_loss = []
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(self.device, dtype=torch.float), targets.to(self.device, dtype=torch.int64)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    test_loss.append(loss.item())
                test_loss = np.mean(test_loss)

            # Save losses
            train_losses[it] = train_loss
            test_losses[it] = test_loss

            if test_loss < best_test_loss:
                torch.save(model.state_dict(), f'.{model_savepoint}')
                best_test_loss = test_loss
                best_test_epoch = it
                print('model saved')
                log.info(f"model saved {model_savepoint}")
                no_improve_epochs = 0  # reset counter
            else:
                no_improve_epochs += 1

            dt = datetime.now() - t0
            print(f'Epoch {it + 1}/{epochs}, Train Loss: {train_loss:.4f}, \
              Validation Loss: {test_loss:.4f}, Duration: {dt}, Best Val Epoch: {best_test_epoch}')
            log.info(f'Epoch {it + 1}/{epochs}, Train Loss: {train_loss:.4f}, \
              Validation Loss: {test_loss:.4f}, Duration: {dt}, Best Val Epoch: {best_test_epoch}')


            # Early stopping check
            if patience is not None and no_improve_epochs >= patience:
                print(f"Early stopping triggered at epoch {it + 1} "
                      f"(no improvement in val loss for {patience} epochs).")
                log.info(f"Early stopping at epoch {it + 1}, "
                         f"best val epoch: {best_test_epoch}, best val loss: {best_test_loss:.4f}")
                # truncate the loss arrays to the number of completed epochs
                train_losses = train_losses[:it + 1]
                test_losses = test_losses[:it + 1]
                break

        return train_losses, test_losses

    def evaulate_model(self, model_savepoint, model, test_loader):
        state = torch.load(f'.{model_savepoint}', weights_only=True)
        model.load_state_dict(state)
        model.eval()
        all_targets = []
        all_predictions = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                # Move to GPU
                inputs, targets = inputs.to(self.device, dtype=torch.float), targets.to(self.device, dtype=torch.int64)

                # Forward pass
                outputs = model(inputs)

                # Get prediction
                # torch.max returns both max and argmax
                _, predictions = torch.max(outputs, 1)

                all_targets.append(targets.cpu().numpy())
                all_predictions.append(predictions.cpu().numpy())
        all_targets = np.concatenate(all_targets)
        all_predictions = np.concatenate(all_predictions)
        return all_targets, all_predictions
