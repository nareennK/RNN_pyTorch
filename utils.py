import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def load_and_preprocess_data(file_path):

    try:
        df = pd.read_csv(file_path)
        print("Dataset loaded successfully.")

        df["Date"] = df['Month'].astype(str) + "-" + df["Year"].astype(str)
        df["Date"] = pd.to_datetime(df["Date"], format="%m-%Y")
        df = df.set_index("Date")
        df = df.drop(["Year", "Month", "Day"], axis=1)
        print("Date column processed and indexed.")

        if df.isnull().sum().any():
            print("Missing values detected:")
            print(df.isnull().sum())
            df = df.fillna(df.mean())
            print("Missing values filled with column means.")
        else:
            print("No missing values detected.")

       
        df = (df - df.min()) / (df.max() - df.min())
        print("Data normalized using Min-Max scaling.")
        return df
    
    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")
        return None



def split_data(df):

    try:
        features = df[["Specific Humidity", "Relative Humidity", "Temperature"]].values
        target = df["Precipitation"].values

        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42, shuffle=False)
        print("Data split into train and test sets.")

        return X_train, X_test, y_train, y_test
    
    except Exception as e:
        print(f"An error occurred during data splitting: {e}")


def create_dataloaders(X_train, y_train, X_test, y_test, batch_size=4):

    try:
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

        train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)
        print("DataLoaders created successfully.")
        return train_loader, test_loader
    
    except Exception as e:
        print(f"An error occurred while creating DataLoaders: {e}")


class RNNModel(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out


def train_model(model, train_loader, criterion, optimizer, epochs=1000):

    try:
        epoch_losses = []  
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.view(X_batch.size(0), 1, -1)  
                optimizer.zero_grad()
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

           
            epoch_losses.append(epoch_loss / len(train_loader))

            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}")

        print("Model training completed.")
        

        plt.plot(range(epochs), epoch_losses, label='Training Loss', color='red')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss over Epochs')
        plt.grid(True)
        plt.show()

    except Exception as e:
        print(f"An error occurred during model training: {e}")



def evaluate_model(model, test_loader):
    
    try:
        model.eval()
        with torch.no_grad():
            test_predictions = []
            actuals = []
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.view(X_batch.size(0), 1, -1)
                predictions = model(X_batch)
                test_predictions.extend(predictions.numpy())
                actuals.extend(y_batch.numpy())

        test_predictions = torch.tensor(test_predictions).view(-1).numpy()
        actuals = torch.tensor(actuals).view(-1).numpy()

        mse = mean_squared_error(actuals, test_predictions)
        mae = mean_absolute_error(actuals, test_predictions)
        r2 = r2_score(actuals, test_predictions)

        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"R-squared (R2): {r2:.4f}")

        return mse, mae, r2
    
    except Exception as e:
        print(f"An error occurred during model evaluation: {e}")