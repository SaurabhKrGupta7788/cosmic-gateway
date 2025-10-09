
from django.shortcuts import render
from django.http import JsonResponse
import joblib
import xgboost as xgb
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import io
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import copy
from django.conf import settings
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt

import base64


# --- 1. GLOBAL SETUP: Load data and models ONCE ---

models_dir = os.path.join(settings.BASE_DIR, "main", "models")
rf_model_path = os.path.join(models_dir, "random_forest_model.pkl")
scaler_path = os.path.join(models_dir, "scaler.pkl")
xgb_model_path = os.path.join(models_dir, "xgb_best_model.json")
best_model_path = os.path.join(models_dir, "best_model.pth")

rf_model = joblib.load(rf_model_path)
loaded_scaler = joblib.load(scaler_path)

bst = xgb.Booster()
bst.load_model(xgb_model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- DATA LOADING (runs once) ---
df = pd.read_csv('C:\\Users\\100ra\\OneDrive\\Desktop\\Ram\\Nasa_prjct\\Nasa_prjct\\main\\KOI.csv')
df = df.drop(df[df.koi_disposition == 'FALSE POSITIVE'].index, axis=0).reset_index(drop=True)
df['koi_disposition'] = df['koi_disposition'].map({"CANDIDATE":1, "CONFIRMED":0})
X = df.iloc[:,1:].dropna()
y = df.loc[X.index].iloc[:,0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_scaled = loaded_scaler.transform(X_train)
X_test_scaled = loaded_scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.long)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

print("\nData prepared for training!")

# --- 2. MODEL DEFINITIONS and HELPER FUNCTIONS ---

class NeuralNet(nn.Module):
    def __init__(self, in_neurons=13, num_hidden_layers=2, hidden_neurons=128, output_dim=1, dropout=0.5):
        super().__init__()
        self.input = nn.Linear(in_neurons, hidden_neurons)
        layer = []
        for i in range(num_hidden_layers):
            layer.append(nn.Linear(hidden_neurons, hidden_neurons))
            layer.append(nn.ReLU())
            layer.append(nn.Dropout(dropout))
        self.layer = nn.Sequential(*layer)
        self.output = nn.Linear(hidden_neurons, output_dim)

    def forward(self, x):
        x = torch.relu(self.input(x))
        x = self.layer(x)
        x = self.output(x)
        return x

# Load the pre-trained model for the main prediction page
main_prediction_model = NeuralNet()
main_prediction_model.load_state_dict(torch.load(best_model_path, map_location=device))
main_prediction_model.eval()
main_prediction_model.to(device)


def train_new_model(n_hidden_layer, num_neurons, d_rate, learning_rate):
    """This function now only handles the training loop."""
    model_to_train = NeuralNet(
        in_neurons=13,
        num_hidden_layers=n_hidden_layer,
        hidden_neurons=num_neurons,
        output_dim=1,
        dropout=d_rate
    ).to(device)
    
    optimizer = torch.optim.Adam(model_to_train.parameters(), lr=learning_rate)

    neg = np.sum(y_train.to_numpy() == 0)
    pos = np.sum(y_train.to_numpy() == 1)
    pos_weight = torch.tensor([neg / (pos + 1e-12)], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    n_epochs = 500
    patience = 15
    best_accuracy = 0.0
    counter = 0
    best_model_state = None
    test_accuracy_history = []
    
    from sklearn.metrics import accuracy_score

    for i in range(n_epochs):
        model_to_train.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device).float().view(-1)
            outputs = model_to_train(x).squeeze()
            loss = criterion(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model_to_train.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device).long()
                outputs = model_to_train(x).squeeze()
                preds = (torch.sigmoid(outputs) >= 0.5).long()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        
        te_acc = accuracy_score(all_labels, all_preds) * 100
        test_accuracy_history.append(te_acc)
        
        if te_acc > best_accuracy:
            best_accuracy = te_acc
            best_model_state = copy.deepcopy(model_to_train.state_dict())
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {i}")
                break

    return best_model_state, test_accuracy_history, best_accuracy, i

def ensemble_predict(x, model, rf_model, xgb_model, device, weights=(0.6, 0.8, 0.6)):
    w1, w2, w3 = weights
    x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        nn_output = model(x_tensor).squeeze()
        nn_probs = torch.sigmoid(nn_output).cpu().numpy()
    rf_probs = rf_model.predict_proba(x)[:, 1]
    xgb_dmatrix = xgb.DMatrix(x)
    xgb_probs = xgb_model.predict(xgb_dmatrix)
    final_probs = (w1 * nn_probs + w2 * rf_probs + w3 * xgb_probs) / (w1 + w2 + w3)
    final_preds = (final_probs >= 0.5).astype(int)
    print(final_probs)
    return final_preds, final_probs

def main(request):
    return render(request, 'main/index.html')

def k2(request):
    return render(request, 'main/k2.html')

# def tune_model_view(request):
#     return render(request, 'main/train.html')       

def about(request):
    return render(request, 'main/about.html')   

def how_it_works(request):
    return render(request, 'main/how_it_works.html')


# In your views.py

# Add JsonResponse to your imports
from django.http import HttpResponse, JsonResponse 
import json # You might need this for some cases, good to have


# Add your other simple views here (k2, about, etc.)

def predict_koi(request):
    # This view uses your main, globally-loaded model
    if request.method == 'POST':
        try:
            koi_period = float(request.POST.get('koi_period'))
            koi_impact = float(request.POST.get('koi_impact'))
            koi_duration = float(request.POST.get('koi_duration'))
            koi_depth = float(request.POST.get('koi_depth'))
            koi_prad = float(request.POST.get('koi_prad'))
            koi_sma = float(request.POST.get('koi_sma'))
            koi_teq = float(request.POST.get('koi_teq'))
            koi_insol = float(request.POST.get('koi_insol'))
            koi_model_snr = float(request.POST.get('koi_model_snr'))
            koi_steff = float(request.POST.get('koi_steff'))
            koi_slogg = float(request.POST.get('koi_slogg'))
            koi_srad = float(request.POST.get('koi_srad'))
            koi_kepmag = float(request.POST.get('koi_kepmag'))

            # Create array and scale (this part is the same)
            x = np.array([koi_period, koi_impact, koi_duration, koi_depth, koi_prad, koi_sma, koi_teq, koi_insol, koi_model_snr, koi_steff, koi_slogg, koi_srad, koi_kepmag])
            x_reshaped = x.reshape(1, -1)
            x_scaled = loaded_scaler.transform(x_reshaped)

            pred, prob = ensemble_predict(x_scaled, main_prediction_model, rf_model, bst, device)
            
            prediction_label = 'POTENTIAL CANDIDATE' if pred[0] == 1 else 'CONFIRMED EXOPLANET'
            probability_score = f"{prob[0] * 100:.2f}%"

            return JsonResponse({
                'status': 'success',
                'prediction': prediction_label,
                'probability': probability_score
            })
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=405)


def tune_model_view(request):
    """
    This view handles the training request from train.html
    """
    if request.method == 'POST':
        try:
            # Get hyperparameters from the form
            n_hidden_layer = int(request.POST.get('n_hidden_layer'))
            num_neurons = int(request.POST.get('num_neurons'))
            d_rate = float(request.POST.get('d_rate'))
            learning_rate = float(request.POST.get('learning_rate'))

            # Train a new model with these parameters
            best_model_state, test_acc_history, best_acc, last_epoch = train_new_model(
                n_hidden_layer, num_neurons, d_rate, learning_rate
            )

            # --- Save the temporary model to the user's session ---
            buffer = io.BytesIO()
            torch.save(best_model_state, buffer)
            model_bytes = buffer.getvalue()
            
            # 1. Encode the bytes into a Base64 string
            model_base64 = base64.b64encode(model_bytes).decode('utf-8')
            
            # 2. Save the string to the session
            request.session['tuned_model_state_b64'] = model_base64 
            request.session['tuned_model_params'] = {
                'num_hidden_layers': n_hidden_layer,
                'hidden_neurons': num_neurons,
                'dropout': d_rate
            }

            # --- Generate the accuracy plot ---
            plt.figure(figsize=(10, 8))
            plt.plot(test_acc_history, label='Test Accuracy', color='orange')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy (%)')
            plt.title('Custom Model Test Accuracy')
            plt.legend()
            plt.grid(True)
            
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', bbox_inches='tight')
            img_buffer.seek(0)
            plot_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            plt.close()

            return JsonResponse({
                'status': 'success',
                'results': {
                    'final_accuracy': f"{test_acc_history[-1]:.2f}" if test_acc_history else "N/A",
                    'best_accuracy': f"{best_acc:.2f}",
                    'last_epoch': last_epoch + 1,
                },
                'plot_url': plot_base64
            })
            
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)

    # For GET requests, just show the page
    return render(request, 'main/train.html' )

# main/views.py

def predict_koi_train(request):
    """
    This view handles predictions using the temporary, session-stored model.
    """
    if request.method == 'POST':
        # --- FIX #1: Check for the correct session key ---
        if 'tuned_model_state_b64' not in request.session:
            return JsonResponse({'status': 'error', 'message': 'Please train a model first.'}, status=400)
        
        try:
            params = request.session['tuned_model_params']
            
            # --- FIX #2: Retrieve data using the correct session key ---
            model_base64 = request.session['tuned_model_state_b64']
            model_bytes = base64.b64decode(model_base64)
            
            temp_model = NeuralNet(
                num_hidden_layers=params['num_hidden_layers'],
                hidden_neurons=params['hidden_neurons'],
                dropout=params['dropout']
            ).to(device)
            
            buffer = io.BytesIO(model_bytes)
            temp_model.load_state_dict(torch.load(buffer, map_location=device))
            temp_model.eval()

            # Extract 13 KOI features from request.POST
            x = np.array([
                float(request.POST.get('koi_period')),
                float(request.POST.get('koi_impact')),
                float(request.POST.get('koi_duration')),
                float(request.POST.get('koi_depth')),
                float(request.POST.get('koi_prad')),
                float(request.POST.get('koi_sma')),
                float(request.POST.get('koi_teq')),
                float(request.POST.get('koi_insol')),
                float(request.POST.get('koi_model_snr')),
                float(request.POST.get('koi_steff')),
                float(request.POST.get('koi_slogg')),
                float(request.POST.get('koi_srad')),
                float(request.POST.get('koi_kepmag'))
            ])
            x_scaled = loaded_scaler.transform(x.reshape(1, -1))

            pred, prob = ensemble_predict(x_scaled, temp_model, rf_model, bst, device)

            prediction_label = 'POTENTIAL CANDIDATE' if pred[0] == 1 else 'CONFIRMED EXOPLANET'
            probability_score = f"{prob[0] * 100:.2f}%"

            return JsonResponse({
                'status': 'success',
                'prediction': prediction_label,
                'probability': probability_score
            })

        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)

    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=405)