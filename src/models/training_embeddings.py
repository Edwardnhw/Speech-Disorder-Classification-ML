# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
import torch
from transformers import AutoFeatureExtractor, WhisperModel, WhisperProcessor
from transformers import pipeline
import torchaudio
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier

from torchinfo import summary
from sklearn.preprocessing import LabelEncoder, StandardScaler


# %%
processor = WhisperProcessor.from_pretrained("openai/whisper-base")
model = WhisperModel.from_pretrained("openai/whisper-base")

def process_file(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    waveform = torch.mean(waveform, dim=0, keepdim=False)
    #waveform = torchaudio.transforms.Vad(sample_rate=16000)(waveform).squeeze(0)
    
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
   
    
    with torch.no_grad():
        encoder_outputs = model.encoder(inputs.input_features).last_hidden_state
    #print(encoder_outputs.shape)
    return encoder_outputs

process_file("data/voc-als-data-wav/CT001_phonationE.wav").shape

# %%
all_df = pd.read_csv("edward_uppercase_acoustic_features.csv", sep=',',  decimal=',')
print(all_df['Dataset'].unique())
print(all_df['label'].unique())
print(all_df['Phoneme'].unique())
#all_df = all_df.sample(frac=1).reset_index(drop=True) # shuffle

# %%
files = all_df['voiced_file_path']
embeddings = []
for file in files:
    embeddings.append(process_file(file))
embeddings_t = torch.cat(embeddings)

# %%
all_df['embedding_idx'] = range(len(all_df))

# %%
def train_df(df,model):
    y = df['label']
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    subjects = df['subjectID'].unique()
    train_subjects, test_subjects = train_test_split(subjects, test_size=0.3, random_state=0)
    X = embeddings_t.mean(dim=1).detach().numpy()
    X_train, X_test, y_train, y_test = X[df['embedding_idx'][df['subjectID'].isin(train_subjects)]], X[df['embedding_idx'][df['subjectID'].isin(test_subjects)]], y[df['subjectID'].isin(train_subjects)], y[df['subjectID'].isin(test_subjects)]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))
    return accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='weighted'), {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "encoder": encoder
    }
train_df(all_df, LogisticRegression(max_iter=10000, class_weight='balanced'))[0]

# %%
df = all_df[(all_df['Phoneme'] == 'A') & (all_df['label'].isin(['HC', 'ALS']))]

# %%
import seaborn as sns

def test_model(model):
    accuracies = pd.DataFrame(columns=all_df['Phoneme'].unique(), index=all_df['label'].unique(), dtype='float')
    f1_scores = pd.DataFrame(columns=all_df['Phoneme'].unique(), index=all_df['label'].unique(), dtype='float')
    for phoneme in all_df['Phoneme'].unique():
        for label in all_df['label'].unique():
            if label == 'HC':
                continue
            print(f'Phoneme: {phoneme}, label: {label}')
            df = all_df[(all_df['Phoneme'] == phoneme) & (all_df['label'].isin(['HC', label]))]
            if df['label'].nunique() < 2:
                continue
            acc, f1, _ = train_df(df, model)
            accuracies.loc[label, phoneme] = acc
            f1_scores.loc[label, phoneme] = f1
    return accuracies, f1_scores
            
model = LogisticRegression(max_iter=10000, class_weight='balanced')
acc, f1s = test_model(model)
acc.drop(labels=['HC'], inplace=True)
f1s.drop(labels=['HC'], inplace=True)
acc

# %%
def plot_heatmap(data, title, name):
    plt.figure(figsize=(8, 6))
    disease_order = [ "PD","ALS", "MSA", "PSP"]
    vowel_order = ['A', 'E', 'I', 'O', 'U']
    data = data.loc[disease_order, vowel_order]
    sns.set_style(style='white') 
    sns.heatmap(data.T, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Accuracy'},
        vmin=0.5,
        vmax=1.0,
    )
    # 
    # sns.heatmap(
    #     data.T,
    #     annot=True,
    #     cmap='Blues',
    #     fmt=".2f",
    #     linewidths=0.5,
    #     cbar_kws={'label': title},
    #     annot_kws={"size": 10},
    #     vmin=0.5,
    #     vmax=1.0,
    # )
    # plt.yticks(rotation=0) 
    plt.title(title)
    plt.grid(False)
    plt.xlabel('Disease')
    plt.ylabel('Vowel')
    plt.savefig(f'plots/Embeddings_{name}.png')
    plt.savefig(f'plots/Embeddings_{name}.pdf')
    plt.show()
    
plot_heatmap(acc, 'Test Accuracy using Logistic Regression and Embeddings', 'acc_logistic')
plot_heatmap(f1s, 'F1 Score', 'f1_logistic')

# %%
model = MLPClassifier(max_iter=10000, hidden_layer_sizes=(256, 128, 64), activation='relu', alpha=0.01)
acc, f1s = test_model(model)
acc.drop(labels=['HC'], inplace=True)
f1s.drop(labels=['HC'], inplace=True)
acc

# %%
plot_heatmap(acc, 'Test Accuracy using MLP and Embeddings', 'acc_mlp')
plot_heatmap(f1s, 'F1 Score', 'f1_mlp')

# %%
als_df = all_df[(all_df['Dataset'] == 'VOC-ALS') & (all_df['Phoneme'] == 'A')]
model = LogisticRegression(max_iter=10000, class_weight='balanced')
acc, f1, info = train_df(als_df, model)
test_df = all_df[(all_df['Dataset'] == 'MINSK') & (all_df['Phoneme'] == 'A')]
test_y = info['encoder'].transform(test_df['label'])
X_test = embeddings_t.mean(dim=1).detach().numpy()[test_df['embedding_idx']]
y_pred = model.predict(X_test)
print(classification_report(test_y, y_pred, target_names=info['encoder'].classes_))

# %%
pd2df = all_df[(all_df['Dataset'] == 'PD_dataset_2') & (all_df['Phoneme'] == 'A')]
model = LogisticRegression(max_iter=10000, class_weight='balanced')
acc, f1, info = train_df(pd2df, model)
test_df = all_df[(all_df['Dataset'] == 'Italian') & (all_df['Phoneme'] == 'A')]
test_y = info['encoder'].transform(test_df['label'])
X_test = embeddings_t.mean(dim=1).detach().numpy()[test_df['embedding_idx']]
y_pred = model.predict(X_test)
print(classification_report(test_y, y_pred, target_names=info['encoder'].classes_))

# %%
from sklearn.metrics import mean_squared_error, r2_score

def train_df_regress(df,model):
    df = df.fillna({'Severity': 0})
    y = df['Severity']
    subjects = df['subjectID'].unique()
    if df.size < 10:
        return np.nan, np.nan, None
    train_subjects, test_subjects = train_test_split(subjects, test_size=0.3, random_state=0)
    X = embeddings_t.mean(dim=1).detach().numpy()
    X_train, X_test, y_train, y_test = X[df['embedding_idx'][df['subjectID'].isin(train_subjects)]], X[df['embedding_idx'][df['subjectID'].isin(test_subjects)]], y[df['subjectID'].isin(train_subjects)], y[df['subjectID'].isin(test_subjects)]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return mean_squared_error(y_test, y_pred) / len(y_test), r2_score(y_test, y_pred) / len(y_test), {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }
train_df_regress(all_df, LinearRegression())[0]

# %%

def test_model_regress(model):
    mses = pd.DataFrame(columns=all_df['Phoneme'].unique(), index=all_df['label'].unique(), dtype='float')
    r2s = pd.DataFrame(columns=all_df['Phoneme'].unique(), index=all_df['label'].unique(), dtype='float')
    for phoneme in all_df['Phoneme'].unique():
        for label in all_df['label'].unique():
            if label == 'HC':
                continue
            print(f'Phoneme: {phoneme}, label: {label}')
            df = all_df[(all_df['Phoneme'] == phoneme) & (all_df['label'].isin(['HC', label]))]
            if df['label'].nunique() < 2:
                continue
            ms, r2, _ = train_df_regress(df, model)
            mses.loc[label, phoneme] = ms
            r2s.loc[label, phoneme] = r2
    return mses, r2s
            
model = LinearRegression()
mses, r2s = test_model_regress(model)
mses.drop(labels=['HC'], inplace=True)
r2s.drop(labels=['HC'], inplace=True)
mses

# %%



