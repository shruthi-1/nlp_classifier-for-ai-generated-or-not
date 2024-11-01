import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('.venv/data.csv')
df['label'] = df['label'].map({'CG': 1, 'OG': 0})
texts = df['text_'].astype(str).values
labels = df['label'].values
df.dropna(subset=['label'], inplace=True)
texts = df['text_'].astype(str).values
labels = df['label'].values

tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
tokenizer.fit_on_texts(texts)
max_len = 120
random_seed = 42
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, train_size=3000, test_size=1000, random_state=random_seed, stratify=labels
)
X_train_seq = tokenizer.texts_to_sequences(train_texts)
X_test_seq = tokenizer.texts_to_sequences(test_texts)
X_train_padded = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
X_test_padded = pad_sequences(X_test_seq, maxlen=max_len, padding='post')

model = Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=200),
    GlobalAveragePooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
epochs = 10
batch_size = 32
history = model.fit(
    X_train_padded, train_labels,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_test_padded, test_labels),
    verbose=1
)

def predict_ai_generated(text):
    text_seq = tokenizer.texts_to_sequences([text])
    text_padded = pad_sequences(text_seq, maxlen=max_len, padding='post')
    prediction = model.predict(text_padded)[0][0]
    return "AI-generated" if prediction >= 0.5 else "Human-written"

new_text = "how does one rate this? i don't know. But I like the way he described the two stores. I think they had great chemistry "
print("Prediction:", predict_ai_generated(new_text))
