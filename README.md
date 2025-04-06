import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------Intents Data -------------------
intents = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["Hi", "Hello", "Hey", "Good morning", "Good evening"],
            "responses": ["Hello! Welcome to Stationery Express.", "Hi there! Looking for some stationery?", "Hey! How can I help you with your supplies today?"]
        },
        {
            "tag": "browse_products",
            "patterns": ["Show me pens", "I want to see notebooks", "Do you have markers?", "What items are available?", "Can I see your products?"],
            "responses": ["Sure! We have pens, pencils, notebooks, markers, and much more.", "Absolutely! You can browse all items in the 'Products' section.", "Yes, we have a wide variety of stationery. What are you looking for specifically?"]
        },
        {
            "tag": "place_order",
            "patterns": ["I want to buy a notebook", "How can I place an order?", "I need 10 pens", "Order 5 diaries", "Can I book a marker?"],
            "responses": ["You can place an order directly from our website. Just add the items to your cart and proceed to checkout.", "Sure, add the items to your cart and complete the payment to place an order.", "Got it! Add the items to your cart to start your booking."]
        },
        {
            "tag": "order_status",
            "patterns": ["Where is my order?", "Track my delivery", "Order status", "Has my item shipped?", "When will my delivery arrive?"],
            "responses": ["You can track your order in the 'My Orders' section.", "Let me check... Please enter your order ID.", "Your order should be arriving soon! Check 'My Orders' for live updates."]
        },
        {
            "tag": "return_policy",
            "patterns": ["Can I return items?", "Return policy", "What if I don’t like the product?", "Can I get a refund?"],
            "responses": ["Yes, we have a 7-day return policy. Items must be unused and in original condition.", "You can request a return from your order history within 7 days of delivery.", "We offer refunds for eligible returns. Please check our policy page for more info."]
        },
        {
            "tag": "store_hours",
            "patterns": ["What are your working hours?", "When do you open?", "Are you open on weekends?", "What time do you close?"],
            "responses": ["We’re open from 9 AM to 9 PM, Monday to Saturday.", "Our store hours are 9 AM to 9 PM.", "We’re closed on Sundays, but our website is open 24/7!"]
        },
        {
            "tag": "thanks",
            "patterns": ["Thanks", "Thank you", "That’s helpful", "Appreciate it"],
            "responses": ["You're welcome!", "Happy to help!", "Anytime! Let us know if you need anything else."]
        },
        {
            "tag": "goodbye",
            "patterns": ["Bye", "See you later", "Goodbye"],
            "responses": ["Goodbye!", "Have a great day!", "See you next time at Stationery Express!"]
        },
        {
            "tag": "noanswer",
            "patterns": [],
            "responses": ["Sorry, I didn't understand that.", "Can you rephrase?", "I'm not sure I follow. Try asking about our products or help."]
        }
    ]
}

# -------------------Preprocessing -------------------
sentences = []
labels = []
responses = {}

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        sentences.append(pattern)
        labels.append(intent["tag"])
    responses[intent["tag"]] = intent["responses"]

tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
padded_sequences = pad_sequences(sequences, padding='post')

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

vocab_size = len(tokenizer.word_index) + 1
max_len = padded_sequences.shape[1]
num_classes = len(set(labels_encoded))

# -------------------Dataset and Model -------------------
class ChatDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.x_data = torch.tensor(padded_sequences, dtype=torch.long)
        self.y_data = torch.tensor(labels_encoded, dtype=torch.long)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

class ChatModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(ChatModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = torch.mean(x, dim=1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

dataset = ChatDataset()
loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=8, shuffle=True)

model = ChatModel(vocab_size=vocab_size, embed_dim=16, num_classes=num_classes)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ------------------- Training -------------------
for epoch in range(300):
    for words, labels_batch in loader:
        optimizer.zero_grad()
        output = model(words)
        loss = loss_fn(output, labels_batch)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

# -------------------Chat Function -------------------
def predict_class(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    input_tensor = torch.tensor(padded, dtype=torch.long)
    with torch.no_grad():
        output = model(input_tensor)
        predicted = torch.argmax(output, dim=1).item()
    return label_encoder.inverse_transform([predicted])[0]

def chat():
    print("🤖 Stationery Chatbot is ready! Type 'quit' to exit.")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            print("Bot: Goodbye!")
            break
        tag = predict_class(inp)
        print("Bot:", random.choice(responses[tag]))

chat()
