from transformers import AutoModelForSequenceClassification

# Load the pre-trained model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

# Access the dropout layer inside the model
dropout_layer = model.bert.encoder.layer[0].output.dropout


print("Dropout rate:", dropout_layer.p)

print(dropout_layer)