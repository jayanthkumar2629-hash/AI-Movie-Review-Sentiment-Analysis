from transformers import pipeline

model_path = "./models/distilbert-sentiment"

classifier = pipeline(
    "sentiment-analysis",
    model=model_path,
    tokenizer=model_path
)


def predict_sentiment(review):
    result = classifier(review)[0]

    label = result["label"]
    confidence = result["score"]

    return label, confidence
    print(predict_sentiment("The movie was absolutely fantastic and amazing!"))