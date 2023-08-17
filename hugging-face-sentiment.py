from transformers import pipeline

## Sentiment Analysis Pre-trained one

classifier = pipeline("sentiment-analysis")
result = classifier("I've been waiting for a HuggingFace course my whole life.")
print(result)

## Sentiment Analysis Pre-trained two
result = classifier(
    ["I've been waiting for a HuggingFace course my whole life.", "I hate this so much!"]
)
print(f"Sentiment Scores: {result}")