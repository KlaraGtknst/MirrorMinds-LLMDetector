import pandas as pd

# Sample data with an 'Essay' column
data = {
    "Essay": [
        "Technology has drastically changed the way we communicate in the 21st century.",
        "Climate change poses a serious threat to the environment and human life.",
        "Education plays a vital role in shaping the future of young individuals.",
        "The rise of social media has influenced political discourse globally.",
        "Healthcare systems need to be more accessible and equitable for all citizens.",
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to .pkl file
df.to_pickle("Datasets/sample_essays.pkl")

print("'sample_essays.pkl' created with 5 sample essays.")
