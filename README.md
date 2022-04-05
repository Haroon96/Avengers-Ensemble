# Avengers Ensemble!
Code-release for "Avengers Ensemble! Improving transferability of authorship obfuscation".

Adapted from [Mutant-X](https://github.com/asad1996172/Mutant-X) by Mahmood et al.

## Getting Started
1. Clone this repository.
2. Download the word embeddings from [here](https://www.dropbox.com/sh/y3srrf82n9jbx8x/AAAlHlICEftupAJ3WZnS8W3Aa?dl=0) and extract them to the `common` directory.
3. Install the required libraries using `pip install -r requirements.txt`.

## Training the classifier
- See `classifier/Ensemble.py`.

## Running the obfuscator
-  See `obfuscator/Obfuscator.py`.

## Featured datasets
- Amazon Mechanical Turk with 3, 5, and 10 authors (`amt-3`, `amt-5`, `amt-10`)
- Blogs dataset with 5 and 10 authors (`BlogsAll-5`, `BlogsAll-10`)
