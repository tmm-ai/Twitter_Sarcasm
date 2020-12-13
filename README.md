# Text Classification Competition: Twitter Sarcasm Detection Evaluation

Please look at "CS410 Final Project: Text Classification / Twitter Sarcasm.pdf" for all documentation deliverables and if you want details about our code and training. This README.md is a brief summary on how to evaluate our model results. This assumes you are running our model and not training your own model to validate our results, since training your own model may give different results due to changes in initialization. More details about how to evaluate model results if anything is not clear may be present in pdf mentioned above.

We provided our answer.txt in this repo. If you add this to livedatalab, you can confirm our rank on the leaderboard (under username tmmai). If you want to evaluate our results by having the model predict on the test set, then there are two options you can take: run our notebook on google colab or clone this repo and run our python scripts. 

### Run on Google Colab
1. Download our model from here: https://drive.google.com/drive/folders/1raCd-g4chwuufv1JBDNXyNxITJDMk_K2. 
2. Get our train.jsonl and test.jsonl files and place them in your google drive in subdirectory called data i.e. drive/MyDrive/data/train.jsonl
3. Run through our notebook, https://colab.research.google.com/drive/1zcMMw8xe6vh9rMPlBB_i-HZGrxsk5UJj#scrollTo=Qbr6a4LYcl_w, but skip training step. Instead, load our model into your drive and run on validation and test set. Then, submit answer.txt from last cell to livedatalab.

Demo video / code walkthrough of Colab verion: https://youtu.be/OUu71EapO5U

### Run with python scripts
1. Clone this repo and cd into it
2. Run pip install --upgrade -r requirements.txt (make sure you are using python3)
3. Download our model from here: https://drive.google.com/drive/folders/1raCd-g4chwuufv1JBDNXyNxITJDMk_K2. Place these files in a folder called roberta_model within the data subdirectory i.e. data/roberta_model
4. Run python cs410_FINAL_EVAL.py. This code will evaluate on validation set and print f1 score. It will also write in the data folder a answer.txt file with predictions on tests 
