# W210_Capstone

## Setup:

### Server Side:
1. Create a server. Currently I am using AWS t2.micro with Ubuntu 16.04.
1. Install python on the server. Current scripts use python3.
    1.  I prefer [Anaconda](https://www.anaconda.com/download), it includes numerous libraries and tools to manage and control dev environments.
1. Follow the linked instructions to install [MongoDB](https://docs.mongodb.com/getting-started/shell/tutorial/install-mongodb-on-ubuntu/)
1. Run `pip install fuzzywuzzy python-Levenshtein names` from terminal
    1.  May also need to run `sudo apt-get install gcc`
1. Run `conda install pymongo` from terminal
1. Copy `cosmetics_app.py` to your designated folder on the server
1. Copy `ewg_ingredients.json` to the same folder
1. Run `python3 cosmetics_app.py` will start the server for running or testing the app
    1. Use `tmux` to keep the server running in case the session ends.

### Populate Database:
1. Run `./load_data_to_mongo.py -bg` to populate the database with ewg data and auto generate some users.
    1. This requires `ewg_ingredients.json`, `ewg_products.json`, and `comodegenic.json` to be in the same folder as `load_data_to_mongo.py`.
    1. Running `load_data_to_mongo.py` with the -h flag displays other useful options for DB operations.

### Application Side:
1. Install MIT App Inventor per instruction on both your local machine and testing phone.
1. Import the project from App_Inventor folder to the MIT App Inventor
1. To connect to phone, click Connect -> AI Companion and it will pop up a bar code and a code. Open app inventor app from your phone and either scan bar code or put in the code from your phone. 

