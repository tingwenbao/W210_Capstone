# W210_Capstone

## Setup:

### Server Side:
1. Create a server. Currently I am using AWS t2.micro with Ubuntu 16.04.
1. Install python on the server. Current scripts use python3.
	1. 	I prefer [Anaconda](https://www.anaconda.com/download), it include numerous libraries and tools to manage and control dev environments.
1. Copy the `cosmetics_app.py` to your designated folder on the server
1. Copy the `ewg_ingredients.json` to the same folder
1. 
1. Install [MongoDB](https://docs.mongodb.com/getting-started/shell/tutorial/install-mongodb-on-ubuntu/) the using linked instructions
1. Run `python3 cosmetics_app.py` will start the server for running or testing the app

### Application Side:
1. Install MIT App Inventor per instruction on both your local machine and testing phone.
1. Import the project from App_Inventor folder to the MIT App Inventor
1. To connect to phone, click Connect -> AI Companion and it will pop up a bar code and a code. Open app inventor app from your phone and either scan bar code or put in the code from your phone. 

