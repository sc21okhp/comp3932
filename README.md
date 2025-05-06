# SC21OKHP comp3932  
GitHub Repository for SC21OKHP COMP3932 Synoptic Project (EPA) (38376)  
The report associated with this project can be found uploaded to GradeScope
  
# Folder Structure  
* dataset - This folder contains the 3 tables from the MIMIC-III-10k dataset used for the project  
* results - This folder contains CSV files with the results obtained from testing the models  
* versions - This folder contains the code  
  *  CategoricalClassifiers - This folder contains the code to run the categorical classifiers  
  *  ResultVisuals - This folder contains the code to run the graphs generated from the results  
  *  SurvivalClassifiers - This folder contains the code to run the survival classifiers  

# Usage
A requirements plaintext document can be found in the versions folder that contains all the python libraries needed to run the project. You may wish to run this project in a virtual environment, if so, please create one now. To install these libraries please navigate to the versions folder and run:  
```pip install -r requirements.txt```  

Please run this command to ensure you have the correct dependencies installed.  
  
A version control plaintext document can be found in the versions folder that contains all the information about the different versions of the project. Please select the version you wish to run.  
  
The latest version can be found by looking for the largest integer X after the V followed by the largest integer Y after the period such that VX.Y.py is a valid file name.  
  
In order to run the file please navigate to the correct folder and type the following command in the command line:  
```python v<X>.<Y>.py```

Replacing X and Y with the chosen integers.  
