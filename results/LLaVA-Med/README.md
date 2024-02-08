# Run MultiMedEval on LLaVA-Med

The LLaVA-Med folder acts as a standalone evaluation script for LLaVA-Med. It contains the script necessary to run MultiMedEval.

The user should define the MultiMedEval config file as explained in the main README of the MultiMedEval repo.
The user should define the path to store the LLama-7b checkpoint and a path to store the LLaVA-Med checkpoint after applying the delta to LLaVA.

The user should install all the dependencies mentioned in the requirements.txt file.

Running the benchmark.py file will do the setup and evaluation on the model.