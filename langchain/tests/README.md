# Testscript

Testvragen toevoegen aan `langchain\tests\testvragen.csv` of locatie.

Test script draaien en analyseren resultaten:

```shell
source ./venv/bin/activate # activeer de virtuele omgeving
python3 langchain/tests/test_model.py --file langchain/tests/testvragen.csv # analyseer de antwoorden
python3 langchain/tests/report.py # controleer de resultaten
```
