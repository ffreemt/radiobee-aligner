REM nodemon -V -w radiobee -x "sleep 3 && python -m radiobee"
REM nodemon -V -w radiobee -x python -m radiobee
REM nodemon -V -w radiobee -x py -3.8 -m radiobee
nodemon -V -w radiobee -x "run-p pyright flake8 && py -3.8 -m radiobee"
