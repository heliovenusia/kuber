1) install Docker Desktop for Windows AMD64 from https://www.docker.com/products/docker-desktop/
2) start Docker Desktop. Ensure "Engine is running" message from Docker Desktop Terminal. Else, wait for it to start and run.
3) download kuber folder to preferred location on target machine
4) run Delete_Client_Docker_Setup.bat
5) run Setup_Client.bat
6) you may now delete Delete_Client_Docker_Setup.bat and Setup_Client.bat from targer machine. this should not be triggered in future.
7) Start_Kuber.bat and Stop_Kuber.bat are the daily-use batch files for the end-user

-----
Training is done on 3 years data. Need not retrain as of now, we will decide after seeing performance in production. 
*** Retraining now will cause loss of 3 years of training data. Incremental training is kept disabled to specifically gauge performance ***

-----
