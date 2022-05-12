1) extract_localizations.sh: File to run the python script "test.py" for getting the localizations from tator. It gets all
the localizations and stores in some tmp folder. I am moving them to our required folder using the second command and renaming
to a jpg file.

2) Tator: The directory contains files that I am using to deal with tator. I will be using another codebase for the regression model

3) test.py: This python file is the file that is running on all localizations inside the project. I am planning to make the code modular.

4) images: Contains all the images for the localizations for image regression

5)make_summary.py: Contains the code for extracting the localizations from a particular section of the tator app.

10 May 3:09 PM we have 4922 annotations

May 12 1:38 PM

Not all annotations are for the fish. Some annotations are also for tally and individuals. We have about 3324 fill annotations.