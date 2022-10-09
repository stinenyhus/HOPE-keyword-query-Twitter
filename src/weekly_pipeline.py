"""
This script runs the hope analysis pipeline once a week

"""
import schedule
import time
import os
from datetime import date

# define the job: "run the defined bash script from terminal"
def job():
    if date.today().day == 1:  # only do job on the first of the month
        os.system("bash run_queries.sh")


# schedule the job to run every Friday at 7 in the morning
schedule.every().saturday.at("16:50").do(job)

# run the ongoing script to check every hour if it's 7 am yet
while True:
    schedule.run_pending()
    time.sleep(1)  # 60 seconds * 60 mintues = 1 hour
