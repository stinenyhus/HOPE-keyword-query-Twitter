"""
This script runs the hope analysis pipeline once a week

"""
import schedule
import time
import os

# define the job: "run the defined bash script from terminal"
def job():
    os.system('bash run_queries.sh')

# schedule the job to run every Friday at 7 in the morning
schedule.every().friday.at("07:00").do(job)

# run the ongoing script to check every hour if it's 7 am yet
while True:
    schedule.run_pending()
    time.sleep(3600) # 60 seconds * 60 mintues = 1 hour