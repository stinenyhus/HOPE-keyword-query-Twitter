echo "Starting pipeline on all queries at $(date '+%d-%m %H:%M')"

# Vaccine
DATE=$(date '+%d%m') # Get current date
bash pipeline.sh -k vaccin &> logs/vaccin_$(date '+%d%m').log 
echo "Finished running pipeline on vaccin at $(date '+%d-%m %H:%M')"

# Corona
DATE=$(date '+%d%m') # Get current date
bash pipeline.sh -k corona &> logs/corona_$(date '+%d%m').log
echo "Finished running pipeline on corona at $(date '+%d-%m %H:%M')"

# Omicron English tweets
DATE=$(date '+%d%m') # Get current date
bash pipeline.sh -k denmark &> logs/denmark_$(date '+%d%m').log &
echo "Finished running pipeline on denmark (i.e., English tweets) at $(date '+%d-%m %H:%M')"

# Coronapas
DATE=$(date '+%d%m') # Get current date
bash pipeline.sh -k coronapas &> logs/coronapas_$(date '+%d%m').log
echo "Finished running pipeline on coronapas at $(date '+%d-%m %H:%M')"

# Mundbind
DATE=$(date '+%d%m') # Get current date
bash pipeline.sh -k mundbind &> logs/mundbind_$(date '+%d%m').log 
echo "Finished running pipeline on mundbind at $(date '+%d-%m %H:%M')"

# Mette Frederiksen
DATE=$(date '+%d%m') # Get current date
bash pipeline.sh -k mettef &> logs/mettef_$(date '+%d%m').log
echo "Finished running pipeline on mettef at $(date '+%d-%m %H:%M')"

# Restriktion
DATE=$(date '+%d%m') # Get current date
bash pipeline.sh -k restriktion &> logs/restriktion_$(date '+%d%m').log
echo "Finished running pipeline on restriktion at $(date '+%d-%m %H:%M')"

# Genåbning
DATE=$(date '+%d%m') # Get current date
bash pipeline.sh -k genåb &> logs/genåb_$(date '+%d%m').log
echo "Finished running pipeline on genåb at $(date '+%d-%m %H:%M')"

# Tvang
DATE=$(date '+%d%m') # Get current date
bash pipeline.sh -k tvang &> logs/tvang_$(date '+%d%m').log 
echo "Finished running pipeline on tvang at $(date '+%d-%m %H:%M')"

# Samfundskritisk
DATE=$(date '+%d%m') # Get current date
bash pipeline.sh -k samfundskri &> logs/samfundskri_$(date '+%d%m').log 
echo "Finished running pipeline on samfundskri at $(date '+%d-%m %H:%M')"

# Omicron
DATE=$(date '+%d%m') # Get current date
bash pipeline.sh -k omicron &> logs/omicron_$(date '+%d%m').log 
echo "Finished running pipeline on omicron (Danish tweets) at $(date '+%d-%m %H:%M')"

# Lockdown
bash pipeline.sh -k lockdown &> logs/lockdown_$(date '+%d%m').log 
echo "Finished running pipeline on lockdown at $(date '+%d-%m %H:%M')"

# Pressemøde
bash pipeline.sh -k pressemøde &> logs/pressemøde_$(date '+%d%m').log 
echo "Finished running pipeline on pressemøde at $(date '+%d-%m %H:%M')"

# 17 januar
bash pipeline.sh -k 17jan &> logs/17jan_$(date '+%d%m').log 
echo "Finished running pipeline on 17jan at $(date '+%d-%m %H:%M')"