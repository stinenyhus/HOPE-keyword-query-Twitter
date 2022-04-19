echo "Starting pipeline on all queries at $(date '+%d-%m %H:%M')"

# Vaccine
echo "Starting pipeline on vaccin at $(date '+%d-%m %H:%M')"
bash pipeline.sh -k vaccin &> logs/vaccin.log 
echo "Finished pipeline on vaccin)"

# Corona
echo "Starting pipeline on corona at $(date '+%d-%m %H:%M')"
bash pipeline.sh -k corona &> logs/corona.log
echo "Finished pipeline on corona"

# Omicron English tweets
echo "Starting pipeline on denmark (i.e., English tweets) at $(date '+%d-%m %H:%M')"
bash pipeline.sh -k denmark &> logs/denmark.log &
echo "Finished pipeline on denmark (i.e., English tweets)"

# Coronapas
echo "Starting pipeline on coronapas at $(date '+%d-%m %H:%M')"
bash pipeline.sh -k coronapas &> logs/coronapas.log
echo "Finished pipeline on coronapas"

# Mundbind
echo "Starting pipeline on mundbind at $(date '+%d-%m %H:%M')"
bash pipeline.sh -k mundbind &> logs/mundbind.log 
echo "Finished pipeline on mundbind"

# Mette Frederiksen
echo "Starting pipeline on mettef at $(date '+%d-%m %H:%M')"
bash pipeline.sh -k mettef &> logs/mettef.log
echo "Finished pipeline on mettef"

# Restriktion
echo "Starting pipeline on restriktion at $(date '+%d-%m %H:%M')"
bash pipeline.sh -k restriktion &> logs/restriktion.log
echo "Finished pipeline on restriktion"

# Genåbning
echo "Starting pipeline on genåb at $(date '+%d-%m %H:%M')"
bash pipeline.sh -k genåb &> logs/genåb.log
echo "Finished pipeline on genåb"

# Tvang
echo "Starting pipeline on tvang at $(date '+%d-%m %H:%M')"
bash pipeline.sh -k tvang &> logs/tvang.log 
echo "Finished pipeline on tvang"

# Samfundskritisk
echo "Starting pipeline on samfundskri at $(date '+%d-%m %H:%M')"
bash pipeline.sh -k samfundskri &> logs/samfundskri.log 
echo "Finished pipeline on samfundskri"

# Omicron
echo "Starting pipeline on omicron (Danish tweets) at $(date '+%d-%m %H:%M')"
bash pipeline.sh -k omicron &> logs/omicron.log 
echo "Finished pipeline on omicron (Danish tweets)"

# Lockdown
echo "Starting pipeline on lockdown at $(date '+%d-%m %H:%M')"
bash pipeline.sh -k lockdown &> logs/lockdown.log 
echo "Finished pipeline on lockdown"

# Pressemøde
echo "Starting pipeline on pressemøde at $(date '+%d-%m %H:%M')"
bash pipeline.sh -k pressemøde &> logs/pressemøde.log 
echo "Finished pipeline on pressemøde"

# # 17 januar
# bash pipeline.sh -k 17jan &> logs/17jan_$(date '+%d%m').log 
# echo "Finished running pipeline on 17jan at $(date '+%d-%m %H:%M')"

# Daily proportion
echo "Starting calculating daily proportion using keywords from 'selvtest'"
bash daily_proportion_pipeline.sh -k selvtest &> logs/daily_proportion.log
echo "Finished daily proportion pipeline"

echo "Finished pipeline on all queries at $(date '+%d-%m %H:%M')"
echo 