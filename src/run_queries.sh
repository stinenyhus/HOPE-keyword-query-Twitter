DATE=1802

# Vaccine
bash pipeline.sh -k vaccin &> logs/vaccin_${DATE}.log 
echo "Finished running pipeline on vaccin"

# Corona
bash pipeline.sh -k corona &> logs/corona_${DATE}.log
echo "Finished running pipeline on corona"

# Omicron English tweets
bash pipeline.sh -k denmark &> logs/denmark_${DATE}.log &
echo "Finished running pipeline on denmark (i.e., English tweets)"

# Coronapas
bash pipeline.sh -k coronapas &> logs/coronapas_${DATE}.log
echo "Finished running pipeline on coronapas"

# Mundbind
bash pipeline.sh -k mundbind &> logs/mundbind_${DATE}.log 
echo "Finished running pipeline on mundbind"

# Mette Frederiksen
bash pipeline.sh -k mettef &> logs/mettef_${DATE}.log
echo "Finished running pipeline on mettef"

# Restriktion
bash pipeline.sh -k restriktion &> logs/restriktion_${DATE}.log
echo "Finished running pipeline on restriktion"

# Genåbning
bash pipeline.sh -k genåb &> logs/genåb_${DATE}.log
echo "Finished running pipeline on genåb"

# Tvang
bash pipeline.sh -k tvang &> logs/tvang_${DATE}.log 
echo "Finished running pipeline on tvang"

# Samfundskritisk
bash pipeline.sh -k samfundskri &> logs/samfundskri_${DATE}.log 
echo "Finished running pipeline on samfundskri"

# Omicron
bash pipeline.sh -k omicron &> logs/omicron_${DATE}.log 
echo "Finished running pipeline on omicron (Danish tweets)"

# Lockdown
bash pipeline.sh -k lockdown &> logs/lockdown_${DATE}.log 
echo "Finished running pipeline on lockdown"

# Pressemøde
bash pipeline.sh -k pressemøde &> logs/pressemøde_${DATE}.log 
echo "Finished running pipeline on pressemøde"

# 17 januar
bash pipeline.sh -k 17jan &> logs/17jan_${DATE}.log 
echo "Finished running pipeline on 17jan"