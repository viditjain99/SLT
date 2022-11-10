# SLT

# install dependencies
pip install -r requirements.txt

# create data files in json format 
python utils/jsonize.py --dataset aslg --mode train dev test --src en --tgt gloss.asl


